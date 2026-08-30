//! Thin GPU buffer + shape/dtype. No autodiff, no graph.
//!
//! Phase 4: optional byte offset for bank/slice views (no host round-trip),
//! and GPU blit copy for deep_copy (no host memcpy).
//!
//! Audit 4 P0: pooled buffers are Arc-owned; last drop schedules cold recycle +
//! `removeAllocation` after the in-flight CB completes (see [`GpuRuntime`]).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use std::sync::{Arc, Weak};

use crate::runtime::{BufferKind, GpuRuntime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    BF16,
}

impl DType {
    pub fn size_of(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 => 2,
        }
    }
}

/// Checked storage size shared by allocation, views, and kernel validation.
pub(crate) fn checked_nbytes(shape: &[usize], dtype: DType) -> Result<usize, String> {
    let n = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| "tensor element count overflow".to_string())?;
    n.checked_mul(dtype.size_of()).filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| "tensor byte size overflow".to_string())
}

/// Shared Metal buffer with recycle / residency policy.
pub(crate) struct PooledBuffer {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) nbytes: usize,
    pub(crate) kind: BufferKind,
    pub(crate) runtime: Weak<GpuRuntime>,
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Bump views share the slab; Hot weights stay resident for the run.
        // Only Cold pool temps are recycled + removed from residency after sync.
        if self.kind != BufferKind::Cold {
            return;
        }
        let Some(rt) = self.runtime.upgrade() else {
            return;
        };
        // Keep the MTLBuffer alive until after CB completion via pending queue.
        let buffer = self.buffer.clone();
        let nbytes = self.nbytes;
        rt.schedule_cold_recycle(buffer, nbytes);
    }
}

/// Owning handle to a shared-memory Metal buffer plus logical shape.
#[derive(Clone)]
pub struct GpuBuffer {
    pub(crate) inner: Arc<PooledBuffer>,
}

impl GpuBuffer {
    pub fn nbytes(&self) -> usize {
        self.inner.nbytes
    }

    pub fn metal(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.inner.buffer
    }

    pub(crate) fn kind(&self) -> BufferKind {
        self.inner.kind
    }

    /// Host pointer into unified memory (StorageModeShared).
    pub fn contents_f32(&self) -> &mut [f32] {
        assert_eq!(self.inner.nbytes % 4, 0);
        let ptr = self.inner.buffer.contents().as_ptr() as *mut f32;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.inner.nbytes / 4) }
    }

    pub fn contents_u16(&self) -> &mut [u16] {
        assert_eq!(self.inner.nbytes % 2, 0);
        let ptr = self.inner.buffer.contents().as_ptr() as *mut u16;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.inner.nbytes / 2) }
    }

    pub fn write_f32(&self, data: &[f32]) {
        let dst = self.contents_f32();
        assert_eq!(dst.len(), data.len());
        dst.copy_from_slice(data);
    }

    /// Write into the leading `data.len()` floats (buffer may be oversized scratch).
    pub fn write_f32_prefix(&self, data: &[f32]) {
        let dst = self.contents_f32();
        assert!(
            data.len() <= dst.len(),
            "write_f32_prefix: data {} > buf {}",
            data.len(),
            dst.len()
        );
        dst[..data.len()].copy_from_slice(data);
    }

    pub fn read_f32(&self) -> Vec<f32> {
        self.contents_f32().to_vec()
    }

    pub fn write_bf16_bits(&self, data: &[u16]) {
        let dst = self.contents_u16();
        assert_eq!(dst.len(), data.len());
        dst.copy_from_slice(data);
    }

    pub fn contents_u8(&self) -> &mut [u8] {
        let ptr = self.inner.buffer.contents().as_ptr() as *mut u8;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.inner.nbytes) }
    }

    pub fn write_bytes(&self, data: &[u8]) {
        let dst = self.contents_u8();
        assert!(data.len() <= dst.len());
        dst[..data.len()].copy_from_slice(data);
    }

    pub fn contents_u32(&self) -> &mut [u32] {
        assert_eq!(self.inner.nbytes % 4, 0);
        let ptr = self.inner.buffer.contents().as_ptr() as *mut u32;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.inner.nbytes / 4) }
    }

    pub fn write_u32(&self, data: &[u32]) {
        let dst = self.contents_u32();
        assert_eq!(dst.len(), data.len());
        dst.copy_from_slice(data);
    }

    pub fn read_u32(&self) -> Vec<u32> {
        self.contents_u32().to_vec()
    }

    pub fn zero(&self) {
        let ptr = self.inner.buffer.contents().as_ptr() as *mut u8;
        unsafe { std::ptr::write_bytes(ptr, 0, self.inner.nbytes) };
    }
}

/// Logical tensor: shape + dtype over a GpuBuffer (row-major, contiguous view).
#[derive(Clone)]
pub struct Tensor {
    pub buffer: GpuBuffer,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Byte offset into `buffer` for bank / slice views.
    pub byte_offset: usize,
    pub(crate) runtime: Arc<GpuRuntime>,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes_logical(&self) -> usize {
        self.numel() * self.dtype.size_of()
    }

    pub fn runtime(&self) -> &Arc<GpuRuntime> {
        &self.runtime
    }

    /// View into the same buffer at an element offset (same dtype).
    pub fn view(&self, shape: &[usize], elem_offset: usize) -> Tensor {
        self.validate().expect("invalid source view");
        let nbytes = checked_nbytes(shape, self.dtype).expect("view size overflow");
        let off = elem_offset.checked_mul(self.dtype.size_of())
            .and_then(|off| self.byte_offset.checked_add(off)).expect("view offset overflow");
        assert!(off.checked_add(nbytes).is_some_and(|end| end <= self.buffer.nbytes()),
            "view out of bounds");
        Tensor {
            buffer: self.buffer.clone(),
            shape: shape.to_vec(),
            dtype: self.dtype,
            byte_offset: off,
            runtime: Arc::clone(&self.runtime),
        }
    }

    /// Validate public metadata before passing a view to a GPU kernel.
    pub(crate) fn validate(&self) -> Result<(), String> {
        let bytes = checked_nbytes(&self.shape, self.dtype)?;
        if self.byte_offset % self.dtype.size_of() != 0 ||
            !self.byte_offset.checked_add(bytes).is_some_and(|end| end <= self.buffer.nbytes()) {
            return Err("tensor view is misaligned or out of bounds".into());
        }
        if !self.buffer.inner.runtime.ptr_eq(&Arc::downgrade(&self.runtime)) {
            return Err("tensor buffer belongs to a different runtime".into());
        }
        Ok(())
    }

    pub(crate) fn overlaps(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.buffer.inner, &other.buffer.inner)
            && self.byte_offset < other.byte_offset + other.nbytes_logical()
            && other.byte_offset < self.byte_offset + self.nbytes_logical()
    }

    /// Allocate a new buffer and GPU-copy contents (encoded into the active batch).
    pub fn deep_copy(&self) -> Result<Tensor, String> {
        let t = match self.dtype {
            DType::F32 => self.runtime.alloc_tensor_f32(&self.shape)?,
            DType::BF16 => self.runtime.alloc_tensor_bf16(&self.shape)?,
        };
        gpu_copy(self, &t)?;
        Ok(t)
    }
}

/// Device blit: `dst = src` (same numel/dtype). Encodes into the active batch.
pub fn gpu_copy(src: &Tensor, dst: &Tensor) -> Result<(), String> {
    assert_eq!(src.numel(), dst.numel());
    assert_eq!(src.dtype, dst.dtype);
    let rt = src.runtime();
    let n = src.numel();
    let kernel = match src.dtype {
        DType::F32 => "copy_f32",
        DType::BF16 => "copy_bf16",
    };
    let p = rt.pipeline(kernel)?;
    crate::dispatch::dispatch_1d(rt, &p, n, |bnd| {
        crate::dispatch::set_tensor(bnd, src, 0);
        crate::dispatch::set_tensor(bnd, dst, 1);
        crate::dispatch::set_u32(bnd, n as u32, 2);
    })
}

/// Host f32 → bf16 bit pattern (round-to-nearest-even truncate).
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    // Preserve NaNs even when all payload bits would be truncated. Quiet them
    // before rounding; negative payloads can otherwise overflow the u32 sum.
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16;
    round as u16
}

pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

pub fn f32_slice_to_bf16(data: &[f32]) -> Vec<u16> {
    data.iter().copied().map(f32_to_bf16_bits).collect()
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn bf16_preserves_special_values_and_ties() {
        for bits in [0x7f800001, 0x7fffffff, 0xff800001, 0xffffffff] {
            assert!(bf16_bits_to_f32(f32_to_bf16_bits(f32::from_bits(bits))).is_nan(),
                "NaN payload {bits:x} became finite or infinity");
        }
        for bits in [0, 0x80000000, 0x7f800000, 0xff800000] {
            assert_eq!(f32_to_bf16_bits(f32::from_bits(bits)), (bits >> 16) as u16);
        }
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f808000)), 0x3f80);
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f818000)), 0x3f82);
        for b in 0..=u16::MAX {
            let x = bf16_bits_to_f32(b);
            if !x.is_nan() { assert_eq!(f32_to_bf16_bits(x), b); }
        }
    }

    #[test]
    fn allocation_rejects_overflow() {
        let rt = GpuRuntime::new().unwrap();
        assert!(rt.alloc_tensor_f32(&[usize::MAX, usize::MAX]).is_err());
        assert!(rt.alloc_tensor_bf16(&[usize::MAX, usize::MAX]).is_err());
        assert!(rt.alloc_tensor_f32_hot(&[usize::MAX / 4 + 2]).is_err());
        assert!(rt.alloc_temp_f32(&[usize::MAX, usize::MAX]).is_err());
    }

    #[test]
    #[should_panic(expected = "view")]
    fn view_rejects_offset_overflow() {
        let rt = GpuRuntime::new().unwrap();
        let t = rt.alloc_tensor_f32(&[4]).unwrap();
        let _ = t.view(&[1], usize::MAX / 4 + 1);
    }
}
