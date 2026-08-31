//! Metal 4 bind + dispatch helpers.
//!
//! [`Binder`] targets the Metal 4 argument table + const arena. Call sites use
//! `set_*` / `bind_*` sugar; [`GpuRuntime::with_binder`] opens the training
//! command buffer encoder.

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4CommandEncoder, MTL4ComputeCommandEncoder, MTL4VisibilityOptions,
    MTLBuffer, MTLComputePipelineState, MTLSize, MTLStages,
};

use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::{GpuBuffer, Tensor};

/// Metal 4 compute binder (argument table + const arena).
pub struct Binder<'a> {
    runtime: &'a GpuRuntime,
    error: Option<String>,
    max_buffers: usize,
    max_threads: Option<usize>,
    enc: &'a ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    table: &'a ProtocolObject<dyn MTL4ArgumentTable>,
    const_staging: &'a ProtocolObject<dyn MTLBuffer>,
    const_cursor: &'a mut usize,
}

impl<'a> Binder<'a> {
    pub(crate) fn finish(&self) -> Result<(), String> {
        match &self.error { Some(e) => Err(e.clone()), None => Ok(()) }
    }
    fn fail(&mut self, message: impl Into<String>) {
        if self.error.is_none() { self.error = Some(message.into()); }
    }
    fn valid_index(&mut self, index: usize) -> bool {
        if index >= self.max_buffers { self.fail("argument-table buffer index out of range"); }
        self.error.is_none()
    }
    fn write_constants(&mut self, bytes: &[u8]) -> u64 {
        if self.error.is_some() { return 0; }
        let start = self.const_cursor.checked_add(15).map(|n| n & !15);
        let end = start.and_then(|n| n.checked_add(bytes.len().max(4)));
        let (Some(start), Some(end)) = (start, end) else {
            self.fail("constant arena offset overflow"); return 0;
        };
        if bytes.is_empty() || end > self.const_staging.length() as usize {
            self.fail("constant arena exhausted or empty payload"); return 0;
        }
        // SAFETY: checked the entire destination range before writing.
        unsafe {
            let dst = self.const_staging.contents().as_ptr().cast::<u8>().add(start);
            std::ptr::write_bytes(dst, 0, bytes.len().max(4));
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
        *self.const_cursor = end;
        self.const_staging.gpuAddress() + start as u64
    }

    pub(crate) fn new(
        enc: &'a ProtocolObject<dyn MTL4ComputeCommandEncoder>,
        table: &'a ProtocolObject<dyn MTL4ArgumentTable>,
        const_staging: &'a ProtocolObject<dyn MTLBuffer>,
        const_cursor: &'a mut usize,
        max_buffers: usize,
        runtime: &'a GpuRuntime,
    ) -> Self {
        Self {
            runtime,
            error: None,
            max_buffers,
            max_threads: None,
            enc,
            table,
            const_staging,
            const_cursor,
        }
    }

    pub fn set_pipeline(&mut self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) {
        if self.error.is_some() { return; }
        self.max_threads = Some(pipeline.maxTotalThreadsPerThreadgroup() as usize);
        self.enc.setComputePipelineState(pipeline);
    }

    pub fn bind_buf(
        &mut self,
        buf: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        index: usize,
    ) {
        if !self.valid_index(index) { return; }
        if offset >= buf.length() as usize {
            self.fail("buffer binding offset out of bounds"); return;
        }
        let Some(addr) = buf.gpuAddress().checked_add(offset as u64) else {
            self.fail("GPU address overflow"); return;
        };
        unsafe {
            self.table.setAddress_atIndex(addr, index);
        }
    }

    pub fn bind_tensor(&mut self, t: &Tensor, index: usize) {
        if let Err(e) = t.validate() { self.fail(e); return; }
        if !std::ptr::eq(t.runtime().as_ref(), self.runtime) {
            self.fail("tensor belongs to another runtime"); return;
        }
        self.bind_buf(t.buffer.metal(), t.byte_offset, index);
    }

    pub fn bind_gpu_buf(&mut self, b: &GpuBuffer, index: usize) {
        if !std::ptr::eq(b.inner.runtime.as_ptr(), self.runtime) {
            self.fail("buffer belongs to another runtime"); return;
        }
        self.bind_buf(b.metal(), 0, index);
    }

    pub fn bind_bytes(&mut self, bytes: &[u8], index: usize) {
        if !self.valid_index(index) { return; }
        let addr = self.write_constants(bytes);
        if self.error.is_some() { return; }
        unsafe { self.table.setAddress_atIndex(addr, index); }
    }

    pub fn bind_u32(&mut self, v: u32, index: usize) {
        self.bind_bytes(&v.to_ne_bytes(), index);
    }

    pub fn bind_f32(&mut self, v: f32, index: usize) {
        self.bind_bytes(&v.to_ne_bytes(), index);
    }

    /// Dispatch threadgroups. Optionally inserts a Dispatch→Dispatch Device
    /// barrier after the dispatch (default on; skip via
    /// `METAL_NATIVE_HAZARD_BARRIERS=1`). Packed multi-dispatch ops that need
    /// RAW/WAR still call [`Self::barrier`] explicitly.
    pub fn dispatch(&mut self, threadgroups: MTLSize, threads_per_tg: MTLSize) {
        if self.error.is_some() { return; }
        let lanes = threads_per_tg.width.checked_mul(threads_per_tg.height)
            .and_then(|n|n.checked_mul(threads_per_tg.depth));
        let valid = lanes.zip(self.max_threads).is_some_and(|(n,max)| n>0 && n<=max)
            && [threadgroups.width,threadgroups.height,threadgroups.depth].iter()
                .all(|&n|n>0 && n<=u32::MAX as usize);
        if !valid { self.fail("invalid dispatch geometry or missing pipeline"); return; }

        self.enc.setArgumentTable(Some(self.table));
        self.enc
            .dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        if !crate::ab_flags::hazard_barriers() {
            self.enc
                .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    MTLStages::Dispatch,
                    MTLStages::Dispatch,
                    MTL4VisibilityOptions::Device,
                );
        }
    }

    /// Explicit producer→consumer barrier inside a packed encoder
    /// (Dispatch→Dispatch Device).
    pub fn barrier(&mut self) {
        if self.error.is_some() { return; }
        self.enc
            .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
    }
}

// --- Free helpers (call-site sugar) -----------------------------------------

pub fn set_tensor(bnd: &mut Binder<'_>, t: &Tensor, index: usize) {
    bnd.bind_tensor(t, index);
}

pub fn set_gpu_buf(bnd: &mut Binder<'_>, buf: &GpuBuffer, index: usize) {
    bnd.bind_gpu_buf(buf, index);
}

pub fn set_u32(bnd: &mut Binder<'_>, v: u32, index: usize) {
    bnd.bind_u32(v, index);
}

pub fn set_f32(bnd: &mut Binder<'_>, v: f32, index: usize) {
    bnd.bind_f32(v, index);
}

/// Dispatch `n` threads with automatic threadgroup sizing.
pub fn dispatch_1d(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    n: usize,
    encode_bufs: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if n == 0 {
        return Ok(());
    }
    if n > u32::MAX as usize { return Err("1D dispatch exceeds uint indexing".into()); }
    let width = pipeline.threadExecutionWidth() as usize;
    let tpt = width.min(n).max(1);
    let groups = (n + tpt - 1) / tpt;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        encode_bufs(bnd);
        bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

pub fn dispatch_2d(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    nx: usize,
    ny: usize,
    encode_bufs: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if nx == 0 || ny == 0 {
        return Ok(());
    }
    if [nx, ny].iter().any(|&n| n > u32::MAX as usize) {
        return Err("dispatch extent exceeds uint indexing".into());
    }
    let width = pipeline.threadExecutionWidth() as usize;
    let tx = width.min(nx).max(1);
    let groups_x = nx.div_ceil(tx);
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        encode_bufs(bnd);
        bnd.dispatch(mtl_size(groups_x, ny, 1), mtl_size(tx, 1, 1));
        Ok(())
    })
}

pub fn dispatch_3d(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    nx: usize,
    ny: usize,
    nz: usize,
    encode_bufs: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if nx == 0 || ny == 0 || nz == 0 {
        return Ok(());
    }
    if [nx, ny, nz].iter().any(|&n| n > u32::MAX as usize) {
        return Err("dispatch extent exceeds uint indexing".into());
    }
    let width = pipeline.threadExecutionWidth() as usize;
    let tx = width.min(nx).max(1);
    let groups_x = nx.div_ceil(tx);
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        encode_bufs(bnd);
        bnd.dispatch(mtl_size(groups_x, ny, nz), mtl_size(tx, 1, 1));
        Ok(())
    })
}

/// 2D grid of threadgroups with fixed threads-per-threadgroup (FA-2 tiles).
pub fn dispatch_2d_tg(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    groups_x: usize,
    groups_y: usize,
    threads_per_tg: usize,
    encode_bufs: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if groups_x == 0 || groups_y == 0 || threads_per_tg == 0 {
        return Ok(());
    }
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        encode_bufs(bnd);
        bnd.dispatch(
            mtl_size(groups_x, groups_y, 1),
            mtl_size(threads_per_tg, 1, 1),
        );
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binder_path_copy_via_dispatch_1d() {
        let rt = GpuRuntime::new().expect("runtime");
        let n = 32usize;
        let src = rt.alloc_buffer(n * 4).unwrap();
        let dst = rt.alloc_buffer(n * 4).unwrap();
        unsafe {
            let p = src.metal().contents().as_ptr() as *mut f32;
            for i in 0..n {
                *p.add(i) = (i as f32) * 2.0;
            }
        }
        let pipe = rt.pipeline("copy_f32").unwrap();
        dispatch_1d(&rt, &pipe, n, |bnd| {
            set_gpu_buf(bnd, &src, 0);
            set_gpu_buf(bnd, &dst, 1);
            set_u32(bnd, n as u32, 2);
        })
        .unwrap();
        rt.synchronize().unwrap();
        let out = unsafe {
            std::slice::from_raw_parts(dst.metal().contents().as_ptr() as *const f32, n)
        };
        for i in 0..n {
            assert_eq!(out[i], (i as f32) * 2.0);
        }
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    #[test]
    fn oversized_extent_rejected_before_callback() {
        let rt = GpuRuntime::new().unwrap();
        let p = rt.pipeline("copy_f32").unwrap();
        let called = std::cell::Cell::new(false);
        assert!(dispatch_2d(&rt, &p, usize::MAX, 1, |_| called.set(true)).is_err());
        assert!(!called.get(), "oversized 2D dispatch reached encoder");
        assert!(dispatch_3d(&rt, &p, 1, 1, usize::MAX, |_| called.set(true)).is_err());
        assert!(!called.get(), "oversized 3D dispatch reached encoder");
    }

    #[test]
    fn binder_rejects_foreign_runtime_storage() {
        let rt = GpuRuntime::new().unwrap();
        let other = GpuRuntime::new().unwrap();
        let t = other.alloc_tensor_f32(&[4]).unwrap();
        let map = t.buffer.contents_f32();
        assert!(rt.with_binder(|b| { b.bind_tensor(&t, 0); Ok(()) }).is_err());
        assert_eq!(map[0], 0.0);
    }

    #[test]
    fn oversized_constants_return_error_instead_of_panicking() {
        let rt=GpuRuntime::new().unwrap();
        let bytes=vec![0u8;rt.metal4.const_staging.length() as usize+1];
        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||
            rt.with_binder(|b| { b.bind_bytes(&bytes,0); Ok(()) })));
        assert!(result.is_ok(), "Result API panicked on full arena");
        assert!(result.unwrap().is_err());
    }
    #[test]
    fn binder_rejects_bad_view_without_a_dispatch() {
        let rt=GpuRuntime::new().unwrap();
        let mut t=rt.alloc_tensor_f32(&[4]).unwrap();
        t.byte_offset=usize::MAX;
        assert!(rt.with_binder(|b| { b.bind_tensor(&t,0); Ok(()) }).is_err());
    }
}
