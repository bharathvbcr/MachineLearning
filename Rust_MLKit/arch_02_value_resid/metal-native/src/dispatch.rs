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
    enc: &'a ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    table: &'a ProtocolObject<dyn MTL4ArgumentTable>,
    const_staging: &'a ProtocolObject<dyn MTLBuffer>,
    const_cursor: &'a mut usize,
    /// Auto-barrier mode latched for this binder scope (kept in sync with the
    /// metal-runtime Binder so gemm.rs stays a hand-synced copy).
    skip_auto_barriers: bool,
}

impl<'a> Binder<'a> {
    pub(crate) fn new(
        enc: &'a ProtocolObject<dyn MTL4ComputeCommandEncoder>,
        table: &'a ProtocolObject<dyn MTL4ArgumentTable>,
        const_staging: &'a ProtocolObject<dyn MTLBuffer>,
        const_cursor: &'a mut usize,
    ) -> Self {
        Self {
            enc,
            table,
            const_staging,
            const_cursor,
            skip_auto_barriers: crate::ab_flags::hazard_barriers(),
        }
    }

    /// True when this scope skips the per-dispatch auto barrier, so packed
    /// multi-dispatch ops must insert [`Self::barrier`] at their RAW edges.
    #[inline]
    pub fn needs_explicit_barriers(&self) -> bool {
        self.skip_auto_barriers
    }

    pub fn set_pipeline(&mut self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) {
        self.enc.setComputePipelineState(pipeline);
    }

    pub fn bind_buf(
        &mut self,
        buf: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        index: usize,
    ) {
        let addr = buf.gpuAddress().wrapping_add(offset as u64);
        unsafe {
            self.table.setAddress_atIndex(addr, index);
        }
    }

    pub fn bind_tensor(&mut self, t: &Tensor, index: usize) {
        self.bind_buf(t.buffer.metal(), t.byte_offset, index);
    }

    pub fn bind_gpu_buf(&mut self, b: &GpuBuffer, index: usize) {
        self.bind_buf(b.metal(), 0, index);
    }

    pub fn bind_bytes(&mut self, bytes: &[u8], index: usize) {
        let align = 16usize;
        let mut cursor = (*self.const_cursor + align - 1) & !(align - 1);
        let staging_len = self.const_staging.length() as usize;
        if cursor + bytes.len() > staging_len {
            panic!(
                "const arena exhausted: need {} at cursor {cursor} (cap {staging_len})",
                bytes.len()
            );
        }
        unsafe {
            let dst = (self.const_staging.contents().as_ptr() as *mut u8).add(cursor);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
            self.table.setAddress_atIndex(
                self.const_staging.gpuAddress() + cursor as u64,
                index,
            );
        }
        cursor += bytes.len().max(4);
        *self.const_cursor = cursor;
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
        self.enc.setArgumentTable(Some(self.table));
        self.enc
            .dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        if !self.skip_auto_barriers {
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
    // Callers bind `n as u32` for the kernels' `uint` element counts, so a
    // count past u32::MAX would silently wrap to a partial pass — reject it
    // here at the one seam every 1D op flows through.
    if n > u32::MAX as usize {
        return Err("dispatch_1d exceeds u32 kernel element indexing".into());
    }
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
    let width = pipeline.threadExecutionWidth() as usize;
    let tx = width.min(nx).max(1);
    let groups_x = (nx + tx - 1) / tx;
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
    let width = pipeline.threadExecutionWidth() as usize;
    let tx = width.min(nx).max(1);
    let groups_x = (nx + tx - 1) / tx;
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

    #[test]
    fn dispatch_1d_rejects_u32_overflow_before_encoding() {
        let rt = GpuRuntime::new().expect("runtime");
        let pipe = rt.pipeline("copy_f32").unwrap();
        rt.take_dispatch_count();
        let result = dispatch_1d(&rt, &pipe, u32::MAX as usize + 1, |_| {
            panic!("encode closure must not run for an oversized dispatch");
        });
        assert!(result.is_err(), "oversized dispatch_1d accepted");
        assert_eq!(rt.take_dispatch_count(), 0);
    }
}
