//! Edge-aware à-trous wavelet denoiser for the path tracer.
//!
//! Runs after accumulation and before the tonemap pass. It performs several
//! iterations of an SVGF-style à-trous filter (a 5x5 B-spline kernel applied at
//! exponentially growing tap spacing), using edge-stopping weights on the guide
//! normal and on luminance so that geometric and lighting edges are preserved
//! while Monte-Carlo noise is smoothed away.
//!
//! Albedo demodulation is used: the first iteration divides the radiance by the
//! first-hit albedo so only the incident lighting is filtered, and the last
//! iteration re-multiplies by it — preserving crisp texture/albedo detail.
//!
//! The pass ping-pongs between two scratch storage buffers of the same shape as
//! the accumulation buffer; the final iteration writes back into a buffer that
//! the tonemap pass reads. Everything operates at the traced (guide) resolution.
//!
//! ## Cached bind groups
//!
//! The per-iteration uniforms (`width`, `height`, `step`, the demodulate/
//! remodulate flags, the edge-stopping sigmas) and the bind groups are fully
//! determined by the resolution and the iteration count, so they are built once
//! and reused across frames — rebuilt only when the resolution or the iteration
//! count changes. On a progressive renderer the denoiser runs every frame, so
//! re-creating bind groups and re-uploading identical uniforms each frame was
//! pure per-frame CPU/driver overhead.

use bytemuck::{Pod, Zeroable};

use crate::context::Context;

use super::accumulation::Accumulation;

/// Per-iteration uniforms, mirroring the WGSL `DenoiseUniforms`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct DenoiseUniforms {
    width: u32,
    height: u32,
    /// Tap spacing for this iteration (1, 2, 4, ...).
    step: i32,
    /// 1 on the first iteration: read the raw accumulation buffer and divide by
    /// the albedo (demodulate).
    demodulate: u32,
    /// 1 on the last iteration: re-multiply the filtered lighting by the albedo.
    remodulate: u32,
    /// Normal edge-stopping exponent.
    sigma_normal: f32,
    /// Luminance edge-stopping scale.
    sigma_luminance: f32,
    _pad0: f32,
}

const PIXEL_SIZE: u64 = 16; // vec4<f32>
const SIGMA_NORMAL: f32 = 64.0;
const SIGMA_LUMINANCE: f32 = 4.0;

/// Cached per-iteration GPU state for a specific (resolution, iteration count).
/// Rebuilt only when that key changes, then reused every frame.
struct Cache {
    width: u32,
    height: u32,
    iterations: usize,
    /// Two scratch buffers (`array<vec4<f32>>`) the filter ping-pongs between.
    scratch: [wgpu::Buffer; 2],
    /// Per-iteration uniform buffers (held to keep them alive for the bind
    /// groups that reference them; the shader reads them, this code does not).
    _uniforms: Vec<wgpu::Buffer>,
    /// Per-iteration bind groups.
    bind_groups: Vec<wgpu::BindGroup>,
}

/// Owns the à-trous compute pipeline and the cached per-resolution state.
pub struct Denoise {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    cache: Option<Cache>,
}

impl Denoise {
    /// Builds the denoiser pipeline. Scratch buffers, uniforms, and bind groups
    /// are created lazily to match the accumulation resolution on the first
    /// `run`.
    pub fn new() -> Denoise {
        let ctxt = Context::get();

        let bind_group_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt_denoise_bind_group_layout"),
            entries: &[
                storage_entry(0, true),  // src (read)
                storage_entry(1, false), // dst (read_write)
                storage_entry(2, true),  // shared accumulation buffer (guide regions)
                uniform_entry(3),
            ],
        });

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rt_denoise_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let shader = ctxt.create_shader_module(
            Some("rt_denoise_shader"),
            include_str!("../../builtin/raytrace/denoise.wgsl"),
        );

        let pipeline = ctxt
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rt_denoise_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Denoise {
            pipeline,
            bind_group_layout,
            cache: None,
        }
    }

    fn make_scratch(width: u32, height: u32, label: &str) -> wgpu::Buffer {
        let count = (width.max(1) as u64) * (height.max(1) as u64);
        Context::get().create_buffer_simple(
            Some(label),
            count * PIXEL_SIZE,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
    }

    /// (Re)builds the cached scratch buffers, uniforms, and bind groups if the
    /// resolution or the iteration count changed.
    ///
    /// All cached state references either `accum.buffer` (stable while the
    /// resolution is unchanged — `Accumulation::ensure` only recreates it on a
    /// resize) or the scratch buffers, so the bind groups stay valid across
    /// frames until the key changes.
    fn ensure_cache(&mut self, accum: &Accumulation, iterations: usize) {
        let width = accum.width;
        let height = accum.height;

        if let Some(c) = &self.cache {
            if c.width == width && c.height == height && c.iterations == iterations {
                return;
            }
        }

        let ctxt = Context::get();
        let scratch = [
            Self::make_scratch(width, height, "rt_denoise_scratch0"),
            Self::make_scratch(width, height, "rt_denoise_scratch1"),
        ];

        // One uniform buffer per iteration: each pass needs a distinct
        // step/demodulate/remodulate. Written once here; constant across frames.
        let mut uniforms = Vec::with_capacity(iterations);
        let mut bind_groups = Vec::with_capacity(iterations);
        for i in 0..iterations {
            let uniform = ctxt.create_buffer_simple(
                Some("rt_denoise_uniform"),
                std::mem::size_of::<DenoiseUniforms>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            ctxt.write_buffer(
                &uniform,
                0,
                bytemuck::bytes_of(&DenoiseUniforms {
                    width,
                    height,
                    step: 1i32 << i as u32,
                    demodulate: (i == 0) as u32,
                    remodulate: (i == iterations - 1) as u32,
                    sigma_normal: SIGMA_NORMAL,
                    sigma_luminance: SIGMA_LUMINANCE,
                    _pad0: 0.0,
                }),
            );

            // Iteration 0 reads the raw accumulation buffer (demodulating on the
            // fly); later iterations alternate between the two scratch buffers.
            let src = if i == 0 {
                &accum.buffer
            } else {
                &scratch[(i - 1) % 2]
            };
            let dst = &scratch[i % 2];

            bind_groups.push(ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rt_denoise_bind_group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        // The shared accumulation buffer holds the albedo/normal
                        // guide regions the shader reads.
                        resource: accum.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            }));
            uniforms.push(uniform);
        }

        self.cache = Some(Cache {
            width,
            height,
            iterations,
            scratch,
            _uniforms: uniforms,
            bind_groups,
        });
    }

    /// Runs `iterations` à-trous passes over the radiance in `accum`, returning a
    /// reference to the storage buffer holding the denoised HDR radiance (laid
    /// out exactly like `accum.buffer`, so the tonemap pass can read it directly).
    ///
    /// `iterations` must be at least 1; the caller is responsible for skipping
    /// the denoiser entirely when it is disabled or the image has converged.
    pub fn run<'a>(
        &'a mut self,
        encoder: &mut wgpu::CommandEncoder,
        accum: &Accumulation,
        iterations: u32,
        gpu: &mut crate::renderer::timings::GpuTimer,
    ) -> &'a wgpu::Buffer {
        let iterations = iterations.max(1) as usize;
        self.ensure_cache(accum, iterations);
        let cache = self.cache.as_ref().expect("cache just ensured");

        let groups_x = accum.width.div_ceil(8);
        let groups_y = accum.height.div_ceil(8);
        for bind_group in &cache.bind_groups {
            let denoise_ts = gpu.compute_scope("denoise");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rt_denoise_pass"),
                timestamp_writes: denoise_ts,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }

        // The final iteration's destination is `scratch[(iterations - 1) % 2]`.
        &self.cache.as_ref().expect("cache ensured").scratch[(iterations - 1) % 2]
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
