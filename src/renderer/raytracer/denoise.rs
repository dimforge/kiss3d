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

/// Owns the à-trous compute pipeline and its two ping-pong scratch buffers.
pub struct Denoise {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Per-iteration uniform buffers (one per possible iteration; sized lazily).
    uniforms: Vec<wgpu::Buffer>,
    /// Two scratch buffers (`array<vec4<f32>>`) the filter ping-pongs between.
    scratch: [wgpu::Buffer; 2],
    width: u32,
    height: u32,
}

impl Denoise {
    /// Builds the denoiser pipeline. Scratch buffers are (re)allocated lazily to
    /// match the accumulation resolution on the first `run`.
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
            uniforms: Vec::new(),
            scratch: [
                Self::make_scratch(1, 1, "rt_denoise_scratch0"),
                Self::make_scratch(1, 1, "rt_denoise_scratch1"),
            ],
            width: 1,
            height: 1,
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

    /// Resizes the scratch buffers if the accumulation resolution changed.
    fn ensure(&mut self, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        self.scratch = [
            Self::make_scratch(width, height, "rt_denoise_scratch0"),
            Self::make_scratch(width, height, "rt_denoise_scratch1"),
        ];
        self.width = width;
        self.height = height;
    }

    /// Lazily grows the pool of per-iteration uniform buffers to at least `n`.
    fn ensure_uniforms(&mut self, n: usize) {
        let ctxt = Context::get();
        while self.uniforms.len() < n {
            self.uniforms.push(ctxt.create_buffer_simple(
                Some("rt_denoise_uniform"),
                std::mem::size_of::<DenoiseUniforms>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ));
        }
    }

    /// Runs `iterations` à-trous passes over the radiance in `accum`, returning a
    /// reference to the storage buffer holding the denoised HDR radiance (laid
    /// out exactly like `accum.buffer`, so the tonemap pass can read it directly).
    ///
    /// `iterations` must be at least 1; the caller is responsible for skipping
    /// the denoiser entirely when it is disabled.
    pub fn run<'a>(
        &'a mut self,
        encoder: &mut wgpu::CommandEncoder,
        accum: &'a Accumulation,
        iterations: u32,
    ) -> &'a wgpu::Buffer {
        let ctxt = Context::get();
        let width = accum.width;
        let height = accum.height;
        self.ensure(width, height);

        let iterations = iterations.max(1) as usize;
        self.ensure_uniforms(iterations);

        // Ping-pong: iteration 0 reads the raw accumulation buffer (demodulating
        // on the fly) and writes scratch[0]; subsequent iterations alternate
        // between the two scratch buffers. The final write lands in
        // `scratch[(iterations - 1) % 2]`, which is what we return.
        for i in 0..iterations {
            let step = 1i32 << i as u32;
            let demodulate = (i == 0) as u32;
            let remodulate = (i == iterations - 1) as u32;

            ctxt.write_buffer(
                &self.uniforms[i],
                0,
                bytemuck::bytes_of(&DenoiseUniforms {
                    width,
                    height,
                    step,
                    demodulate,
                    remodulate,
                    sigma_normal: 64.0,
                    sigma_luminance: 4.0,
                    _pad0: 0.0,
                }),
            );

            let src = if i == 0 {
                &accum.buffer
            } else {
                &self.scratch[(i - 1) % 2]
            };
            let dst = &self.scratch[i % 2];

            let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
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
                        resource: self.uniforms[i].as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rt_denoise_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }

        &self.scratch[(iterations - 1) % 2]
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
