//! Compute-pipeline assembly for the path tracer.
//!
//! The WGSL module is built at runtime by concatenating the shared preamble, a
//! backend-specific intersection snippet, and the shared kernel (wgpu has no
//! `#include`). Bind group 0 holds the frame uniforms and accumulation buffer;
//! bind group 1 holds the scene buffers (and, for the compute backend, the BVH).

use bytemuck::{Pod, Zeroable};

use crate::context::Context;

use super::accumulation::Accumulation;
use super::gpu_scene::GpuScene;
use super::RayBackend;

/// Per-frame uniforms, mirroring the WGSL `FrameUniforms` (std140, 112 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FrameUniforms {
    /// Inverse view-projection matrix (column-major).
    pub inv_view_proj: [[f32; 4]; 4],
    /// Camera position in world space.
    pub cam_eye: [f32; 3],
    /// Render target width in pixels.
    pub width: u32,
    /// Render target height in pixels.
    pub height: u32,
    /// Index of the sample being accumulated (0 = first / reset).
    pub sample_index: u32,
    /// Number of triangles in the scene.
    pub num_triangles: u32,
    /// Number of lights in the scene.
    pub num_lights: u32,
    /// Ambient intensity (sky term).
    pub ambient: f32,
    /// Maximum path length.
    pub max_bounces: u32,
    /// RNG seed (varies per frame).
    pub seed: u32,
    /// Number of samples to trace this dispatch.
    pub samples_per_frame: u32,
}

const PREAMBLE: &str = include_str!("../../builtin/raytrace/rt_preamble.wgsl");
const INTERSECT_BVH: &str = include_str!("../../builtin/raytrace/rt_intersect_bvh.wgsl");
#[cfg(feature = "raytracing")]
const INTERSECT_RAYQUERY: &str = include_str!("../../builtin/raytrace/rt_intersect_rayquery.wgsl");
const KERNEL: &str = include_str!("../../builtin/raytrace/rt_kernel.wgsl");

/// The compute pipeline plus its persistent frame-uniform buffer and layouts.
pub struct PathTracePipeline {
    pipeline: wgpu::ComputePipeline,
    group0_layout: wgpu::BindGroupLayout,
    group1_layout: wgpu::BindGroupLayout,
    frame_uniform: wgpu::Buffer,
}

impl PathTracePipeline {
    /// Builds the path-tracing compute pipeline for the given backend.
    pub fn new(backend: RayBackend) -> PathTracePipeline {
        let ctxt = Context::get();

        let intersect = match backend {
            RayBackend::Compute => INTERSECT_BVH,
            #[cfg(feature = "raytracing")]
            RayBackend::HardwareRayQuery => INTERSECT_RAYQUERY,
        };
        // The `enable` directive must precede all declarations, so it is prepended
        // to the whole module for the ray-query backend.
        let prologue = match backend {
            RayBackend::Compute => "",
            #[cfg(feature = "raytracing")]
            RayBackend::HardwareRayQuery => "enable wgpu_ray_query;\n",
        };
        let source = format!("{prologue}{PREAMBLE}\n{intersect}\n{KERNEL}");
        let shader = ctxt.create_shader_module(Some("rt_path_tracer"), &source);

        let group0_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt_group0_layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, false), // accumulation (read_write)
            ],
        });

        let group1_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt_group1_layout"),
            entries: &Self::group1_entries(backend),
        });

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rt_pipeline_layout"),
            bind_group_layouts: &[Some(&group0_layout), Some(&group1_layout)],
            immediate_size: 0,
        });

        let pipeline = ctxt
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rt_path_tracer_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let frame_uniform = ctxt.create_buffer_simple(
            Some("rt_frame_uniform"),
            std::mem::size_of::<FrameUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        PathTracePipeline {
            pipeline,
            group0_layout,
            group1_layout,
            frame_uniform,
        }
    }

    #[cfg(not(feature = "raytracing"))]
    fn group1_entries(_backend: RayBackend) -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![
            storage_entry(0, true), // vertices
            storage_entry(1, true), // triangles
            storage_entry(2, true), // materials
            storage_entry(3, true), // lights
            storage_entry(4, true), // bvh
        ]
    }

    #[cfg(feature = "raytracing")]
    fn group1_entries(backend: RayBackend) -> Vec<wgpu::BindGroupLayoutEntry> {
        let mut entries = vec![
            storage_entry(0, true), // vertices
            storage_entry(1, true), // triangles
            storage_entry(2, true), // materials
            storage_entry(3, true), // lights
        ];
        match backend {
            RayBackend::Compute => entries.push(storage_entry(4, true)), // bvh
            RayBackend::HardwareRayQuery => entries.push(wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            }),
        }
        entries
    }

    /// Updates the frame uniforms for this frame.
    pub fn write_uniforms(&self, uniforms: &FrameUniforms) {
        Context::get().write_buffer(&self.frame_uniform, 0, bytemuck::bytes_of(uniforms));
    }

    /// Records the path-tracing dispatch into `encoder` for the compute backend.
    pub fn dispatch_compute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &GpuScene,
        accum: &Accumulation,
        width: u32,
        height: u32,
    ) {
        let ctxt = Context::get();

        let group0 = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_group0"),
            layout: &self.group0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.frame_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: accum.buffer.as_entire_binding(),
                },
            ],
        });

        let group1 = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_group1"),
            layout: &self.group1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene.vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scene.triangles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scene.materials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scene.lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scene.bvh.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rt_path_trace_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group0, &[]);
        pass.set_bind_group(1, &group1, &[]);
        let gx = width.div_ceil(8);
        let gy = height.div_ceil(8);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    /// Records the path-tracing dispatch for the hardware ray-query backend.
    ///
    /// Phase 2: builds bind group 1 with the TLAS bound at binding 4 instead of
    /// the BVH buffer. The shared kernel and group 0 are identical to the compute
    /// path.
    #[cfg(feature = "raytracing")]
    pub fn dispatch_hardware(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &GpuScene,
        accum: &Accumulation,
        width: u32,
        height: u32,
    ) {
        let ctxt = Context::get();

        let tlas = scene
            .tlas
            .as_ref()
            .expect("hardware backend requires a built TLAS");

        let group0 = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_group0"),
            layout: &self.group0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.frame_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: accum.buffer.as_entire_binding(),
                },
            ],
        });

        let group1 = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_group1"),
            layout: &self.group1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene.vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scene.triangles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scene.materials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scene.lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::AccelerationStructure(tlas),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rt_path_trace_pass_hw"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group0, &[]);
        pass.set_bind_group(1, &group1, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
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
