//! An useless post-process effect: waves deformations.

// This a simple post-process. I do this only to learn how works post-processing; so it might be
// useless for anybody else.
// This is inspired _a lot_ by: http://en.wikibooks.org/wiki/Opengl::Programming/Post-Processing

use std::f32;

use crate::context::Context;
use crate::post_processing::post_processing_effect::{PostProcessingContext, PostProcessingEffect};
use crate::resource::RenderTarget;
use bytemuck::{Pod, Zeroable};

/// Vertex data for full-screen quad.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QuadVertex {
    position: [f32; 2],
}

/// Uniforms for waves effect.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WavesUniforms {
    offset: f32,
    _padding: [f32; 3],
}

/// An useless post-processing effect mainly to test that everything works correctly.
///
/// It deforms the displayed scene with a wave effect.
pub struct Waves {
    pipeline: wgpu::RenderPipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    time: f32,
}

impl Default for Waves {
    fn default() -> Self {
        Self::new()
    }
}

impl Waves {
    /// Creates a new Waves post processing effect.
    pub fn new() -> Waves {
        let ctxt = Context::get();

        // Create bind group layout for texture + sampler
        let texture_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("waves_texture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create bind group layout for uniforms
        let uniform_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("waves_uniform_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("waves_pipeline_layout"),
            bind_group_layouts: &[&texture_bind_group_layout, &uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Load shader
        let shader =
            ctxt.create_shader_module(Some("waves_shader"), include_str!("../builtin/waves.wgsl"));

        // Vertex buffer layout
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };

        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("waves_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ctxt.surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create full-screen quad vertices
        let vertices = [
            QuadVertex {
                position: [-1.0, -1.0],
            },
            QuadVertex {
                position: [1.0, -1.0],
            },
            QuadVertex {
                position: [-1.0, 1.0],
            },
            QuadVertex {
                position: [1.0, 1.0],
            },
        ];

        let vertex_buffer = ctxt.create_buffer_init(
            Some("waves_vertex_buffer"),
            bytemuck::cast_slice(&vertices),
            wgpu::BufferUsages::VERTEX,
        );

        // Create uniform buffer
        let uniform_buffer = ctxt.create_buffer_simple(
            Some("waves_uniform_buffer"),
            std::mem::size_of::<WavesUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        // Create uniform bind group
        let uniform_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("waves_uniform_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Waves {
            pipeline,
            texture_bind_group_layout,
            uniform_bind_group_layout,
            uniform_buffer,
            uniform_bind_group,
            vertex_buffer,
            time: 0.0,
        }
    }
}

impl PostProcessingEffect for Waves {
    fn update(&mut self, dt: f32, _: f32, _: f32, _: f32, _: f32) {
        self.time += dt;
    }

    fn draw(&mut self, target: &RenderTarget, context: &mut PostProcessingContext) {
        let ctxt = Context::get();

        // Get the source texture and sampler from the render target
        let (color_view, sampler) = match target {
            RenderTarget::Offscreen(o) => (&o.color_view, &o.sampler),
            RenderTarget::Screen => return, // Can't post-process the screen directly
        };

        // Update uniforms
        let move_amount = self.time * 2.0 * f32::consts::PI * 0.75; // 3/4 of a wave cycle per second
        let uniforms = WavesUniforms {
            offset: move_amount,
            _padding: [0.0; 3],
        };
        ctxt.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Create texture bind group for this frame
        let texture_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("waves_texture_bind_group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        // Create render pass to the output view
        {
            let mut render_pass = context
                .encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("waves_render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: context.output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &texture_bind_group, &[]);
            render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..4, 0..1);
        }
    }
}
