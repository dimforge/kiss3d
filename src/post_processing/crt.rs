//! A CRT stylization post-process: screen curvature, chromatic aberration,
//! scanlines and a vignette — a classic retro look for 2D scenes.

use crate::context::Context;
use crate::post_processing::post_processing_effect::{PostProcessingContext, PostProcessingEffect};
use crate::resource::RenderTarget;
use bytemuck::{Pod, Zeroable};

/// Vertex data for the full-screen quad.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QuadVertex {
    position: [f32; 2],
}

/// Uniforms mirroring `CrtUniforms` in `crt.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CrtUniforms {
    curvature: f32,
    aberration: f32,
    scanline_intensity: f32,
    scanline_count: f32,
    vignette: f32,
    _pad: [f32; 3],
}

/// A CRT-television stylization effect: barrel-distorted screen curvature,
/// chromatic aberration, scanlines and a vignette.
///
/// Each term is independently adjustable and disabled by setting its strength to 0.
/// Apply it to a 2D scene with
/// [`Window::render_2d_with`](crate::window::Window::render_2d_with).
///
/// ```no_run
/// # use kiss3d::post_processing::Crt;
/// let mut crt = Crt::new();
/// crt.set_curvature(0.15);
/// crt.set_scanlines(0.3, 480.0);
/// ```
pub struct Crt {
    pipeline: wgpu::RenderPipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    uniforms: CrtUniforms,
}

impl Default for Crt {
    fn default() -> Self {
        Self::new()
    }
}

impl Crt {
    /// Creates a CRT effect with moderate, all-on default settings.
    pub fn new() -> Crt {
        let ctxt = Context::get();

        let texture_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("crt_texture_bind_group_layout"),
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

        let uniform_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("crt_uniform_bind_group_layout"),
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
            label: Some("crt_pipeline_layout"),
            bind_group_layouts: &[
                Some(&texture_bind_group_layout),
                Some(&uniform_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let shader = ctxt.create_shader_module(
            Some("crt_shader"),
            &crate::builtin::compile_shader_with_common(
                "package::crt",
                include_str!("../builtin/crt.wgsl"),
            ),
        );

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
            label: Some("crt_pipeline"),
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
            multiview_mask: None,
            cache: None,
        });

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
            Some("crt_vertex_buffer"),
            bytemuck::cast_slice(&vertices),
            wgpu::BufferUsages::VERTEX,
        );

        let uniform_buffer = ctxt.create_buffer_simple(
            Some("crt_uniform_buffer"),
            std::mem::size_of::<CrtUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let uniform_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("crt_uniform_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Crt {
            pipeline,
            texture_bind_group_layout,
            uniform_buffer,
            uniform_bind_group,
            vertex_buffer,
            uniforms: CrtUniforms {
                curvature: 0.12,
                aberration: 0.004,
                scanline_intensity: 0.25,
                scanline_count: 480.0,
                vignette: 0.35,
                _pad: [0.0; 3],
            },
        }
    }

    /// Sets the screen-curvature (barrel-distortion) strength; 0 disables it.
    pub fn set_curvature(&mut self, curvature: f32) {
        self.uniforms.curvature = curvature;
    }

    /// Sets the chromatic-aberration strength (UV offset at the screen edge); 0 disables it.
    pub fn set_aberration(&mut self, aberration: f32) {
        self.uniforms.aberration = aberration;
    }

    /// Sets the scanline darkening `intensity` (`[0, 1]`) and the number of scanlines
    /// down the screen. An intensity of 0 disables scanlines.
    pub fn set_scanlines(&mut self, intensity: f32, count: f32) {
        self.uniforms.scanline_intensity = intensity;
        self.uniforms.scanline_count = count;
    }

    /// Sets the vignette strength (`[0, 1]`); 0 disables it.
    pub fn set_vignette(&mut self, vignette: f32) {
        self.uniforms.vignette = vignette;
    }
}

impl PostProcessingEffect for Crt {
    fn update(&mut self, _dt: f32, _w: f32, _h: f32, _znear: f32, _zfar: f32) {}

    fn draw(&mut self, target: &RenderTarget, context: &mut PostProcessingContext) {
        let ctxt = Context::get();

        let (color_view, sampler) = match target {
            RenderTarget::Offscreen(o) => (&o.color_view, &o.sampler),
            RenderTarget::Screen => return,
        };

        ctxt.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));

        let texture_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("crt_texture_bind_group"),
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

        let mut render_pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("crt_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: context.output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &texture_bind_group, &[]);
        render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..4, 0..1);
    }
}
