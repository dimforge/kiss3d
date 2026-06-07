//! FXAA (Fast Approximate Anti-Aliasing) post-processing effect.
//!
//! A cheap, purely color-based screen-space anti-aliasing pass that smooths
//! luminance edges. Unlike MSAA it needs no extra samples or geometry passes, so
//! it works on any render path (including the path tracer and offscreen
//! rendering) — at the cost of some softening of fine detail.

use crate::context::Context;
use crate::post_processing::post_processing_effect::{PostProcessingContext, PostProcessingEffect};
use crate::resource::RenderTarget;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QuadVertex {
    position: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FxaaUniforms {
    inv_resolution: [f32; 2],
    edge_threshold: f32,
    edge_threshold_min: f32,
}

/// FXAA post-processing anti-aliasing.
///
/// Set it as the window's post-processing effect to smooth aliased edges:
/// ```no_run
/// # use kiss3d::prelude::*;
/// # use kiss3d::post_processing::Fxaa;
/// # async fn main() {
/// # let mut window = Window::new("Example").await;
/// # let mut scene = SceneNode3d::empty();
/// # let mut camera = OrbitCamera3d::default();
/// let mut fxaa = Fxaa::new();
/// window
///     .render(Some(&mut scene), None, Some(&mut camera), None, None, Some(&mut fxaa))
///     .await;
/// # }
/// ```
pub struct Fxaa {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniforms: FxaaUniforms,
}

impl Default for Fxaa {
    fn default() -> Self {
        Self::new()
    }
}

impl Fxaa {
    /// Creates a new FXAA effect with default edge thresholds.
    pub fn new() -> Fxaa {
        let ctxt = Context::get();

        let bind_group_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fxaa_bind_group_layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fxaa_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let shader =
            ctxt.create_shader_module(Some("fxaa_shader"), include_str!("../builtin/fxaa.wgsl"));

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
            label: Some("fxaa_pipeline"),
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
            Some("fxaa_vertex_buffer"),
            bytemuck::cast_slice(&vertices),
            wgpu::BufferUsages::VERTEX,
        );

        let uniforms = FxaaUniforms {
            inv_resolution: [1.0 / 800.0, 1.0 / 600.0],
            edge_threshold: 0.125,
            edge_threshold_min: 0.0312,
        };
        let uniform_buffer = ctxt.create_buffer_simple(
            Some("fxaa_uniform_buffer"),
            std::mem::size_of::<FxaaUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        Fxaa {
            pipeline,
            bind_group_layout,
            vertex_buffer,
            uniform_buffer,
            uniforms,
        }
    }

    /// Sets the relative and absolute luma-contrast thresholds.
    ///
    /// `edge_threshold` (default `0.125`) is the fraction of the local maximum
    /// luma above which an edge is processed; `edge_threshold_min` (default
    /// `0.0312`) ignores contrast below this absolute amount (dark noise).
    pub fn set_thresholds(&mut self, edge_threshold: f32, edge_threshold_min: f32) {
        self.uniforms.edge_threshold = edge_threshold;
        self.uniforms.edge_threshold_min = edge_threshold_min;
    }
}

impl PostProcessingEffect for Fxaa {
    fn update(&mut self, _dt: f32, w: f32, h: f32, _znear: f32, _zfar: f32) {
        self.uniforms.inv_resolution = [1.0 / w.max(1.0), 1.0 / h.max(1.0)];
    }

    fn draw(&mut self, target: &RenderTarget, context: &mut PostProcessingContext) {
        let ctxt = Context::get();

        let (color_view, sampler) = match target {
            RenderTarget::Offscreen(o) => (&o.color_view, &o.sampler),
            RenderTarget::Screen => return,
        };

        ctxt.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));

        let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fxaa_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut render_pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fxaa_render_pass"),
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
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..4, 0..1);
    }
}
