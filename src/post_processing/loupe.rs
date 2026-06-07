//! Magnifier loupe post-processing effect.
//!
//! Blits the rendered scene to the screen and overlays a magnified, nearest-
//! neighbour crop of a focus region in a corner, so individual pixels read as crisp
//! blocks. It's a debugging/inspection aid — handy for eyeballing anti-aliasing
//! quality, sub-pixel detail, shimmering, or any per-pixel artifact that's invisible
//! at 1:1.
//!
//! The loupe can also *wrap* another post-processing effect (via
//! [`Loupe::wrapping`] / [`Loupe::set_inner`]): the inner effect runs first into an
//! intermediate target and the loupe magnifies *its* output, letting you inspect the
//! result of, e.g., [`Fxaa`](crate::post_processing::Fxaa) under magnification even
//! though only one post-processing effect can be active at a time.
//!
//! ```no_run
//! # use kiss3d::prelude::*;
//! # use kiss3d::post_processing::Loupe;
//! # #[kiss3d::main]
//! # async fn main() {
//! # let mut window = Window::new("Example").await;
//! # let mut scene = SceneNode3d::empty();
//! # let mut camera = OrbitCamera3d::default();
//! let mut loupe = Loupe::new();
//! loupe.set_zoom(12.0);
//! window
//!     .render(Some(&mut scene), None, Some(&mut camera), None, None, Some(&mut loupe))
//!     .await;
//! # }
//! ```

use crate::context::Context;
use crate::post_processing::post_processing_effect::{PostProcessingContext, PostProcessingEffect};
use crate::resource::RenderTarget;
use bytemuck::{Pod, Zeroable};
use glamx::Vec2;

/// Which corner of the viewport the magnified inset is drawn in.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoupeCorner {
    /// Top-left corner.
    TopLeft,
    /// Top-right corner.
    TopRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-right corner.
    BottomRight,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct LoupeUniforms {
    resolution: [f32; 2],
    focus_px: [f32; 2],
    inset_min: [f32; 2],
    inset_max: [f32; 2],
    region_half_px: f32,
    _pad: [f32; 3],
    border_color: [f32; 4],
}

/// An intermediate surface-format target the inner effect renders into, so the
/// loupe can magnify the post-processed result. Allocated lazily, only when an
/// inner effect is set.
struct MidTarget {
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    width: u32,
    height: u32,
}

impl MidTarget {
    fn new(width: u32, height: u32) -> Self {
        let ctxt = Context::get();
        let texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("loupe_mid_texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ctxt.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("loupe_mid_sampler"),
            ..Default::default()
        });
        MidTarget {
            view,
            sampler,
            width,
            height,
        }
    }
}

/// Magnifier loupe post-processing effect.
pub struct Loupe {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniforms: LoupeUniforms,
    zoom: f32,
    focus: Vec2,
    corner: LoupeCorner,
    size: f32,
    border_color: [f32; 3],
    inner: Option<Box<dyn PostProcessingEffect>>,
    mid: Option<MidTarget>,
}

impl Default for Loupe {
    fn default() -> Self {
        Self::new()
    }
}

impl Loupe {
    /// Creates a magnifier with sensible defaults: 8× zoom on the screen center,
    /// drawn in the bottom-right corner.
    pub fn new() -> Loupe {
        let ctxt = Context::get();

        let bind_group_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("loupe_bind_group_layout"),
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
            label: Some("loupe_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let shader = ctxt.create_shader_module(
            Some("loupe_shader"),
            &crate::builtin::compile_shader_with_common(
                "package::loupe",
                include_str!("../builtin/loupe.wgsl"),
            ),
        );

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };

        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("loupe_pipeline"),
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

        let vertices: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
        let vertex_buffer = ctxt.create_buffer_init(
            Some("loupe_vertex_buffer"),
            bytemuck::cast_slice(&vertices),
            wgpu::BufferUsages::VERTEX,
        );

        let uniform_buffer = ctxt.create_buffer_simple(
            Some("loupe_uniform_buffer"),
            std::mem::size_of::<LoupeUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        Loupe {
            pipeline,
            bind_group_layout,
            vertex_buffer,
            uniform_buffer,
            uniforms: LoupeUniforms {
                resolution: [1.0, 1.0],
                focus_px: [0.0, 0.0],
                inset_min: [0.0, 0.0],
                inset_max: [0.0, 0.0],
                region_half_px: 1.0,
                _pad: [0.0; 3],
                border_color: [1.0, 0.9, 0.2, 1.0],
            },
            zoom: 8.0,
            focus: Vec2::new(0.5, 0.5),
            corner: LoupeCorner::BottomRight,
            size: 0.4,
            border_color: [1.0, 0.9, 0.2],
            inner: None,
            mid: None,
        }
    }

    /// Creates a loupe that magnifies the output of `inner` rather than the raw
    /// scene. See [`set_inner`](Self::set_inner).
    pub fn wrapping(inner: Box<dyn PostProcessingEffect>) -> Loupe {
        let mut loupe = Loupe::new();
        loupe.set_inner(Some(inner));
        loupe
    }

    /// Sets the magnification factor (`>= 1`). Default `8`.
    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom.max(1.0);
    }

    /// The current magnification factor.
    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    /// Sets the magnified point as normalized coordinates in `[0, 1]`, with `(0, 0)`
    /// at the top-left of the viewport. Default `(0.5, 0.5)` (screen center).
    pub fn set_focus(&mut self, focus: Vec2) {
        self.focus = focus.clamp(Vec2::ZERO, Vec2::ONE);
    }

    /// Sets which corner the inset is drawn in. Default [`LoupeCorner::BottomRight`].
    pub fn set_corner(&mut self, corner: LoupeCorner) {
        self.corner = corner;
    }

    /// Sets the inset side length as a fraction of the smaller viewport dimension,
    /// clamped to `(0, 1]`. Default `0.4`.
    pub fn set_size(&mut self, fraction: f32) {
        self.size = fraction.clamp(0.01, 1.0);
    }

    /// Sets the RGB color of the region outline and inset frame.
    pub fn set_border_color(&mut self, rgb: [f32; 3]) {
        self.border_color = rgb;
    }

    /// Sets (or clears) the wrapped inner effect. When set, the inner effect renders
    /// first into an intermediate target and the loupe magnifies *its* output;
    /// otherwise the loupe magnifies the raw rendered scene.
    pub fn set_inner(&mut self, inner: Option<Box<dyn PostProcessingEffect>>) {
        self.inner = inner;
    }

    /// Mutable access to the wrapped inner effect, if any (e.g. to tweak its
    /// parameters at runtime).
    pub fn inner_mut(&mut self) -> Option<&mut (dyn PostProcessingEffect + 'static)> {
        self.inner.as_deref_mut()
    }
}

impl PostProcessingEffect for Loupe {
    fn update(&mut self, dt: f32, w: f32, h: f32, znear: f32, zfar: f32) {
        if let Some(inner) = &mut self.inner {
            inner.update(dt, w, h, znear, zfar);
            let (iw, ih) = (w.max(1.0) as u32, h.max(1.0) as u32);
            if self.mid.as_ref().map(|m| (m.width, m.height)) != Some((iw, ih)) {
                self.mid = Some(MidTarget::new(iw, ih));
            }
        } else {
            self.mid = None;
        }

        // Square inset, `size` × the smaller viewport dimension, inset by a margin
        // from the chosen corner.
        let margin = 12.0;
        let side = (self.size * w.min(h)).clamp(1.0, (w - 2.0 * margin).max(1.0));
        let (left, top) = match self.corner {
            LoupeCorner::TopLeft => (margin, margin),
            LoupeCorner::TopRight => (w - margin - side, margin),
            LoupeCorner::BottomLeft => (margin, h - margin - side),
            LoupeCorner::BottomRight => (w - margin - side, h - margin - side),
        };

        self.uniforms = LoupeUniforms {
            resolution: [w, h],
            focus_px: [self.focus.x * w, self.focus.y * h],
            inset_min: [left, top],
            inset_max: [left + side, top + side],
            region_half_px: side / (2.0 * self.zoom),
            _pad: [0.0; 3],
            border_color: [
                self.border_color[0],
                self.border_color[1],
                self.border_color[2],
                1.0,
            ],
        };
    }

    fn draw(&mut self, target: &RenderTarget, context: &mut PostProcessingContext) {
        let ctxt = Context::get();
        ctxt.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));

        // Source for the loupe: the inner effect's output when wrapping one (rendered
        // into our intermediate target first), otherwise the raw resolved scene.
        let (src_view, src_sampler) = match (&mut self.inner, &self.mid) {
            (Some(inner), Some(mid)) => {
                let mut inner_ctx = PostProcessingContext {
                    encoder: context.encoder,
                    output_view: &mid.view,
                };
                inner.draw(target, &mut inner_ctx);
                (&mid.view, &mid.sampler)
            }
            _ => match target {
                RenderTarget::Offscreen(o) => (&o.color_view, &o.sampler),
                RenderTarget::Screen => return,
            },
        };

        let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("loupe_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(src_sampler),
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
                label: Some("loupe_render_pass"),
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
