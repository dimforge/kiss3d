//! Planar reflectors (mirrors): make any object's surface show a reflected view of
//! the scene — but integrated into the **default PBR
//! material**, so a reflective surface can also have its full PBR shading (base
//! color, textures, roughness, metallic) with the reflection blended on top.
//!
//! A [`Reflector`] is a lightweight per-object resource (set via
//! [`Object3d::set_reflector`](crate::scene::Object3d::set_reflector) or the
//! [`SceneNode3d::add_reflector`](crate::scene::SceneNode3d::add_reflector)
//! convenience) holding the object's own reflection render target, its object-space
//! plane normal, and an intensity. Each frame the window finds reflector objects,
//! reads each one's world plane from its node transform, renders the scene from a
//! [`MirrorCamera`] into the reflector's target, and stores the reflected
//! view-projection; the default material then samples that target (projected) and
//! composites it as an additive delta over the environment specular.
//!
//! The reflection is folded into the **projection** (`proj · reflect`) so the mirror
//! camera's view stays a valid rigid [`Pose3`]; the reflected projection flips
//! winding (rendered with back-face culling off), and geometry behind the mirror is
//! clipped by a world-space clip plane (the default material's `set_clip_plane`).
//! Each reflector owns its render target, so any number of differently-oriented
//! mirrors can coexist. The virtual-camera seam ([`MirrorCamera`]) is kept separate
//! from how it's derived, so portals (a surface showing a linked camera's view) can
//! reuse the same texture + projected-sampling path later.

use std::cell::Cell;

use glamx::{Mat3, Mat4, Pose3, Vec3, Vec4};

use crate::camera::Camera3d;
use crate::context::Context;
use crate::event::WindowEvent;
use crate::window::Canvas;

/// 4x4 reflection (Householder) matrix for the view-space plane with unit normal
/// `n` and offset `d` (plane: `n·x + d = 0`).
fn householder(n: Vec3, d: f32) -> Mat4 {
    let (a, b, c) = (n.x, n.y, n.z);
    Mat4::from_cols(
        Vec4::new(1.0 - 2.0 * a * a, -2.0 * a * b, -2.0 * a * c, 0.0),
        Vec4::new(-2.0 * a * b, 1.0 - 2.0 * b * b, -2.0 * b * c, 0.0),
        Vec4::new(-2.0 * a * c, -2.0 * b * c, 1.0 - 2.0 * c * c, 0.0),
        Vec4::new(-2.0 * a * d, -2.0 * b * d, -2.0 * c * d, 1.0),
    )
}

/// A camera that renders the scene mirrored across a world plane, for a reflector.
/// Its view is the main camera's (a valid rigid `Pose3`); the reflection is folded
/// into the projection (`proj · reflect`), which flips winding (so the mirror pass
/// must disable back-face culling).
pub struct MirrorCamera {
    view: Pose3,
    proj: Mat4,
    eye: Vec3,
    znear: f32,
    zfar: f32,
}

impl MirrorCamera {
    /// Builds the mirror camera for the world plane `(point, normal)` from the main
    /// camera's pass-0 view + projection, eye and clip planes.
    pub fn new(
        view: Pose3,
        proj: Mat4,
        eye: Vec3,
        znear: f32,
        zfar: f32,
        plane_point: Vec3,
        plane_normal: Vec3,
    ) -> Self {
        // Reflection plane in view space. The view is rigid, so the normal
        // transforms by its 3x3 rotation; the point by the full view matrix.
        let view_mat = view.to_mat4();
        let m3 = Mat3::from_cols(
            view_mat.x_axis.truncate(),
            view_mat.y_axis.truncate(),
            view_mat.z_axis.truncate(),
        );
        let n = (m3 * plane_normal).normalize();
        let p_view = (view_mat * plane_point.extend(1.0)).truncate();
        let d = -n.dot(p_view);
        let reflect = householder(n, d);

        // Reflected eye (across the world plane), so reflective surfaces shade right.
        let nn = plane_normal.normalize();
        let eye_refl = eye - 2.0 * (nn.dot(eye - plane_point)) * nn;

        MirrorCamera {
            view,
            proj: proj * reflect,
            eye: eye_refl,
            znear,
            zfar,
        }
    }

    /// The reflected clip transform (`proj · reflect · view`): projects a world
    /// position into the reflection texture's clip space (the surface samples it).
    pub fn reflector_view_proj(&self) -> Mat4 {
        self.proj * self.view.to_mat4()
    }
}

impl Camera3d for MirrorCamera {
    fn handle_event(&mut self, _: &Canvas, _: &WindowEvent) {}
    fn update(&mut self, _: &Canvas) {}
    fn eye(&self) -> Vec3 {
        self.eye
    }
    fn view_transform(&self) -> Pose3 {
        self.view
    }
    fn transformation(&self) -> Mat4 {
        self.proj * self.view.to_mat4()
    }
    fn inverse_transformation(&self) -> Mat4 {
        self.transformation().inverse()
    }
    fn clip_planes(&self) -> (f32, f32) {
        (self.znear, self.zfar)
    }
    fn view_transform_pair(&self, _pass: usize) -> (Pose3, Mat4) {
        (self.view, self.proj)
    }
}

/// A per-object planar reflector: the reflection render target (the object's own
/// mirror texture) plus its object-space plane normal and intensity.
///
/// Attach one to a (typically flat) object with
/// [`Object3d::set_reflector`](crate::scene::Object3d::set_reflector); the window
/// renders the mirrored scene into [`Self::color_view`] each frame and the default
/// material composites it over the object's PBR shading. The reflection plane is the
/// object's world transform applied to its origin + [`Self::local_normal`].
pub struct Reflector {
    width: u32,
    height: u32,
    _color: wgpu::Texture,
    color_view: wgpu::TextureView,
    _depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// Object-space plane normal (default +Z, the local normal of a `quad`).
    local_normal: Vec3,
    /// Reflection strength in `[0, 1]` (scales the composited reflection).
    intensity: f32,
    /// Normal-alignment falloff exponent. `0` disables it (uniform reflection); when
    /// `> 0`, the reflection fades by `max(dot(N, plane_normal), 0)^normal_falloff`,
    /// so it vanishes where the surface normal `N` diverges from the reflector's
    /// plane normal (e.g. a sphere reflects only on the cap facing the plane normal,
    /// fading toward its silhouette; larger values fade faster).
    normal_falloff: f32,
    /// World -> reflection-texture clip transform, set by the window each frame.
    view_proj: Cell<Mat4>,
    /// Bumped whenever the target is reallocated (resize). The material keys its
    /// cached texture bind group on this: the `color_view` lives in a fixed struct
    /// slot, so its address is unchanged after a resize and can't be used to detect
    /// that the underlying texture was replaced.
    generation: u64,
}

impl Default for Reflector {
    fn default() -> Self {
        Self::new()
    }
}

impl Reflector {
    /// Creates a reflector with a 1×1 placeholder target (the window resizes it to
    /// the viewport), object-space normal +Z, and full intensity.
    pub fn new() -> Reflector {
        let (color, color_view, depth, depth_view) = Self::make_targets(1, 1);
        Reflector {
            width: 1,
            height: 1,
            _color: color,
            color_view,
            _depth: depth,
            depth_view,
            local_normal: Vec3::Z,
            intensity: 1.0,
            normal_falloff: 0.0,
            view_proj: Cell::new(Mat4::IDENTITY),
            generation: 1,
        }
    }

    /// Sets the object-space plane normal (builder form). Defaults to +Z (a `quad`'s
    /// local normal); use e.g. +Y for a horizontal face whose object frame is upright.
    pub fn with_local_normal(mut self, normal: Vec3) -> Reflector {
        self.local_normal = normal.normalize();
        self
    }

    /// Sets the reflection intensity in `[0, 1]` (builder form).
    pub fn with_intensity(mut self, intensity: f32) -> Reflector {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Sets the normal-alignment falloff exponent (builder form). `0` (the default)
    /// keeps the reflection uniform across the surface; larger values make it fade
    /// faster as the surface normal diverges from the reflector's plane normal —
    /// useful on curved reflectors (e.g. a sphere) so the planar reflection only
    /// shows on the cap facing the plane normal. See [`Self::normal_falloff`].
    pub fn with_normal_falloff(mut self, falloff: f32) -> Reflector {
        self.normal_falloff = falloff.max(0.0);
        self
    }

    fn make_targets(
        w: u32,
        h: u32,
    ) -> (
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
    ) {
        let ctxt = Context::get();
        let color = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("reflector_color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::post_processing::HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("reflector_depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Context::depth_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        (color, color_view, depth, depth_view)
    }

    /// Resizes the reflection target if needed (the window calls this each frame to
    /// match the viewport).
    pub fn resize(&mut self, width: u32, height: u32) {
        let (w, h) = (width.max(1), height.max(1));
        if self.width == w && self.height == h {
            return;
        }
        let (color, color_view, depth, depth_view) = Self::make_targets(w, h);
        self._color = color;
        self.color_view = color_view;
        self._depth = depth;
        self.depth_view = depth_view;
        self.width = w;
        self.height = h;
        self.generation += 1;
    }

    /// Counter bumped on each target reallocation (resize); the material keys its
    /// cached reflection bind group on this so it rebinds the new texture.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The reflection color target (the window renders the mirror view here; the
    /// material samples it).
    pub fn color_view(&self) -> &wgpu::TextureView {
        &self.color_view
    }

    /// The reflection depth target.
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// The object-space plane normal.
    pub fn local_normal(&self) -> Vec3 {
        self.local_normal
    }

    /// Sets the object-space plane normal.
    pub fn set_local_normal(&mut self, normal: Vec3) {
        self.local_normal = normal.normalize();
    }

    /// The reflection intensity in `[0, 1]`.
    pub fn intensity(&self) -> f32 {
        self.intensity
    }

    /// Sets the reflection intensity in `[0, 1]`.
    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity.clamp(0.0, 1.0);
    }

    /// The normal-alignment falloff exponent (`0` = disabled). See
    /// [`Self::with_normal_falloff`].
    pub fn normal_falloff(&self) -> f32 {
        self.normal_falloff
    }

    /// Sets the normal-alignment falloff exponent (`0` = disabled). See
    /// [`Self::with_normal_falloff`].
    pub fn set_normal_falloff(&mut self, falloff: f32) {
        self.normal_falloff = falloff.max(0.0);
    }

    /// The world -> reflection-clip transform the material samples with.
    pub fn view_proj(&self) -> Mat4 {
        self.view_proj.get()
    }

    /// Stores the world -> reflection-clip transform (the window sets this each frame).
    pub fn set_view_proj(&self, vp: Mat4) {
        self.view_proj.set(vp);
    }
}

/// Weighted-blended OIT accumulation targets + composite pipeline for the
/// reflector capture pass, so transparent (alpha < 1) surfaces also appear in
/// mirrors. One set is shared by all reflectors of a window (each capture
/// clears the targets).
///
/// The HDR pipeline owns the equivalent targets for the main pass, but those
/// match the film's MSAA sample count while the reflector capture is always
/// single-sampled — hence this dedicated single-sample set (same formats, same
/// `hdr_oit.wgsl` composite).
pub(crate) struct ReflectorOit {
    width: u32,
    height: u32,
    _accum: wgpu::Texture,
    accum_view: wgpu::TextureView,
    _reveal: wgpu::Texture,
    reveal_view: wgpu::TextureView,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
}

impl ReflectorOit {
    pub fn new(width: u32, height: u32) -> ReflectorOit {
        let ctxt = Context::get();
        let layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reflector_oit_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let shader = ctxt.create_shader_module(
            Some("reflector_oit_shader"),
            include_str!("../builtin/hdr_oit.wgsl"),
        );
        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reflector_oit_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: (std::mem::size_of::<f32>() * 2) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };
        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("reflector_oit_composite_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: std::slice::from_ref(&vertex_layout),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_composite"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::post_processing::HDR_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
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
            Some("reflector_oit_quad"),
            bytemuck::cast_slice(&vertices),
            wgpu::BufferUsages::VERTEX,
        );

        let (accum, accum_view, reveal, reveal_view) = Self::make_targets(width, height);
        ReflectorOit {
            width: width.max(1),
            height: height.max(1),
            _accum: accum,
            accum_view,
            _reveal: reveal,
            reveal_view,
            layout,
            pipeline,
            vertex_buffer,
        }
    }

    fn make_targets(
        w: u32,
        h: u32,
    ) -> (
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
    ) {
        let ctxt = Context::get();
        let make = |label: &str, format: wgpu::TextureFormat| {
            let tex = ctxt.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        let (accum, accum_view) = make(
            "reflector_oit_accum",
            crate::post_processing::OIT_ACCUM_FORMAT,
        );
        let (reveal, reveal_view) = make(
            "reflector_oit_reveal",
            crate::post_processing::OIT_REVEAL_FORMAT,
        );
        (accum, accum_view, reveal, reveal_view)
    }

    /// Resizes the OIT targets if needed.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (w, h) = (width.max(1), height.max(1));
        if self.width == w && self.height == h {
            return;
        }
        let (accum, accum_view, reveal, reveal_view) = Self::make_targets(w, h);
        self._accum = accum;
        self.accum_view = accum_view;
        self._reveal = reveal;
        self.reveal_view = reveal_view;
        self.width = w;
        self.height = h;
    }

    /// The OIT accumulation target (additive premultiplied-weighted color).
    pub fn accum_view(&self) -> &wgpu::TextureView {
        &self.accum_view
    }

    /// The OIT revealage target (multiplicative coverage).
    pub fn reveal_view(&self) -> &wgpu::TextureView {
        &self.reveal_view
    }

    /// Composites the accumulated transparency over `target`
    /// (SrcAlpha / OneMinusSrcAlpha, like the HDR pipeline's OIT composite).
    pub fn composite(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let ctxt = Context::get();
        let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reflector_oit_composite_bind_group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.accum_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.reveal_view),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("reflector_oit_composite_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..4, 0..1);
    }
}
