//! Reflection probes: localized, parallax-corrected environment maps.
//!
//! A reflection probe captures the surrounding scene (or a baked HDR) into an
//! equirectangular map and lets nearby reflective surfaces sample *that* instead
//! of only the single global skybox, with **parallax correction**: the mirror
//! direction is intersected with the probe's bounding box so the reflection
//! tracks the room geometry rather than appearing infinitely far away.
//!
//! All probes share one mip-chained equirectangular `texture_2d_array` (one layer
//! per probe), reusing the "mip-as-prefilter" approach of [`EnvironmentMap`]: the
//! coarser mips stand in for rougher pre-filtered reflections. The probe records
//! (position, parallax box, intensity, …) are uploaded into the material's frame
//! uniform as a small fixed-size array, so probes need no storage buffers and work
//! on WebGL2.
//!
//! Content can come from two sources:
//! - a baked equirectangular HDR image ([`ReflectionProbes::set_image`]), or
//! - a runtime capture of the live scene (driven by the window; see
//!   `Window::capture_reflection_probe`).

use crate::context::Context;

/// Maximum number of simultaneous reflection probes. Must match `MAX_PROBES` in
/// `builtin/default.wgsl` and `builtin/object_material.rs`.
pub const MAX_PROBES: usize = 8;

/// Equirectangular resolution of each probe layer. A 2:1 map; all layers share
/// this size and mip count (a `texture_2d_array` requires uniform dimensions).
pub const PROBE_WIDTH: u32 = 256;
/// See [`PROBE_WIDTH`].
pub const PROBE_HEIGHT: u32 = 128;

/// A single reflection probe's placement and influence.
#[derive(Copy, Clone, Debug)]
pub struct ReflectionProbe {
    /// World-space center of the probe (the capture viewpoint).
    pub center: glamx::Vec3,
    /// Half-extents of the parallax/influence box (world axis-aligned), centered
    /// on `center`. The mirror ray is intersected with this box for parallax
    /// correction, and a fragment is influenced when inside it.
    pub half_extents: glamx::Vec3,
    /// Width (in world units) of the soft edge over which the probe fades out at
    /// the box boundary, blending back to the global environment.
    pub falloff: f32,
    /// Luminance multiplier applied to this probe's samples.
    pub intensity: f32,
    /// Y-axis rotation (radians) applied when sampling, matching the skybox/IBL
    /// convention. Baked maps that share the skybox orientation use `0.0`.
    pub rotation: f32,
}

impl Default for ReflectionProbe {
    fn default() -> Self {
        ReflectionProbe {
            center: glamx::Vec3::ZERO,
            half_extents: glamx::Vec3::splat(5.0),
            falloff: 0.5,
            intensity: 1.0,
            rotation: 0.0,
        }
    }
}

impl ReflectionProbe {
    /// A probe centered at `center` with a cubic influence box of the given
    /// half-size.
    pub fn new(center: glamx::Vec3, half_extent: f32) -> Self {
        ReflectionProbe {
            center,
            half_extents: glamx::Vec3::splat(half_extent),
            ..Default::default()
        }
    }
}

/// Manages the shared probe array texture and the list of active probes.
pub struct ReflectionProbes {
    // Mip-chained equirectangular array; one layer per probe slot.
    texture: wgpu::Texture,
    /// Array view (all layers, all mips) bound by the lighting shaders.
    array_view: wgpu::TextureView,
    /// Trilinear sampler (repeat U / clamp V), matching the equirect convention.
    sampler: wgpu::Sampler,
    mip_count: u32,
    /// Active probes, in layer order (index == array layer).
    probes: Vec<ReflectionProbe>,
}

impl ReflectionProbes {
    /// Creates an empty probe set (no probes; the array is allocated up front).
    pub fn new() -> ReflectionProbes {
        let ctxt = Context::get();
        let mip_count = (32 - PROBE_WIDTH.max(PROBE_HEIGHT).leading_zeros()).max(1);
        let texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("reflection_probe_array"),
            size: wgpu::Extent3d {
                width: PROBE_WIDTH,
                height: PROBE_HEIGHT,
                depth_or_array_layers: MAX_PROBES as u32,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("reflection_probe_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("reflection_probe_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        ReflectionProbes {
            texture,
            array_view,
            sampler,
            mip_count,
            probes: Vec::new(),
        }
    }

    /// Number of active probes.
    pub fn len(&self) -> usize {
        self.probes.len()
    }

    /// Whether there are no active probes.
    pub fn is_empty(&self) -> bool {
        self.probes.is_empty()
    }

    /// The probe array view bound by the lighting shaders (all layers + mips).
    pub fn array_view(&self) -> &wgpu::TextureView {
        &self.array_view
    }

    /// The probe sampler (equirect repeat-U / clamp-V trilinear).
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// The maximum sampleable LOD (used to map roughness → mip).
    pub fn max_lod(&self) -> f32 {
        (self.mip_count.max(1) - 1) as f32
    }

    /// The active probes, in layer order.
    pub fn probes(&self) -> &[ReflectionProbe] {
        &self.probes
    }

    /// Mutable access to a probe (e.g. to move it before re-capturing).
    pub fn probe_mut(&mut self, idx: usize) -> Option<&mut ReflectionProbe> {
        self.probes.get_mut(idx)
    }

    /// The probe array texture (for the window's capture pass to render into a
    /// specific layer).
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Number of mip levels in the probe array.
    pub fn mip_count(&self) -> u32 {
        self.mip_count
    }

    /// A 2D render-target view of layer `idx`, mip 0 — the destination for a
    /// capture's cube→equirect reprojection.
    pub fn layer_mip0_view(&self, idx: usize) -> wgpu::TextureView {
        self.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("reflection_probe_layer_mip0"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: idx as u32,
            array_layer_count: Some(1),
            ..Default::default()
        })
    }

    /// Registers a probe and returns its index/layer (up to [`MAX_PROBES`]).
    /// The layer's content starts black until [`set_image`](Self::set_image) or a
    /// runtime capture fills it.
    pub fn add(&mut self, probe: ReflectionProbe) -> Option<usize> {
        if self.probes.len() >= MAX_PROBES {
            return None;
        }
        let idx = self.probes.len();
        self.probes.push(probe);
        Some(idx)
    }

    /// Fills probe `idx` from a baked equirectangular HDR image (resized to the
    /// probe resolution) and regenerates its mip chain.
    pub fn set_image(&mut self, idx: usize, img: &image::DynamicImage) {
        if idx >= self.probes.len() {
            return;
        }
        // Resize to the shared probe resolution (array layers must match).
        let resized = img.resize_exact(
            PROBE_WIDTH,
            PROBE_HEIGHT,
            image::imageops::FilterType::Triangle,
        );
        let rgba = resized.to_rgba32f();
        let halves: Vec<u16> = rgba.as_raw().iter().map(|&v| f32_to_f16(v)).collect();

        let ctxt = Context::get();
        ctxt.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: idx as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&halves),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(PROBE_WIDTH * 8),
                rows_per_image: Some(PROBE_HEIGHT),
            },
            wgpu::Extent3d {
                width: PROBE_WIDTH,
                height: PROBE_HEIGHT,
                depth_or_array_layers: 1,
            },
        );

        let mut encoder = ctxt.create_command_encoder(Some("reflection_probe_mipgen"));
        self.generate_layer_mips(&mut encoder, idx, None);
        ctxt.submit(std::iter::once(encoder.finish()));
    }

    /// Renders the mip chain of one array layer by box-downsampling each level
    /// from the previous one (mip 0 must already be populated). Recorded into the
    /// supplied encoder so it can be folded into the frame (used by runtime
    /// capture) or submitted standalone (used by [`set_image`](Self::set_image)).
    pub(crate) fn generate_layer_mips(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: usize,
        mut gpu: Option<&mut crate::renderer::timings::GpuTimer>,
    ) {
        if self.mip_count <= 1 || layer >= MAX_PROBES {
            return;
        }
        let ctxt = Context::get();
        let pipeline = Self::downsample_pipeline();
        let layout = pipeline.get_bind_group_layout(0);
        for mip in 1..self.mip_count {
            let src_view = self.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("reflection_probe_mip_src"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: mip - 1,
                mip_level_count: Some(1),
                base_array_layer: layer as u32,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let dst_view = self.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("reflection_probe_mip_dst"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: layer as u32,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reflection_probe_downsample_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            let mip_ts = gpu.as_deref_mut().and_then(|g| g.render_scope("probe"));
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("reflection_probe_downsample_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: mip_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Builds (and caches) the fullscreen box-downsample pipeline shared by mip
    /// generation. Reuses `env_downsample.wgsl`.
    fn downsample_pipeline() -> wgpu::RenderPipeline {
        let ctxt = Context::get();
        let shader = ctxt.create_shader_module(
            Some("reflection_probe_downsample"),
            &crate::builtin::compile_shader_with_common(
                "package::env_downsample",
                crate::builtin::ENV_DOWNSAMPLE_WESL,
            ),
        );
        let layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reflection_probe_downsample_layout"),
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
        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reflection_probe_downsample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("reflection_probe_downsample_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
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
        })
    }
}

impl Default for ReflectionProbes {
    fn default() -> Self {
        Self::new()
    }
}

// === Runtime capture ===

/// Per-face look directions and up hints for cube capture. MUST match the face
/// layout in `builtin/cube_to_equirect.wgsl` (face `i` looks along `FACE_FORWARD[i]`
/// with up hint `FACE_UP[i]`).
const FACE_FORWARD: [[f32; 3]; 6] = [
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
];
const FACE_UP: [[f32; 3]; 6] = [
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
];

/// A throwaway camera for one cube face of a reflection-probe capture: a 90°-FOV,
/// unit-aspect, right-handed view from the probe center along one of the six axes.
pub struct CubeFaceCamera {
    eye: glamx::Vec3,
    view: glamx::Pose3,
    proj: glamx::Mat4,
    znear: f32,
    zfar: f32,
}

impl CubeFaceCamera {
    /// Builds the capture camera for `face` (0..6) at `eye`.
    pub fn new(eye: glamx::Vec3, face: usize, znear: f32, zfar: f32) -> CubeFaceCamera {
        let f = FACE_FORWARD[face];
        let u = FACE_UP[face];
        let fwd = glamx::Vec3::new(f[0], f[1], f[2]);
        let up = glamx::Vec3::new(u[0], u[1], u[2]);
        let proj = glamx::Mat4::perspective_rh_gl(core::f32::consts::FRAC_PI_2, 1.0, znear, zfar);
        let view = glamx::Pose3::look_at_rh(eye, eye + fwd, up);
        CubeFaceCamera {
            eye,
            view,
            proj,
            znear,
            zfar,
        }
    }
}

impl crate::camera::Camera3d for CubeFaceCamera {
    fn handle_event(&mut self, _: &crate::window::Canvas, _: &crate::event::WindowEvent) {}
    fn update(&mut self, _: &crate::window::Canvas) {}
    fn eye(&self) -> glamx::Vec3 {
        self.eye
    }
    fn view_transform(&self) -> glamx::Pose3 {
        self.view
    }
    fn transformation(&self) -> glamx::Mat4 {
        self.proj * self.view.to_mat4()
    }
    fn inverse_transformation(&self) -> glamx::Mat4 {
        self.transformation().inverse()
    }
    fn clip_planes(&self) -> (f32, f32) {
        (self.znear, self.zfar)
    }
    fn view_transform_pair(&self, _pass: usize) -> (glamx::Pose3, glamx::Mat4) {
        (self.view, self.proj)
    }
}

/// Reusable GPU targets + reprojection pipeline for runtime probe capture: a
/// six-layer color array (one per cube face), a shared face depth buffer, and the
/// cube→equirect reprojection pass.
pub struct ProbeCapture {
    size: u32,
    _color: wgpu::Texture,
    face_views: Vec<wgpu::TextureView>,
    array_view: wgpu::TextureView,
    _depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    reproject_pipeline: wgpu::RenderPipeline,
    reproject_layout: wgpu::BindGroupLayout,
}

impl ProbeCapture {
    /// Allocates capture targets at the given per-face resolution.
    pub fn new(size: u32) -> ProbeCapture {
        let ctxt = Context::get();
        let color = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("probe_capture_color"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let face_views: Vec<_> = (0..6)
            .map(|i| {
                color.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("probe_capture_face"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let array_view = color.create_view(&wgpu::TextureViewDescriptor {
            label: Some("probe_capture_array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let depth = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("probe_capture_depth"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
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
        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("probe_capture_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let shader = ctxt.create_shader_module(
            Some("cube_to_equirect"),
            &crate::builtin::compile_shader_with_common(
                "package::cube_to_equirect",
                include_str!("../builtin/cube_to_equirect.wgsl"),
            ),
        );
        let reproject_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe_reproject_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
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
        let pl = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe_reproject_pipeline_layout"),
            bind_group_layouts: &[Some(&reproject_layout)],
            immediate_size: 0,
        });
        let reproject_pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("probe_reproject_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
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

        ProbeCapture {
            size,
            _color: color,
            face_views,
            array_view,
            _depth: depth,
            depth_view,
            sampler,
            reproject_pipeline,
            reproject_layout,
        }
    }

    /// Per-face resolution.
    pub fn size(&self) -> u32 {
        self.size
    }

    /// The color render target for cube face `face` (0..6).
    pub fn face_color_view(&self, face: usize) -> &wgpu::TextureView {
        &self.face_views[face]
    }

    /// The shared face depth buffer (cleared per face by the capture pass).
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// Reprojects the six captured faces into `dst` (a probe layer's mip-0 view).
    pub(crate) fn reproject(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::TextureView,
        gpu: &mut crate::renderer::timings::GpuTimer,
    ) {
        let ctxt = Context::get();
        let bg = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("probe_reproject_bg"),
            layout: &self.reproject_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.array_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let reproject_ts = gpu.render_scope("probe");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("probe_reproject_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: reproject_ts,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.reproject_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Converts an `f32` to IEEE-754 half-precision bits (truncating mantissa).
/// Mirrors the helper in [`crate::renderer::ibl`].
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    if exp <= 0 {
        sign
    } else if exp >= 0x1f {
        sign | 0x7c00
    } else {
        sign | ((exp as u16) << 10) | (mant as u16)
    }
}
