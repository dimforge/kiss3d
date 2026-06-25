//! Screen-space 2D global illumination ([`Gi2d`]).
//!
//! A post-processing effect that lights the scene by ray-marching incoming light
//! against analytic emitter/occluder discs, giving soft shadows and colored light
//! bleed. This is the brute-force form of the technique that *radiance cascades*
//! accelerate.
//!
//! To stay real-time it runs in two passes with three optimizations:
//! 1. the irradiance field is computed at a fraction of screen resolution
//!    ([`set_resolution_scale`](Gi2d::set_resolution_scale)) and bilinearly upsampled
//!    when composited — GI is low-frequency, so the quality cost is small;
//! 2. **temporal accumulation** ([`set_temporal_blend`](Gi2d::set_temporal_blend)):
//!    the ray fan is rotated each frame and blended with the reprojected previous
//!    frame, so a handful of rays converge to a smooth result;
//! 3. a **blue-noise-jittered**, distance-capped, step-capped march
//!    ([`set_rays`](Gi2d::set_rays), [`set_max_distance`](Gi2d::set_max_distance),
//!    [`set_max_steps`](Gi2d::set_max_steps)).
//!
//! Optionally ([`set_sdf_occluders`](Gi2d::set_sdf_occluders)) the occluders are
//! baked into a distance field each frame with the **jump-flood algorithm**, so the
//! march costs one texture fetch per step regardless of occluder count (vs. a loop
//! over every disc). This is screen-space — occluders outside the view don't cast
//! shadows — so it's best with many on-screen occluders; the default analytic path
//! stays global.
//!
//! Apply it with [`Window::render_2d_with`](crate::window::Window::render_2d_with) or
//! in a chain; each frame set the camera and the emitters/occluders.

use crate::camera::Camera2d;
use crate::color::Color;
use crate::context::Context;
use crate::post_processing::post_processing_effect::{PostProcessingContext, PostProcessingEffect};
use crate::post_processing::HDR_FORMAT;
use crate::resource::RenderTarget;
use bytemuck::{Pod, Zeroable};
use glamx::{Mat3, Vec2};

/// Maximum number of emitter discs (matches `MAX_EMITTERS` in `gi2d_field.wgsl`).
pub const MAX_EMITTERS: usize = 32;
/// Maximum number of occluder discs (matches `MAX_OCCLUDERS` in `gi2d_field.wgsl`).
pub const MAX_OCCLUDERS: usize = 64;

const SEED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const SDF_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// A disc that emits light into the 2D GI solution.
#[derive(Copy, Clone, Debug)]
pub struct GiEmitter2d {
    /// World-space center.
    pub position: Vec2,
    /// Disc radius (its size as an area light).
    pub radius: f32,
    /// Emitted color.
    pub color: Color,
    /// Radiance multiplier.
    pub intensity: f32,
}

impl GiEmitter2d {
    /// A new emitter disc.
    pub fn new(position: Vec2, radius: f32, color: Color, intensity: f32) -> Self {
        GiEmitter2d {
            position,
            radius,
            color,
            intensity,
        }
    }
}

/// A disc that blocks light (casts shadows) in the 2D GI solution.
#[derive(Copy, Clone, Debug)]
pub struct GiOccluder2d {
    /// World-space center.
    pub position: Vec2,
    /// Disc radius.
    pub radius: f32,
}

impl GiOccluder2d {
    /// A new occluder disc.
    pub fn new(position: Vec2, radius: f32) -> Self {
        GiOccluder2d { position, radius }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuEmitter {
    pos_radius: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuOccluder {
    pos_radius: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FieldUniforms {
    inv_vp: [[f32; 4]; 3],
    prev_vp: [[f32; 4]; 3],
    cur_vp: [[f32; 4]; 3],
    params: [f32; 4],
    flags: [f32; 4],
    counts: [f32; 4],
    emitters: [GpuEmitter; MAX_EMITTERS],
    occluders: [GpuOccluder; MAX_OCCLUDERS],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct JfaUniforms {
    inv_vp: [[f32; 4]; 3],
    aux: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CompositeUniforms {
    ambient: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CascadeParams {
    v0: [f32; 4],
    v1: [f32; 4],
    v2: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CascadeCompositeUniforms {
    v0: [f32; 4],
    v1: [f32; 4],
    ambient: [f32; 4],
}

/// Maximum number of radiance-cascade levels (sizes the per-level uniform pool).
const MAX_CASCADES: usize = 8;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QuadVertex {
    position: [f32; 2],
}

fn mat3_to_padded(m: &Mat3) -> [[f32; 4]; 3] {
    let c = m.to_cols_array_2d();
    [
        [c[0][0], c[0][1], c[0][2], 0.0],
        [c[1][0], c[1][1], c[1][2], 0.0],
        [c[2][0], c[2][1], c[2][2], 0.0],
    ]
}

/// A render-target + sampleable texture used for a GI buffer.
struct GiTexture {
    view: wgpu::TextureView,
}

impl GiTexture {
    fn new(width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let ctxt = Context::get();
        let texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("gi2d_texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        GiTexture {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }
}

/// Screen-space 2D global-illumination post-processing effect (see the [module docs](self)).
pub struct Gi2d {
    field_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    jfa_seed_pipeline: wgpu::RenderPipeline,
    jfa_step_pipeline: wgpu::RenderPipeline,
    jfa_resolve_pipeline: wgpu::RenderPipeline,
    tex_bind_group_layout: wgpu::BindGroupLayout,
    tex_only_bind_group_layout: wgpu::BindGroupLayout,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    field_uniform_buffer: wgpu::Buffer,
    field_uniform_bind_group: wgpu::BindGroup,
    composite_uniform_buffer: wgpu::Buffer,
    composite_uniform_bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    vertex_buffer: wgpu::Buffer,

    // Ping-pong GI history textures (low resolution).
    history: [GiTexture; 2],
    cur: usize,
    gi_size: (u32, u32),
    history_valid: bool,
    frame_index: u32,

    // Jump-flood occluder SDF resources (low resolution).
    seed: [GiTexture; 2],
    sdf: GiTexture,
    jfa_step_buffers: Vec<wgpu::Buffer>,
    jfa_step_bind_groups: Vec<wgpu::BindGroup>,
    jfa_resolve_buffer: wgpu::Buffer,
    jfa_resolve_bind_group: wgpu::BindGroup,
    jfa_passes: u32,

    // Radiance-cascade resources.
    cascade_pipeline: wgpu::RenderPipeline,
    cascade_composite_pipeline: wgpu::RenderPipeline,
    cascade: [GiTexture; 2],
    /// Current allocated size of the (decoupled) cascade textures.
    cascade_tex_size: (u32, u32),
    cascade_param_buffers: Vec<wgpu::Buffer>,
    cascade_param_bind_groups: Vec<wgpu::BindGroup>,
    cascade_composite_buffer: wgpu::Buffer,
    cascade_composite_bind_group: wgpu::BindGroup,
    radiance_cascades: bool,
    cascade_count: u32,
    /// Cascade-0 direction-tile edge: base direction count is `base_block^2`. Larger →
    /// finer angular resolution → sharper shadow contours. Decoupled from probe spacing.
    base_block: u32,
    /// Cascade-0 probe spacing in field pixels (independent of `base_block`).
    probe_spacing: u32,

    // Parameters.
    inv_vp: Mat3,
    vp: Mat3,
    prev_vp: Mat3,
    ambient: Color,
    num_rays: u32,
    max_distance: f32,
    max_steps: u32,
    resolution_scale: u32,
    temporal_blend: f32,
    sdf_occluders: bool,
    emitters: Vec<GiEmitter2d>,
    occluders: Vec<GiOccluder2d>,
}

impl Default for Gi2d {
    fn default() -> Self {
        Self::new()
    }
}

impl Gi2d {
    /// Creates a new GI effect with real-time defaults: half-resolution field, 8 rays
    /// per pixel, temporal accumulation, analytic (global) occluders.
    pub fn new() -> Gi2d {
        let ctxt = Context::get();

        let tex_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gi2d_tex_bind_group_layout"),
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

        // Texture-only layout for the jump-flood seed reads (sampled via textureLoad,
        // so the unfilterable Rgba32Float seed needs no sampler).
        let tex_only_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gi2d_tex_only_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gi2d_uniform_bind_group_layout"),
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

        let field_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gi2d_field_pipeline_layout"),
            bind_group_layouts: &[
                Some(&tex_bind_group_layout),
                Some(&uniform_bind_group_layout),
                Some(&tex_bind_group_layout),
            ],
            immediate_size: 0,
        });
        let composite_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gi2d_composite_pipeline_layout"),
            bind_group_layouts: &[
                Some(&tex_bind_group_layout),
                Some(&tex_bind_group_layout),
                Some(&uniform_bind_group_layout),
            ],
            immediate_size: 0,
        });
        let jfa_seed_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gi2d_jfa_seed_pipeline_layout"),
            bind_group_layouts: &[Some(&uniform_bind_group_layout)],
            immediate_size: 0,
        });
        let jfa_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gi2d_jfa_pipeline_layout"),
            bind_group_layouts: &[
                Some(&tex_only_bind_group_layout),
                Some(&uniform_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let field_shader = ctxt.create_shader_module(
            Some("gi2d_field_shader"),
            &crate::builtin::compile_shader_with_common(
                "package::gi2d_field",
                include_str!("../builtin/gi2d_field.wgsl"),
            ),
        );
        let composite_shader = ctxt.create_shader_module(
            Some("gi2d_composite_shader"),
            &crate::builtin::compile_shader_with_common(
                "package::gi2d_composite",
                include_str!("../builtin/gi2d_composite.wgsl"),
            ),
        );
        let jfa_seed_shader = ctxt.create_shader_module(
            Some("gi2d_jfa_seed_shader"),
            &crate::builtin::compile_shader_with_common(
                "package::gi2d_jfa_seed",
                include_str!("../builtin/gi2d_jfa_seed.wgsl"),
            ),
        );
        let jfa_shader = ctxt.create_shader_module(
            Some("gi2d_jfa_shader"),
            &crate::builtin::compile_shader_with_common(
                "package::gi2d_jfa",
                include_str!("../builtin/gi2d_jfa.wgsl"),
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

        let make_pipeline = |label: &str,
                             layout: &wgpu::PipelineLayout,
                             shader: &wgpu::ShaderModule,
                             fs_entry: &str,
                             format: wgpu::TextureFormat| {
            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_buffer_layout.clone()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some(fs_entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
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
            })
        };

        let field_pipeline = make_pipeline(
            "gi2d_field_pipeline",
            &field_layout,
            &field_shader,
            "fs_main",
            HDR_FORMAT,
        );
        let composite_pipeline = make_pipeline(
            "gi2d_composite_pipeline",
            &composite_layout,
            &composite_shader,
            "fs_main",
            ctxt.surface_format,
        );
        let jfa_seed_pipeline = make_pipeline(
            "gi2d_jfa_seed_pipeline",
            &jfa_seed_layout,
            &jfa_seed_shader,
            "fs_main",
            SEED_FORMAT,
        );
        let jfa_step_pipeline = make_pipeline(
            "gi2d_jfa_step_pipeline",
            &jfa_layout,
            &jfa_shader,
            "fs_step",
            SEED_FORMAT,
        );
        let jfa_resolve_pipeline = make_pipeline(
            "gi2d_jfa_resolve_pipeline",
            &jfa_layout,
            &jfa_shader,
            "fs_resolve",
            SDF_FORMAT,
        );

        // Radiance-cascade pipelines.
        let cascade_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gi2d_cascade_pipeline_layout"),
            bind_group_layouts: &[
                Some(&tex_only_bind_group_layout), // upper cascade (textureLoad)
                Some(&uniform_bind_group_layout),  // per-cascade params
                Some(&uniform_bind_group_layout),  // shared field uniforms (scene)
                Some(&tex_bind_group_layout),      // occluder SDF (when enabled)
            ],
            immediate_size: 0,
        });
        let cascade_composite_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gi2d_cascade_composite_pipeline_layout"),
            bind_group_layouts: &[
                Some(&tex_bind_group_layout),      // scene
                Some(&tex_only_bind_group_layout), // cascade 0 (textureLoad)
                Some(&uniform_bind_group_layout),  // composite uniforms
            ],
            immediate_size: 0,
        });
        let cascade_shader = ctxt.create_shader_module(
            Some("gi2d_cascade_shader"),
            &crate::builtin::compile_shader_with_common(
                "package::gi2d_cascade",
                include_str!("../builtin/gi2d_cascade.wgsl"),
            ),
        );
        let cascade_composite_shader = ctxt.create_shader_module(
            Some("gi2d_cascade_composite_shader"),
            &crate::builtin::compile_shader_with_common(
                "package::gi2d_cascade_composite",
                include_str!("../builtin/gi2d_cascade_composite.wgsl"),
            ),
        );
        let cascade_pipeline = make_pipeline(
            "gi2d_cascade_pipeline",
            &cascade_layout,
            &cascade_shader,
            "fs_main",
            HDR_FORMAT,
        );
        let cascade_composite_pipeline = make_pipeline(
            "gi2d_cascade_composite_pipeline",
            &cascade_composite_layout,
            &cascade_composite_shader,
            "fs_main",
            ctxt.surface_format,
        );

        // Per-level cascade uniform pool (filled each frame for the levels in use).
        let mut cascade_param_buffers = Vec::with_capacity(MAX_CASCADES);
        let mut cascade_param_bind_groups = Vec::with_capacity(MAX_CASCADES);
        for _ in 0..MAX_CASCADES {
            let buf = ctxt.create_buffer_simple(
                Some("gi2d_cascade_param_buffer"),
                std::mem::size_of::<CascadeParams>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let bg = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gi2d_cascade_param_bind_group"),
                layout: &uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            cascade_param_buffers.push(buf);
            cascade_param_bind_groups.push(bg);
        }
        let cascade_composite_buffer = ctxt.create_buffer_simple(
            Some("gi2d_cascade_composite_buffer"),
            std::mem::size_of::<CascadeCompositeUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let cascade_composite_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi2d_cascade_composite_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cascade_composite_buffer.as_entire_binding(),
            }],
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
            Some("gi2d_vertex_buffer"),
            bytemuck::cast_slice(&vertices),
            wgpu::BufferUsages::VERTEX,
        );

        let field_uniform_buffer = ctxt.create_buffer_simple(
            Some("gi2d_field_uniform_buffer"),
            std::mem::size_of::<FieldUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let field_uniform_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi2d_field_uniform_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: field_uniform_buffer.as_entire_binding(),
            }],
        });
        let composite_uniform_buffer = ctxt.create_buffer_simple(
            Some("gi2d_composite_uniform_buffer"),
            std::mem::size_of::<CompositeUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let composite_uniform_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi2d_composite_uniform_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: composite_uniform_buffer.as_entire_binding(),
            }],
        });
        // Resolve uses one fixed uniform buffer (step is unused for the resolve pass).
        let jfa_resolve_buffer = ctxt.create_buffer_simple(
            Some("gi2d_jfa_resolve_buffer"),
            std::mem::size_of::<JfaUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let jfa_resolve_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi2d_jfa_resolve_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: jfa_resolve_buffer.as_entire_binding(),
            }],
        });

        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gi2d_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Gi2d {
            field_pipeline,
            composite_pipeline,
            jfa_seed_pipeline,
            jfa_step_pipeline,
            jfa_resolve_pipeline,
            tex_bind_group_layout,
            tex_only_bind_group_layout,
            uniform_bind_group_layout,
            field_uniform_buffer,
            field_uniform_bind_group,
            composite_uniform_buffer,
            composite_uniform_bind_group,
            sampler,
            vertex_buffer,
            history: [
                GiTexture::new(1, 1, HDR_FORMAT),
                GiTexture::new(1, 1, HDR_FORMAT),
            ],
            cur: 0,
            gi_size: (0, 0),
            history_valid: false,
            frame_index: 0,
            seed: [
                GiTexture::new(1, 1, SEED_FORMAT),
                GiTexture::new(1, 1, SEED_FORMAT),
            ],
            sdf: GiTexture::new(1, 1, SDF_FORMAT),
            jfa_step_buffers: Vec::new(),
            jfa_step_bind_groups: Vec::new(),
            jfa_resolve_buffer,
            jfa_resolve_bind_group,
            jfa_passes: 0,
            cascade_pipeline,
            cascade_composite_pipeline,
            cascade: [
                GiTexture::new(1, 1, HDR_FORMAT),
                GiTexture::new(1, 1, HDR_FORMAT),
            ],
            cascade_tex_size: (0, 0),
            cascade_param_buffers,
            cascade_param_bind_groups,
            cascade_composite_buffer,
            cascade_composite_bind_group,
            radiance_cascades: false,
            cascade_count: 5,
            // 16 base directions (4x4 tile) with a fine 2px probe grid — decoupled, so
            // sharp shadow edges without coarsening the probe grid (the cascade
            // textures grow by base_block/probe_spacing in each axis instead).
            base_block: 4,
            probe_spacing: 2,
            inv_vp: Mat3::IDENTITY,
            vp: Mat3::IDENTITY,
            prev_vp: Mat3::IDENTITY,
            ambient: Color::new(0.08, 0.08, 0.1, 1.0),
            num_rays: 8,
            max_distance: 2000.0,
            max_steps: 32,
            resolution_scale: 2,
            temporal_blend: 0.85,
            sdf_occluders: false,
            emitters: Vec::new(),
            occluders: Vec::new(),
        }
    }

    /// Captures `camera`'s view-projection so the GI pass can reconstruct world space
    /// and reproject the temporal history. Call once per frame before rendering.
    pub fn set_camera(&mut self, camera: &impl Camera2d) {
        let (view, proj) = camera.view_transform_pair();
        self.vp = proj * view;
        self.inv_vp = self.vp.inverse();
    }

    /// Replaces the emitter discs (truncated to [`MAX_EMITTERS`]).
    pub fn set_emitters(&mut self, emitters: &[GiEmitter2d]) {
        self.emitters.clear();
        self.emitters
            .extend(emitters.iter().take(MAX_EMITTERS).copied());
    }

    /// Replaces the occluder discs (truncated to [`MAX_OCCLUDERS`]).
    pub fn set_occluders(&mut self, occluders: &[GiOccluder2d]) {
        self.occluders.clear();
        self.occluders
            .extend(occluders.iter().take(MAX_OCCLUDERS).copied());
    }

    /// Sets the scene-wide ambient term applied where no light reaches.
    pub fn set_ambient(&mut self, ambient: Color) {
        self.ambient = ambient;
    }

    /// Sets the number of rays cast per pixel *per frame*. With temporal accumulation
    /// enabled the effective sample count is far higher, so this can stay small.
    pub fn set_rays(&mut self, rays: u32) {
        self.num_rays = rays.max(1);
    }

    /// Sets the maximum ray-march distance in world units (cap it to the scene bounds).
    pub fn set_max_distance(&mut self, distance: f32) {
        self.max_distance = distance;
    }

    /// Sets the maximum number of sphere-trace steps per ray.
    pub fn set_max_steps(&mut self, steps: u32) {
        self.max_steps = steps.max(1);
    }

    /// Sets the irradiance-field downscale factor: 1 = full resolution, 2 = half (the
    /// default), 4 = quarter, … Larger is faster and softer. Changing it rebuilds the
    /// field textures.
    pub fn set_resolution_scale(&mut self, scale: u32) {
        self.resolution_scale = scale.max(1);
    }

    /// Sets the temporal-accumulation blend (fraction of the reprojected previous
    /// frame kept each frame), in `[0, 1)`. 0 disables accumulation; higher is
    /// smoother but ghosts more on fast motion. Default 0.85.
    pub fn set_temporal_blend(&mut self, blend: f32) {
        self.temporal_blend = blend.clamp(0.0, 0.99);
    }

    /// Switches the solver to **radiance cascades**: instead of marching long rays
    /// per pixel, light is gathered into a hierarchy of probe grids (coarser probes
    /// with more directions and longer rays per level) and merged top-down into a
    /// final irradiance, which is far cheaper for the same coverage. Off by default
    /// (the direct ray-march). Temporal accumulation is unused on this path.
    pub fn set_radiance_cascades(&mut self, enabled: bool) {
        self.radiance_cascades = enabled;
    }

    /// Sets the number of radiance-cascade levels (clamped to what the field
    /// resolution supports). More levels reach farther light. Default 5.
    pub fn set_cascade_count(&mut self, count: u32) {
        self.cascade_count = count.clamp(1, MAX_CASCADES as u32);
    }

    /// Sets the cascade-0 base direction count (radiance cascades only), rounded to a
    /// supported value. More directions sharpen shadow contours; the probe grid stays
    /// fine (the cascade textures grow instead). Default 16; typical values 4 / 16 / 64.
    pub fn set_cascade_base_directions(&mut self, directions: u32) {
        let edge = (directions as f32).sqrt().round() as u32;
        self.base_block = edge.max(2).next_power_of_two();
    }

    /// Enables baking occluders into a jump-flooded distance field each frame so the
    /// march cost is independent of occluder count (one texture fetch per step). This
    /// is screen-space — off-screen occluders cast no shadows. Off by default (the
    /// analytic, global per-disc path). See the [module docs](self).
    pub fn set_sdf_occluders(&mut self, enabled: bool) {
        self.sdf_occluders = enabled;
    }

    /// (Re)creates the per-size textures and jump-flood pass buffers, resetting
    /// temporal history when the size changes.
    fn ensure_textures(&mut self, width: u32, height: u32) {
        let cw = (width / self.resolution_scale).max(1);
        let ch = (height / self.resolution_scale).max(1);
        if self.gi_size == (cw, ch) {
            return;
        }
        let ctxt = Context::get();

        self.history = [
            GiTexture::new(cw, ch, HDR_FORMAT),
            GiTexture::new(cw, ch, HDR_FORMAT),
        ];
        self.seed = [
            GiTexture::new(cw, ch, SEED_FORMAT),
            GiTexture::new(cw, ch, SEED_FORMAT),
        ];
        self.sdf = GiTexture::new(cw, ch, SDF_FORMAT);
        self.gi_size = (cw, ch);
        self.history_valid = false;
        self.cur = 0;

        // One jump-flood pass per halving step, down to step 1.
        let maxdim = cw.max(ch);
        let passes = (32 - (maxdim.max(1) - 1).leading_zeros()).max(1); // ceil(log2(maxdim))
        self.jfa_passes = passes;
        self.jfa_step_buffers.clear();
        self.jfa_step_bind_groups.clear();
        for _ in 0..passes {
            let buf = ctxt.create_buffer_simple(
                Some("gi2d_jfa_step_buffer"),
                std::mem::size_of::<JfaUniforms>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let bg = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gi2d_jfa_step_bind_group"),
                layout: &self.uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            self.jfa_step_buffers.push(buf);
            self.jfa_step_bind_groups.push(bg);
        }
    }

    fn build_field_uniforms(&self) -> FieldUniforms {
        let mut emitters = [GpuEmitter::zeroed(); MAX_EMITTERS];
        for (slot, e) in emitters.iter_mut().zip(self.emitters.iter()) {
            slot.pos_radius = [e.position.x, e.position.y, e.radius, e.intensity];
            slot.color = [e.color.r, e.color.g, e.color.b, 0.0];
        }
        let mut occluders = [GpuOccluder::zeroed(); MAX_OCCLUDERS];
        for (slot, o) in occluders.iter_mut().zip(self.occluders.iter()) {
            slot.pos_radius = [o.position.x, o.position.y, o.radius, 0.0];
        }
        FieldUniforms {
            inv_vp: mat3_to_padded(&self.inv_vp),
            prev_vp: mat3_to_padded(&self.prev_vp),
            cur_vp: mat3_to_padded(&self.vp),
            params: [
                self.num_rays as f32,
                (self.frame_index % 4096) as f32,
                self.temporal_blend,
                if self.history_valid { 1.0 } else { 0.0 },
            ],
            // flags: use_sdf, sdf_bias (world units ≈ a couple field texels). The
            // jump-flood field is unsigned (0 inside occluders), so we shift its
            // zero-crossing slightly outside the true surface — making it
            // effectively signed near the boundary so the march's `d <= 0` blocked
            // test fires reliably instead of letting grazing rays tunnel through.
            flags: [
                if self.sdf_occluders { 1.0 } else { 0.0 },
                {
                    let cw = self.gi_size.0.max(1) as f32;
                    let wpp = (2.0 / cw) / self.vp.x_axis.x.abs().max(1e-6);
                    wpp * 2.0
                },
                0.0,
                0.0,
            ],
            counts: [
                self.emitters.len().min(MAX_EMITTERS) as f32,
                self.occluders.len().min(MAX_OCCLUDERS) as f32,
                self.max_distance,
                self.max_steps as f32,
            ],
            emitters,
            occluders,
        }
    }

    fn tex_bind_group(&self, view: &wgpu::TextureView, label: &str) -> wgpu::BindGroup {
        Context::get().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.tex_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    fn tex_only_bind_group(&self, view: &wgpu::TextureView) -> wgpu::BindGroup {
        Context::get().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi2d_seed_bind_group"),
            layout: &self.tex_only_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            }],
        })
    }

    fn fullscreen_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        target: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
        bind_groups: &[&wgpu::BindGroup],
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
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
        pass.set_pipeline(pipeline);
        for (i, bg) in bind_groups.iter().enumerate() {
            pass.set_bind_group(i as u32, *bg, &[]);
        }
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..4, 0..1);
    }

    /// Bakes the occluder distance field for this frame (seed → jump-flood → resolve).
    fn build_sdf(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let ctxt = Context::get();
        let (cw, ch) = self.gi_size;
        let inv_vp = mat3_to_padded(&self.inv_vp);

        // Update the per-pass JFA uniforms (inv_vp + halving step + field size).
        for (k, buf) in self.jfa_step_buffers.iter().enumerate() {
            let step = (1u32 << (self.jfa_passes - 1 - k as u32)) as f32;
            let u = JfaUniforms {
                inv_vp,
                aux: [step, cw as f32, ch as f32, 0.0],
            };
            ctxt.write_buffer(buf, 0, bytemuck::bytes_of(&u));
        }
        let resolve_u = JfaUniforms {
            inv_vp,
            aux: [0.0, cw as f32, ch as f32, 0.0],
        };
        ctxt.write_buffer(&self.jfa_resolve_buffer, 0, bytemuck::bytes_of(&resolve_u));

        // Seed pass (reuses the field uniform buffer for the occluder list) → seed[0].
        self.fullscreen_pass(
            encoder,
            "gi2d_jfa_seed_pass",
            &self.seed[0].view,
            &self.jfa_seed_pipeline,
            &[&self.field_uniform_bind_group],
        );

        // Jump-flood passes, ping-ponging the seed textures.
        let mut src = 0usize;
        for k in 0..self.jfa_passes as usize {
            let dst = 1 - src;
            let seed_bg = self.tex_only_bind_group(&self.seed[src].view);
            self.fullscreen_pass(
                encoder,
                "gi2d_jfa_step_pass",
                &self.seed[dst].view,
                &self.jfa_step_pipeline,
                &[&seed_bg, &self.jfa_step_bind_groups[k]],
            );
            src = dst;
        }

        // Resolve the final seeds into the scalar distance field.
        let seed_bg = self.tex_only_bind_group(&self.seed[src].view);
        self.fullscreen_pass(
            encoder,
            "gi2d_jfa_resolve_pass",
            &self.sdf.view,
            &self.jfa_resolve_pipeline,
            &[&seed_bg, &self.jfa_resolve_bind_group],
        );
    }

    /// (Re)creates the ping-pong cascade textures at the given (decoupled) size.
    fn ensure_cascade_textures(&mut self, tw: u32, th: u32) {
        if self.cascade_tex_size != (tw, th) {
            self.cascade = [
                GiTexture::new(tw, th, HDR_FORMAT),
                GiTexture::new(tw, th, HDR_FORMAT),
            ];
            self.cascade_tex_size = (tw, th);
        }
    }

    /// Builds the radiance-cascade hierarchy top-down and composites cascade 0's
    /// irradiance over the scene into `output_view`.
    fn render_cascades(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        scene_view: &wgpu::TextureView,
        scene_sampler: &wgpu::Sampler,
    ) {
        let ctxt = Context::get();
        let (cw, ch) = self.gi_size;
        let e0 = self.base_block;
        let s0 = self.probe_spacing;

        // Decoupled layout: probe spacing `s` and direction-tile edge `e` are
        // independent, so the cascade texture is (W·e0/s0) × (H·e0/s0) — wider than
        // the field when there are more directions than the probe grid is fine.
        let tw = (cw * e0 / s0).max(1);
        let th = (ch * e0 / s0).max(1);
        self.ensure_cascade_textures(tw, th);

        // Bake the occluder distance field so the cascade march costs one fetch per
        // step instead of looping every occluder (big win for many-occluder scenes).
        if self.sdf_occluders {
            self.build_sdf(encoder);
        }

        let mindim = cw.min(ch);
        // Highest level whose probe spacing (s0 * 2^c) still fits the field.
        let max_n = (32 - (mindim / s0).max(1).leading_zeros()).max(1);
        let n = self.cascade_count.min(max_n).min(MAX_CASCADES as u32).max(1);

        // World units per field pixel (uniform 2D camera) → ray-interval scale.
        let wpp = (2.0 / cw as f32) / self.vp.x_axis.x.abs().max(1e-6);
        let base = wpp * s0 as f32;

        for c in 0..n {
            let e_c = e0 << c;
            let s_c = s0 << c;
            let e_up = e0 << (c + 1);
            let s_up = s0 << (c + 1);
            let up_px = (cw / s_up).max(1);
            let up_py = (ch / s_up).max(1);
            // Telescoping radial intervals: start_c = base·(4^c−1)/3, end_c = start_{c+1}.
            let p4 = 4.0f32.powi(c as i32);
            let start_c = base * (p4 - 1.0) / 3.0;
            let end_c = base * (p4 * 4.0 - 1.0) / 3.0;
            let params = CascadeParams {
                v0: [e_c as f32, s_c as f32, cw as f32, ch as f32],
                v1: [e_up as f32, s_up as f32, up_px as f32, up_py as f32],
                v2: [
                    start_c,
                    end_c,
                    self.max_steps as f32,
                    if c == n - 1 { 1.0 } else { 0.0 },
                ],
            };
            ctxt.write_buffer(
                &self.cascade_param_buffers[c as usize],
                0,
                bytemuck::bytes_of(&params),
            );
        }

        // Build top-down; level `l` writes cascade[l % 2] and reads cascade[(l+1) % 2].
        let sdf_bg = self.tex_bind_group(&self.sdf.view, "gi2d_cascade_sdf_bg");
        for level in (0..n).rev() {
            let dst = (level % 2) as usize;
            let upper = ((level + 1) % 2) as usize;
            let upper_bg = self.tex_only_bind_group(&self.cascade[upper].view);
            self.fullscreen_pass(
                encoder,
                "gi2d_cascade_pass",
                &self.cascade[dst].view,
                &self.cascade_pipeline,
                &[
                    &upper_bg,
                    &self.cascade_param_bind_groups[level as usize],
                    &self.field_uniform_bind_group,
                    &sdf_bg,
                ],
            );
        }

        // Cascade 0 ends up in cascade[0]; gather it and composite over the scene.
        let composite = CascadeCompositeUniforms {
            v0: [
                e0 as f32,
                (e0 * e0) as f32,
                (cw / s0).max(1) as f32,
                (ch / s0).max(1) as f32,
            ],
            v1: [cw as f32, ch as f32, s0 as f32, 0.0],
            ambient: [self.ambient.r, self.ambient.g, self.ambient.b, 0.0],
        };
        ctxt.write_buffer(
            &self.cascade_composite_buffer,
            0,
            bytemuck::bytes_of(&composite),
        );
        let scene_bg = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi2d_cascade_scene_bg"),
            layout: &self.tex_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(scene_sampler),
                },
            ],
        });
        let c0_bg = self.tex_only_bind_group(&self.cascade[0].view);
        self.fullscreen_pass(
            encoder,
            "gi2d_cascade_composite_pass",
            output_view,
            &self.cascade_composite_pipeline,
            &[&scene_bg, &c0_bg, &self.cascade_composite_bind_group],
        );
    }
}

impl PostProcessingEffect for Gi2d {
    fn update(&mut self, _dt: f32, _w: f32, _h: f32, _znear: f32, _zfar: f32) {}

    fn draw(&mut self, target: &RenderTarget, context: &mut PostProcessingContext) {
        let ctxt = Context::get();

        let (scene_view, scene_sampler, width, height) = match target {
            RenderTarget::Offscreen(o) => (&o.color_view, &o.sampler, o.width, o.height),
            RenderTarget::Screen => return,
        };

        self.ensure_textures(width, height);
        let prev = self.cur;
        let next = 1 - self.cur;

        // Field uniforms (scene transform + emitter/occluder lists), consumed by the
        // field/cascade march and the JFA seed pass.
        let field_uniforms = self.build_field_uniforms();
        ctxt.write_buffer(
            &self.field_uniform_buffer,
            0,
            bytemuck::bytes_of(&field_uniforms),
        );

        // Radiance-cascade solver: its own multi-pass path; skip the direct march.
        if self.radiance_cascades {
            self.render_cascades(context.encoder, context.output_view, scene_view, scene_sampler);
            self.frame_index = self.frame_index.wrapping_add(1);
            return;
        }

        // --- Optional: bake the occluder distance field via jump flood. ---
        if self.sdf_occluders {
            self.build_sdf(context.encoder);
        }

        // --- Pass: ray-march + temporal accumulate into the low-res field (next). ---
        let history_bind_group = self.tex_bind_group(&self.history[prev].view, "gi2d_history_bg");
        let sdf_bind_group = self.tex_bind_group(&self.sdf.view, "gi2d_sdf_bg");
        self.fullscreen_pass(
            context.encoder,
            "gi2d_field_pass",
            &self.history[next].view,
            &self.field_pipeline,
            &[
                &history_bind_group,
                &self.field_uniform_bind_group,
                &sdf_bind_group,
            ],
        );

        // --- Pass: composite scene × (ambient + upsampled irradiance) → output. ---
        let composite_uniforms = CompositeUniforms {
            ambient: [self.ambient.r, self.ambient.g, self.ambient.b, 0.0],
        };
        ctxt.write_buffer(
            &self.composite_uniform_buffer,
            0,
            bytemuck::bytes_of(&composite_uniforms),
        );
        let scene_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi2d_scene_bg"),
            layout: &self.tex_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(scene_sampler),
                },
            ],
        });
        let gi_bind_group = self.tex_bind_group(&self.history[next].view, "gi2d_field_bg");
        self.fullscreen_pass(
            context.encoder,
            "gi2d_composite_pass",
            context.output_view,
            &self.composite_pipeline,
            &[
                &scene_bind_group,
                &gi_bind_group,
                &self.composite_uniform_bind_group,
            ],
        );

        // Advance temporal state: the field we just wrote becomes next frame's history.
        self.cur = next;
        self.prev_vp = self.vp;
        self.history_valid = true;
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}
