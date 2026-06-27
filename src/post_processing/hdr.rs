//! HDR finishing for the rasterization pipeline: floating-point film, ACES /
//! Reinhard tonemapping and Kawase dual-filter bloom.
//!
//! The rasterizer renders the scene into a linear `Rgba16Float` HDR texture (so
//! emissive values and bright highlights survive `> 1.0`). This module owns that
//! HDR target plus a bloom mip chain, and resolves everything into the final LDR
//! swapchain / offscreen texture in a single tonemap+composite pass.
//!
//! Pipeline order (see `window/rendering.rs`):
//!   1. the scene is rasterized into the (optionally multisampled) HDR target;
//!   2. MSAA is resolved into a single-sample HDR texture;
//!   3. bloom extracts/blurs bright pixels through the mip chain;
//!   4. the tonemap pass composites bloom, applies exposure + the tonemap
//!      operator + gamma, and writes LDR.
//!
//! Existing [`PostProcessingEffect`](crate::post_processing::PostProcessingEffect)s
//! run **after** tonemapping (on the resolved LDR image), so they are unaffected
//! by the HDR change.

use crate::context::Context;
use bytemuck::{Pod, Zeroable};

/// The floating-point format used for the HDR scene target and bloom chain.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Weighted-blended OIT accumulation target (premultiplied weighted color + weight).
pub const OIT_ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Weighted-blended OIT revealage target (product of `1 - alpha`).
pub const OIT_REVEAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// Number of mip levels in the bloom chain (each half the previous resolution).
const BLOOM_MIPS: u32 = 5;

/// Tonemapping operator applied during the HDR resolve pass.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Tonemap {
    /// No tonemapping; the HDR color is simply clamped to `[0, 1]` (then gamma
    /// encoded). The old, pre-HDR rasterizer look.
    None,
    /// ACES filmic tonemapping. Cinematic, but desaturates saturated colors and
    /// skews some hues — included mostly for comparison. Matches the path tracer.
    Aces,
    /// Reinhard tonemapping (`x / (1 + x)`).
    Reinhard,
    /// AgX neutral filmic tonemapping. Graceful highlight roll-off without ACES's
    /// hue skews; mild, even desaturation toward white at the top.
    AgX,
    /// Khronos PBR Neutral tonemapping (the default). Preserves the saturation of
    /// in-gamut colors and only desaturates true highlights — the least "washed
    /// out", which best matches kiss3d's display-referred colors.
    #[default]
    Neutral,
    /// Tony McMapface (Tomasz Stachowiak), sampled from its baked 3D LUT. A
    /// perceptual, hue-preserving display transform; the LUT is CC0.
    TonyMcMapface,
}

impl Tonemap {
    /// Operator code passed to the tonemap shaders (raster + path tracer).
    pub(crate) fn as_u32(self) -> u32 {
        match self {
            Tonemap::None => 0,
            Tonemap::Aces => 1,
            Tonemap::Reinhard => 2,
            Tonemap::AgX => 3,
            Tonemap::Neutral => 4,
            Tonemap::TonyMcMapface => 5,
        }
    }
}

/// Artistic color-grading controls applied in linear HDR space, just before the
/// tonemap operator. A neutral default (all `1.0`, no hue shift, white balance
/// `[1,1,1]`) leaves the image unchanged.
#[derive(Copy, Clone, Debug)]
pub struct ColorGrading {
    /// Per-channel linear white-balance gain (RGB). `[1, 1, 1]` is neutral.
    pub white_balance: [f32; 3],
    /// Saturation multiplier around luminance (`1.0` neutral, `0.0` grayscale).
    pub saturation: f32,
    /// Contrast multiplier around mid-gray (`1.0` neutral).
    pub contrast: f32,
    /// Gamma exponent applied in linear space (`1.0` neutral).
    pub gamma: f32,
    /// Hue rotation in radians about the RGB grayscale axis (`0.0` neutral).
    pub hue: f32,
}

impl Default for ColorGrading {
    fn default() -> Self {
        ColorGrading {
            white_balance: [1.0, 1.0, 1.0],
            saturation: 1.0,
            contrast: 1.0,
            gamma: 1.0,
            hue: 0.0,
        }
    }
}

/// User-facing HDR finishing settings (exposure, tonemap operator, bloom knobs).
#[derive(Copy, Clone, Debug)]
pub struct HdrSettings {
    /// Linear exposure multiplier applied before tonemapping. `1.0` is neutral.
    pub exposure: f32,
    /// Tonemapping operator.
    pub tonemap: Tonemap,
    /// Whether bloom is applied. Off by default.
    pub bloom_enabled: bool,
    /// Brightness threshold above which pixels contribute to bloom.
    pub bloom_threshold: f32,
    /// Soft-knee width around the threshold for a smooth bloom roll-off.
    pub bloom_knee: f32,
    /// Additive intensity of the bloom contribution.
    pub bloom_intensity: f32,
    /// Artistic color grading applied before the tonemap operator.
    pub color_grading: ColorGrading,
    /// When enabled, the exposure is computed automatically from the scene's
    /// average luminance (eye adaptation) instead of using [`exposure`](Self::exposure).
    pub auto_exposure: bool,
    /// Adaptation speed (per second) for auto-exposure. Higher adapts faster.
    pub auto_exposure_speed: f32,
    /// Smallest exposure multiplier auto-exposure may settle at (brightest scenes).
    pub auto_exposure_min: f32,
    /// Largest exposure multiplier auto-exposure may settle at (darkest scenes).
    pub auto_exposure_max: f32,
    /// Target middle-gray key value for auto-exposure (≈ 0.18).
    pub auto_exposure_key: f32,
}

impl Default for HdrSettings {
    fn default() -> Self {
        HdrSettings {
            exposure: 1.0,
            tonemap: Tonemap::default(),
            // Bloom is subtle/off by default so neutral settings match the old look.
            bloom_enabled: false,
            bloom_threshold: 1.0,
            bloom_knee: 0.5,
            bloom_intensity: 0.04,
            color_grading: ColorGrading::default(),
            auto_exposure: false,
            auto_exposure_speed: 3.0,
            auto_exposure_min: 0.05,
            auto_exposure_max: 8.0,
            auto_exposure_key: 0.18,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QuadVertex {
    position: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BloomUniforms {
    src_texel: [f32; 2],
    threshold: f32,
    knee: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TonemapUniforms {
    exposure: f32,
    operator: u32,
    bloom_intensity: f32,
    // 1.0 when the adapted exposure texture overrides `exposure`.
    auto_exposure: f32,
    // Color grading: white-balance gain (rgb) + force-opaque flag (w: 1.0 writes
    // alpha = 1.0 to the output, else the HDR scene alpha is forwarded).
    white_balance: [f32; 4],
    // (saturation, contrast, gamma, hue).
    grading: [f32; 4],
}

/// Uniforms for the auto-exposure adaptation pass (`auto_exposure_adapt.wgsl`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct AdaptUniforms {
    dt: f32,
    speed: f32,
    min_exposure: f32,
    max_exposure: f32,
    key: f32,
    _pad: [f32; 3],
}

/// A single mip level of the bloom chain.
struct BloomMip {
    // The texture is kept alive alongside its view.
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

/// The set of GPU textures (re)created together when the size or sample count
/// changes.
struct HdrTargets {
    scene_texture: wgpu::Texture,
    scene_view: wgpu::TextureView,
    scene_msaa_texture: Option<wgpu::Texture>,
    scene_msaa_view: Option<wgpu::TextureView>,
    bloom_mips: Vec<BloomMip>,
    // Weighted-blended OIT targets: premultiplied weighted color accumulator and
    // revealage. The transparent geometry pass renders into these; `composite_oit`
    // samples them (always the single-sample copies) and blends the result over the
    // opaque HDR scene. When MSAA is active, the geometry pass renders into the
    // multisampled `*_msaa` attachments and resolves into the single-sample ones.
    oit_accum_texture: wgpu::Texture,
    oit_accum_view: wgpu::TextureView,
    oit_reveal_texture: wgpu::Texture,
    oit_reveal_view: wgpu::TextureView,
    oit_accum_msaa_texture: Option<wgpu::Texture>,
    oit_accum_msaa_view: Option<wgpu::TextureView>,
    oit_reveal_msaa_texture: Option<wgpu::Texture>,
    oit_reveal_msaa_view: Option<wgpu::TextureView>,
}

/// Owns the HDR scene target, bloom chain and resolve pipelines for the
/// rasterizer. One instance lives on each [`Window`](crate::window::Window).
pub struct HdrPipeline {
    settings: HdrSettings,

    // Render-target size and sample count the GPU resources were built for.
    width: u32,
    height: u32,
    sample_count: u32,

    // HDR scene target. When multisampled, `scene_msaa` is the MSAA attachment
    // and `scene` is its single-sample resolve destination; otherwise only
    // `scene` exists and is rendered into directly.
    // Single-sample HDR scene texture, kept alive alongside its view.
    _scene_texture: wgpu::Texture,
    scene_view: wgpu::TextureView,
    // MSAA HDR attachment, kept alive alongside its view.
    _scene_msaa_texture: Option<wgpu::Texture>,
    scene_msaa_view: Option<wgpu::TextureView>,

    // Bloom mip chain (single-sample HDR), smallest first index is mip 0 = half res.
    bloom_mips: Vec<BloomMip>,

    // Weighted-blended OIT targets + composite pipeline. The `*_view`s are the
    // single-sample targets the composite samples; the `*_msaa_view`s (present only
    // under MSAA) are the multisampled render targets the geometry pass draws into
    // and resolves from.
    _oit_accum_texture: wgpu::Texture,
    oit_accum_view: wgpu::TextureView,
    _oit_reveal_texture: wgpu::Texture,
    oit_reveal_view: wgpu::TextureView,
    _oit_accum_msaa_texture: Option<wgpu::Texture>,
    oit_accum_msaa_view: Option<wgpu::TextureView>,
    _oit_reveal_msaa_texture: Option<wgpu::Texture>,
    oit_reveal_msaa_view: Option<wgpu::TextureView>,
    oit_layout: wgpu::BindGroupLayout,
    // Rebuilt whenever the sample count changes, since its `MultisampleState.count`
    // must match the (MSAA) HDR scene attachment it composites into.
    oit_composite_pipeline: wgpu::RenderPipeline,

    sampler: wgpu::Sampler,

    // Bloom pipelines (prefilter / downsample / upsample) and tonemap pipeline.
    bloom_layout: wgpu::BindGroupLayout,
    prefilter_pipeline: wgpu::RenderPipeline,
    downsample_pipeline: wgpu::RenderPipeline,
    upsample_pipeline: wgpu::RenderPipeline,
    tonemap_layout: wgpu::BindGroupLayout,
    tonemap_pipeline: wgpu::RenderPipeline,

    // Tony McMapface 48³ display-transform LUT (sampled by the tonemap pass).
    _tony_lut_texture: wgpu::Texture,
    tony_lut_view: wgpu::TextureView,
    tony_sampler: wgpu::Sampler,

    vertex_buffer: wgpu::Buffer,
    bloom_uniform: wgpu::Buffer,
    tonemap_uniform: wgpu::Buffer,

    // === Auto-exposure ===
    // 1x1 metered average luminance (R16Float).
    _meter_texture: wgpu::Texture,
    meter_view: wgpu::TextureView,
    // Ping-pong pair of 1x1 adapted-exposure textures (R16Float).
    _exposure_texs: [wgpu::Texture; 2],
    exposure_views: [wgpu::TextureView; 2],
    // Index of the texture written this frame (the other holds the previous value).
    exposure_index: usize,
    meter_layout: wgpu::BindGroupLayout,
    meter_pipeline: wgpu::RenderPipeline,
    adapt_layout: wgpu::BindGroupLayout,
    adapt_pipeline: wgpu::RenderPipeline,
    adapt_uniform: wgpu::Buffer,
    // Wall-clock of the previous adaptation, for the dt-based smoothing.
    last_adapt_time: Option<web_time::Instant>,
}

impl HdrPipeline {
    /// Creates the HDR pipeline for the given size, sample count and output
    /// (LDR) format.
    pub fn new(
        width: u32,
        height: u32,
        sample_count: u32,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let ctxt = Context::get();

        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hdr_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Bloom bind group: source texture + sampler + uniforms.
        let bloom_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hdr_bloom_layout"),
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

        let bloom_shader = ctxt.create_shader_module(
            Some("hdr_bloom_shader"),
            &crate::builtin::compile_shader_with_common(
                "package::hdr_bloom",
                include_str!("../builtin/hdr_bloom.wgsl"),
            ),
        );

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };

        let bloom_pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hdr_bloom_pipeline_layout"),
            bind_group_layouts: &[Some(&bloom_layout)],
            immediate_size: 0,
        });

        let make_bloom_pipeline = |label: &str, fs_entry: &str, blend: Option<wgpu::BlendState>| {
            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&bloom_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &bloom_shader,
                    entry_point: Some("vs_main"),
                    buffers: std::slice::from_ref(&vertex_layout),
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &bloom_shader,
                    entry_point: Some(fs_entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend,
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

        let prefilter_pipeline = make_bloom_pipeline("hdr_bloom_prefilter", "fs_prefilter", None);
        let downsample_pipeline =
            make_bloom_pipeline("hdr_bloom_downsample", "fs_downsample", None);
        // The upsample pass additively blends into the larger mip.
        let upsample_pipeline = make_bloom_pipeline(
            "hdr_bloom_upsample",
            "fs_upsample",
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::REPLACE,
            }),
        );

        // Tonemap bind group: scene texture + sampler, bloom texture + sampler, uniforms.
        let tonemap_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hdr_tonemap_layout"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Auto-exposure: the 1x1 adapted-exposure texture (binding 5).
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Tony McMapface 3D LUT + its sampler (shared `tonemap_ops.wgsl`
                // declares these at bindings 6 & 7).
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Tony McMapface display-transform LUT: a 48³ Rgba16Float 3D texture decoded
        // (offline) from the CC0 baked LUT. Sampled with the `x/(x+1)` encoding.
        let tony_lut = Self::create_tony_lut(&ctxt);
        let tony_lut_view = tony_lut.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        let tony_sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hdr_tony_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // `hdr_tonemap` imports the shared `apply_tonemap` from the `tonemap_ops`
        // WESL module (composed here instead of source concatenation).
        let tonemap_wgsl = crate::builtin::compile_wesl(
            &[
                ("package::tonemap_ops", crate::builtin::TONEMAP_OPS_WESL),
                (
                    "package::hdr_tonemap",
                    include_str!("../builtin/hdr_tonemap.wgsl"),
                ),
                ("package::common", crate::builtin::COMMON_WESL),
            ],
            "package::hdr_tonemap",
            &[],
        );
        let tonemap_shader = ctxt.create_shader_module(Some("hdr_tonemap_shader"), &tonemap_wgsl);

        let tonemap_pipeline_layout =
            ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hdr_tonemap_pipeline_layout"),
                bind_group_layouts: &[Some(&tonemap_layout)],
                immediate_size: 0,
            });

        let tonemap_pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hdr_tonemap_pipeline"),
            layout: Some(&tonemap_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &tonemap_shader,
                entry_point: Some("vs_main"),
                buffers: std::slice::from_ref(&vertex_layout),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &tonemap_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
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

        // OIT composite: reads the accum + revealage targets and blends the resolved
        // transparent color over the opaque HDR scene (SrcAlpha / OneMinusSrcAlpha,
        // with the fragment's alpha = 1 - revealage).
        let oit_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hdr_oit_layout"),
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
        let oit_composite_pipeline = Self::create_oit_composite_pipeline(&oit_layout, sample_count);

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
            Some("hdr_vertex_buffer"),
            bytemuck::cast_slice(&vertices),
            wgpu::BufferUsages::VERTEX,
        );

        let bloom_uniform = ctxt.create_buffer_simple(
            Some("hdr_bloom_uniform"),
            std::mem::size_of::<BloomUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let tonemap_uniform = ctxt.create_buffer_simple(
            Some("hdr_tonemap_uniform"),
            std::mem::size_of::<TonemapUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        // === Auto-exposure resources ===
        // 1x1 R16Float targets: one metered-luminance + a ping-pong exposure pair.
        let make_1x1 = |label: &str| {
            let tex = ctxt.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        let (meter_texture, meter_view) = make_1x1("hdr_autoexposure_meter");
        let (exposure_tex0, exposure_view0) = make_1x1("hdr_autoexposure_exp0");
        let (exposure_tex1, exposure_view1) = make_1x1("hdr_autoexposure_exp1");

        // Metering pipeline: scene texture + sampler -> 1x1 average luminance.
        let meter_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hdr_autoexposure_meter_layout"),
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
        let meter_shader = ctxt.create_shader_module(
            Some("hdr_autoexposure_meter"),
            &crate::builtin::compile_shader_with_common(
                "package::auto_exposure_meter",
                include_str!("../builtin/auto_exposure_meter.wgsl"),
            ),
        );
        let make_1x1_pipeline =
            |label: &str, layout: &wgpu::BindGroupLayout, shader: &wgpu::ShaderModule| {
                let pl = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(layout)],
                    immediate_size: 0,
                });
                ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pl),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::R16Float,
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
            };
        let meter_pipeline =
            make_1x1_pipeline("hdr_autoexposure_meter", &meter_layout, &meter_shader);

        // Adaptation pipeline: meter + prev exposure + sampler + uniforms -> new exposure.
        let adapt_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hdr_autoexposure_adapt_layout"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        let adapt_shader = ctxt.create_shader_module(
            Some("hdr_autoexposure_adapt"),
            &crate::builtin::compile_shader_with_common(
                "package::auto_exposure_adapt",
                include_str!("../builtin/auto_exposure_adapt.wgsl"),
            ),
        );
        let adapt_pipeline =
            make_1x1_pipeline("hdr_autoexposure_adapt", &adapt_layout, &adapt_shader);
        let adapt_uniform = ctxt.create_buffer_simple(
            Some("hdr_autoexposure_adapt_uniform"),
            std::mem::size_of::<AdaptUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let targets = Self::create_targets(width, height, sample_count);

        HdrPipeline {
            settings: HdrSettings::default(),
            width,
            height,
            sample_count,
            _scene_texture: targets.scene_texture,
            scene_view: targets.scene_view,
            _scene_msaa_texture: targets.scene_msaa_texture,
            scene_msaa_view: targets.scene_msaa_view,
            bloom_mips: targets.bloom_mips,
            _oit_accum_texture: targets.oit_accum_texture,
            oit_accum_view: targets.oit_accum_view,
            _oit_reveal_texture: targets.oit_reveal_texture,
            oit_reveal_view: targets.oit_reveal_view,
            _oit_accum_msaa_texture: targets.oit_accum_msaa_texture,
            oit_accum_msaa_view: targets.oit_accum_msaa_view,
            _oit_reveal_msaa_texture: targets.oit_reveal_msaa_texture,
            oit_reveal_msaa_view: targets.oit_reveal_msaa_view,
            oit_layout,
            oit_composite_pipeline,
            sampler,
            bloom_layout,
            prefilter_pipeline,
            downsample_pipeline,
            upsample_pipeline,
            tonemap_layout,
            tonemap_pipeline,
            _tony_lut_texture: tony_lut,
            tony_lut_view,
            tony_sampler,
            vertex_buffer,
            bloom_uniform,
            tonemap_uniform,
            _meter_texture: meter_texture,
            meter_view,
            _exposure_texs: [exposure_tex0, exposure_tex1],
            exposure_views: [exposure_view0, exposure_view1],
            exposure_index: 0,
            meter_layout,
            meter_pipeline,
            adapt_layout,
            adapt_pipeline,
            adapt_uniform,
            last_adapt_time: None,
        }
    }

    /// Uploads the embedded Tony McMapface LUT (48³ `Rgba16Float`, decoded offline
    /// from the CC0 baked LUT) as a 3D texture. Shared with the path tracer's
    /// tonemap pass so both sample the identical LUT.
    pub(crate) fn create_tony_lut(ctxt: &Context) -> wgpu::Texture {
        const DIM: u32 = 48;
        let data: &[u8] = include_bytes!("../builtin/tony_mc_mapface.bin");
        let tex = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr_tony_lut"),
            size: wgpu::Extent3d {
                width: DIM,
                height: DIM,
                depth_or_array_layers: DIM,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctxt.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(DIM * 8), // 4 channels * 2 bytes (f16)
                rows_per_image: Some(DIM),
            },
            wgpu::Extent3d {
                width: DIM,
                height: DIM,
                depth_or_array_layers: DIM,
            },
        );
        tex
    }

    /// Builds the OIT composite pipeline for a given MSAA sample count.
    ///
    /// The composite blends the resolved transparent color over the opaque HDR scene
    /// attachment, so its `MultisampleState.count` must match that attachment — hence
    /// it is rebuilt whenever the sample count changes.
    fn create_oit_composite_pipeline(
        oit_layout: &wgpu::BindGroupLayout,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let ctxt = Context::get();
        let oit_shader = ctxt.create_shader_module(
            Some("hdr_oit_shader"),
            include_str!("../builtin/hdr_oit.wgsl"),
        );
        let oit_pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hdr_oit_pipeline_layout"),
            bind_group_layouts: &[Some(oit_layout)],
            immediate_size: 0,
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };
        ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hdr_oit_composite_pipeline"),
            layout: Some(&oit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &oit_shader,
                entry_point: Some("vs_main"),
                buffers: std::slice::from_ref(&vertex_layout),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &oit_shader,
                entry_point: Some("fs_composite"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        // Keep the destination alpha (out.a = dst.a). The opaque
                        // scene's alpha is forwarded to the surface by the tonemap;
                        // overwriting it with the OIT coverage (1 - reveal) drove it
                        // to 0 on the background and made the canvas transparent →
                        // white page on browsers that composite canvas alpha (Firefox).
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
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
                count: sample_count.max(1),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }

    /// (Re)creates the HDR scene target and bloom chain textures.
    fn create_targets(width: u32, height: u32, sample_count: u32) -> HdrTargets {
        let ctxt = Context::get();
        let width = width.max(1);
        let height = height.max(1);
        let sample_count = sample_count.max(1);

        // Single-sample HDR scene texture (sampled by bloom + tonemap). When MSAA
        // is active this is the resolve destination.
        let scene_texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr_scene_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let scene_view = scene_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Multisampled HDR attachment, resolved into `scene_texture`.
        let (scene_msaa_texture, scene_msaa_view) = if sample_count > 1 {
            let msaa = ctxt.create_texture(&wgpu::TextureDescriptor {
                label: Some("hdr_scene_msaa_texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = msaa.create_view(&wgpu::TextureViewDescriptor::default());
            (Some(msaa), Some(view))
        } else {
            (None, None)
        };

        // Bloom mip chain: each level is half the previous resolution.
        let mut bloom_mips = Vec::with_capacity(BLOOM_MIPS as usize);
        let mut mw = width;
        let mut mh = height;
        for i in 0..BLOOM_MIPS {
            mw = (mw / 2).max(1);
            mh = (mh / 2).max(1);
            let tex = ctxt.create_texture(&wgpu::TextureDescriptor {
                label: Some("hdr_bloom_mip"),
                size: wgpu::Extent3d {
                    width: mw,
                    height: mh,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            bloom_mips.push(BloomMip {
                _texture: tex,
                view,
                width: mw,
                height: mh,
            });
            let _ = i;
        }

        // Weighted-blended OIT targets. The single-sample copies are always present
        // (the composite samples them); under MSAA they double as the geometry pass's
        // resolve destinations.
        let make_oit = |label: &str, format: wgpu::TextureFormat, samples: u32| {
            let tex = ctxt.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                // The MSAA attachments are only ever rendered into and resolved, never
                // sampled, so they don't need TEXTURE_BINDING.
                usage: if samples > 1 {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                } else {
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
                },
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        let (oit_accum_texture, oit_accum_view) = make_oit("hdr_oit_accum", OIT_ACCUM_FORMAT, 1);
        let (oit_reveal_texture, oit_reveal_view) =
            make_oit("hdr_oit_reveal", OIT_REVEAL_FORMAT, 1);
        let (
            oit_accum_msaa_texture,
            oit_accum_msaa_view,
            oit_reveal_msaa_texture,
            oit_reveal_msaa_view,
        ) = if sample_count > 1 {
            let (at, av) = make_oit("hdr_oit_accum_msaa", OIT_ACCUM_FORMAT, sample_count);
            let (rt, rv) = make_oit("hdr_oit_reveal_msaa", OIT_REVEAL_FORMAT, sample_count);
            (Some(at), Some(av), Some(rt), Some(rv))
        } else {
            (None, None, None, None)
        };

        HdrTargets {
            scene_texture,
            scene_view,
            scene_msaa_texture,
            scene_msaa_view,
            bloom_mips,
            oit_accum_texture,
            oit_accum_view,
            oit_reveal_texture,
            oit_reveal_view,
            oit_accum_msaa_texture,
            oit_accum_msaa_view,
            oit_reveal_msaa_texture,
            oit_reveal_msaa_view,
        }
    }

    /// Resizes the HDR resources if the size or sample count changed.
    pub fn resize(&mut self, width: u32, height: u32, sample_count: u32) {
        let width = width.max(1);
        let height = height.max(1);
        let sample_count = sample_count.max(1);
        if self.width == width && self.height == height && self.sample_count == sample_count {
            return;
        }
        // The composite pipeline's sample count is baked in, so rebuild it when the
        // sample count changes (cheap, and only happens on MSAA toggle / first frame).
        if self.sample_count != sample_count {
            self.oit_composite_pipeline =
                Self::create_oit_composite_pipeline(&self.oit_layout, sample_count);
        }
        let targets = Self::create_targets(width, height, sample_count);
        self._scene_texture = targets.scene_texture;
        self.scene_view = targets.scene_view;
        self._scene_msaa_texture = targets.scene_msaa_texture;
        self.scene_msaa_view = targets.scene_msaa_view;
        self.bloom_mips = targets.bloom_mips;
        self._oit_accum_texture = targets.oit_accum_texture;
        self.oit_accum_view = targets.oit_accum_view;
        self._oit_reveal_texture = targets.oit_reveal_texture;
        self.oit_reveal_view = targets.oit_reveal_view;
        self._oit_accum_msaa_texture = targets.oit_accum_msaa_texture;
        self.oit_accum_msaa_view = targets.oit_accum_msaa_view;
        self._oit_reveal_msaa_texture = targets.oit_reveal_msaa_texture;
        self.oit_reveal_msaa_view = targets.oit_reveal_msaa_view;
        self.width = width;
        self.height = height;
        self.sample_count = sample_count;
    }

    /// The view the scene must be rendered into (the MSAA attachment when MSAA
    /// is active, the single-sample HDR texture otherwise).
    pub fn scene_render_view(&self) -> &wgpu::TextureView {
        self.scene_msaa_view.as_ref().unwrap_or(&self.scene_view)
    }

    /// The single-sample resolved HDR scene texture (the resolve destination under
    /// MSAA, or the direct render target otherwise). Sampleable + renderable;
    /// used by SSR to read and additively composite reflections before tonemapping.
    pub fn scene_resolved_view(&self) -> &wgpu::TextureView {
        &self.scene_view
    }

    /// The MSAA resolve target (the single-sample HDR texture), or `None` when
    /// MSAA is disabled.
    pub fn scene_resolve_view(&self) -> Option<&wgpu::TextureView> {
        if self.scene_msaa_view.is_some() {
            Some(&self.scene_view)
        } else {
            None
        }
    }

    /// The OIT accumulation attachment the transparent geometry pass renders into
    /// (color attachment 0): the multisampled target under MSAA, the single-sample
    /// one otherwise. Clear to transparent black before rendering.
    pub fn oit_accum_view(&self) -> &wgpu::TextureView {
        self.oit_accum_msaa_view
            .as_ref()
            .unwrap_or(&self.oit_accum_view)
    }

    /// The OIT revealage attachment the transparent geometry pass renders into (color
    /// attachment 1): the multisampled target under MSAA, the single-sample one
    /// otherwise. Clear to white (1.0) before rendering.
    pub fn oit_reveal_view(&self) -> &wgpu::TextureView {
        self.oit_reveal_msaa_view
            .as_ref()
            .unwrap_or(&self.oit_reveal_view)
    }

    /// MSAA resolve target for the OIT accumulation attachment (the single-sample
    /// accum texture), or `None` when MSAA is disabled.
    pub fn oit_accum_resolve_view(&self) -> Option<&wgpu::TextureView> {
        self.oit_accum_msaa_view
            .as_ref()
            .map(|_| &self.oit_accum_view)
    }

    /// MSAA resolve target for the OIT revealage attachment (the single-sample reveal
    /// texture), or `None` when MSAA is disabled.
    pub fn oit_reveal_resolve_view(&self) -> Option<&wgpu::TextureView> {
        self.oit_reveal_msaa_view
            .as_ref()
            .map(|_| &self.oit_reveal_view)
    }

    /// Composites the transparent OIT result over the opaque HDR scene. Run after
    /// the transparent geometry pass and before [`resolve`](Self::resolve).
    pub(crate) fn composite_oit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &mut crate::renderer::timings::GpuTimer,
    ) {
        let ctxt = Context::get();
        // Sample the single-sample (resolved, under MSAA) OIT targets.
        let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hdr_oit_composite_bind_group"),
            layout: &self.oit_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.oit_accum_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.oit_reveal_view),
                },
            ],
        });
        let composite_ts = gpu.render_scope("composite");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hdr_oit_composite_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                // Blend over the opaque HDR scene attachment (the MSAA attachment when
                // multisampling is on — the composite pipeline matches its sample
                // count — else the single-sample HDR texture). The scene's MSAA
                // resolve happens afterwards in `window/rendering.rs`.
                view: self.scene_render_view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: composite_ts,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.oit_composite_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..4, 0..1);
    }

    /// Mutable access to the HDR finishing settings.
    pub fn settings_mut(&mut self) -> &mut HdrSettings {
        &mut self.settings
    }

    /// The current HDR finishing settings.
    pub fn settings(&self) -> &HdrSettings {
        &self.settings
    }

    fn bloom_bind_group(
        &self,
        ctxt: &Context,
        src: &wgpu::TextureView,
        src_w: u32,
        src_h: u32,
    ) -> wgpu::BindGroup {
        ctxt.write_buffer(
            &self.bloom_uniform,
            0,
            bytemuck::bytes_of(&BloomUniforms {
                src_texel: [1.0 / src_w.max(1) as f32, 1.0 / src_h.max(1) as f32],
                threshold: self.settings.bloom_threshold,
                knee: self.settings.bloom_knee,
            }),
        );
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hdr_bloom_bind_group"),
            layout: &self.bloom_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.bloom_uniform.as_entire_binding(),
                },
            ],
        })
    }

    /// Runs the bloom prefilter + downsample + upsample chain. The final blurred
    /// result lands in `bloom_mips[0]` (half resolution), which the tonemap pass
    /// samples.
    fn run_bloom(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &mut crate::renderer::timings::GpuTimer,
    ) {
        let ctxt = Context::get();

        // Prefilter the full-res scene into the first (half-res) mip.
        {
            let bg = self.bloom_bind_group(&ctxt, &self.scene_view, self.width, self.height);
            let bloom_ts = gpu.render_scope("bloom");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_bloom_prefilter_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_mips[0].view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: bloom_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.prefilter_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..4, 0..1);
        }

        // Downsample down the chain: mip[i] -> mip[i+1].
        for i in 0..self.bloom_mips.len() - 1 {
            let src = &self.bloom_mips[i];
            let dst = &self.bloom_mips[i + 1];
            let bg = self.bloom_bind_group(&ctxt, &src.view, src.width, src.height);
            let bloom_ts = gpu.render_scope("bloom");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_bloom_downsample_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: bloom_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.downsample_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..4, 0..1);
        }

        // Upsample back up the chain, additively blending: mip[i+1] -> mip[i].
        for i in (0..self.bloom_mips.len() - 1).rev() {
            let src = &self.bloom_mips[i + 1];
            let dst = &self.bloom_mips[i];
            let bg = self.bloom_bind_group(&ctxt, &src.view, src.width, src.height);
            let bloom_ts = gpu.render_scope("bloom");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_bloom_upsample_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst.view,
                    resolve_target: None,
                    // Load: the upsample pipeline additively blends onto existing content.
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: bloom_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.upsample_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..4, 0..1);
        }
    }

    /// Runs the full HDR resolve: optional bloom, then the tonemap+composite pass
    /// that writes the LDR result into `output_view`.
    ///
    /// The scene must already have been rendered into `scene_render_view` (and,
    /// if MSAA is active, resolved into the single-sample scene texture by the
    /// scene render pass's `resolve_target`).
    /// Meters the scene's average luminance and adapts the exposure toward it.
    /// Returns the index of the exposure texture holding this frame's value (which
    /// the tonemap pass samples). Ping-pongs the two 1x1 exposure textures.
    fn run_auto_exposure(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &mut crate::renderer::timings::GpuTimer,
    ) -> usize {
        let ctxt = Context::get();
        let write_index = self.exposure_index;
        let prev_index = 1 - write_index;

        // dt for the adaptation smoothing (clamped against long stalls / first frame).
        let now = web_time::Instant::now();
        let dt = match self.last_adapt_time {
            Some(t) => (now - t).as_secs_f32().clamp(0.0, 0.5),
            None => 0.0,
        };
        self.last_adapt_time = Some(now);

        ctxt.write_buffer(
            &self.adapt_uniform,
            0,
            bytemuck::bytes_of(&AdaptUniforms {
                dt,
                speed: self.settings.auto_exposure_speed,
                min_exposure: self.settings.auto_exposure_min,
                max_exposure: self.settings.auto_exposure_max,
                key: self.settings.auto_exposure_key,
                _pad: [0.0; 3],
            }),
        );

        // Metering pass: scene -> 1x1 average luminance.
        let meter_bg = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hdr_autoexposure_meter_bg"),
            layout: &self.meter_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        {
            let exposure_ts = gpu.render_scope("exposure");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_autoexposure_meter_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.meter_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: exposure_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.meter_pipeline);
            pass.set_bind_group(0, &meter_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Adaptation pass: (meter, prev exposure) -> new exposure.
        let adapt_bg = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hdr_autoexposure_adapt_bg"),
            layout: &self.adapt_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.meter_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.exposure_views[prev_index]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.adapt_uniform.as_entire_binding(),
                },
            ],
        });
        {
            let exposure_ts = gpu.render_scope("exposure");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_autoexposure_adapt_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.exposure_views[write_index],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: exposure_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.adapt_pipeline);
            pass.set_bind_group(0, &adapt_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // The texture just written is read this frame; ping-pong for the next.
        self.exposure_index = prev_index;
        write_index
    }

    /// `force_opaque` writes alpha = 1.0 to the output instead of forwarding the
    /// HDR scene's alpha. Set it for on-screen window surfaces (a browser
    /// composites the canvas against the page using its alpha, so a sub-1.0 alpha
    /// — e.g. left behind by a transparency pass — would make the canvas
    /// see-through). Leave it off for offscreen/snapshot/embedding targets that
    /// legitimately want the scene's alpha preserved.
    pub(crate) fn resolve(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        force_opaque: bool,
        gpu: &mut crate::renderer::timings::GpuTimer,
    ) {
        let ctxt = Context::get();

        let bloom_enabled = self.settings.bloom_enabled && self.settings.bloom_intensity > 0.0;
        if bloom_enabled {
            self.run_bloom(encoder, gpu);
        }

        // Auto-exposure: meter + adapt before tonemapping. The resulting 1x1
        // exposure texture is sampled by the tonemap pass (binding 5).
        let auto = self.settings.auto_exposure;
        let exposure_index = if auto {
            self.run_auto_exposure(encoder, gpu)
        } else {
            self.exposure_index
        };

        ctxt.write_buffer(
            &self.tonemap_uniform,
            0,
            bytemuck::bytes_of(&TonemapUniforms {
                exposure: self.settings.exposure,
                operator: self.settings.tonemap.as_u32(),
                bloom_intensity: if bloom_enabled {
                    self.settings.bloom_intensity
                } else {
                    0.0
                },
                auto_exposure: if auto { 1.0 } else { 0.0 },
                white_balance: {
                    let w = self.settings.color_grading.white_balance;
                    // .w carries the force-opaque flag (see `force_opaque` above).
                    [w[0], w[1], w[2], if force_opaque { 1.0 } else { 0.0 }]
                },
                grading: [
                    self.settings.color_grading.saturation,
                    self.settings.color_grading.contrast,
                    self.settings.color_grading.gamma,
                    self.settings.color_grading.hue,
                ],
            }),
        );

        // When bloom is disabled, sample the (black) first mip so the bind group
        // is always complete; its zero intensity makes the contribution vanish.
        let bloom_view = &self.bloom_mips[0].view;

        let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hdr_tonemap_bind_group"),
            layout: &self.tonemap_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.tonemap_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(
                        &self.exposure_views[exposure_index],
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.tony_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&self.tony_sampler),
                },
            ],
        });

        let tonemap_ts = gpu.render_scope("tonemap");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hdr_tonemap_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: tonemap_ts,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.tonemap_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..4, 0..1);
    }
}
