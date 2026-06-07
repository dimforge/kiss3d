//! Real-time shadow mapping for the rasterization pipeline.
//!
//! This module renders scene depth from the point of view of each shadow-casting
//! light into a shared **shadow atlas** before the main color pass, then exposes
//! the atlas (plus the per-light light-space matrices) to [`ObjectMaterial`] so the
//! PBR shader can attenuate each light's contribution where it is occluded.
//!
//! [`ObjectMaterial`]: crate::builtin::ObjectMaterial
//!
//! # Texture / atlas layout
//!
//! All shadow maps are packed into a single `Depth32Float` **2D texture array**
//! (`resolution × resolution × MAX_SHADOW_VIEWS`). Using one array texture keeps
//! the material's bind group small — a single depth texture binding plus one
//! comparison sampler — regardless of how many lights are active, so we never run
//! into per-stage texture-binding limits even with the full `MAX_LIGHTS` budget.
//!
//! Each shadow-casting light is assigned a contiguous run of layers:
//!
//! | Light type    | Technique                          | Layers |
//! |---------------|------------------------------------|--------|
//! | Directional   | single tight orthographic cascade  | 1      |
//! | Spot          | perspective map fit to the cone    | 1      |
//! | Point         | cube map unrolled to 6 perspectives| 6      |
//!
//! Point lights select the relevant cube face in the shader from the dominant
//! axis of the light→fragment vector. The total number of views is capped at
//! [`MAX_SHADOW_VIEWS`]; lights that do not fit keep lighting without shadows.

use crate::camera::Camera3d;
use crate::context::Context;
use crate::light::{CollectedLight, LightCollection, LightType, MAX_LIGHTS};
use crate::scene::SceneNode3d;
use bytemuck::{Pod, Zeroable};
use glamx::{Mat4, Vec3};

/// Maximum number of shadow views (atlas layers) across all lights.
///
/// A directional or spot light uses one view; a point light uses six. The cap
/// bounds GPU memory and keeps the per-frame uniform fixed-size.
pub const MAX_SHADOW_VIEWS: usize = 16;

/// Maximum number of lights with shadow metadata in a frame. The primary tier
/// occupies slots `0..MAX_LIGHTS` (indexed by uniform slot); clustered shadow
/// casters occupy `MAX_LIGHTS..MAX_SHADOW_LIGHTS`. Real usage is still bounded by
/// [`MAX_SHADOW_VIEWS`] (each light needs at least one atlas view).
pub const MAX_SHADOW_LIGHTS: usize = MAX_LIGHTS + MAX_SHADOW_VIEWS;

/// Maximum number of cascades a directional light may use (cascaded shadow maps).
pub const MAX_CASCADES: u32 = 4;

/// Rasterizer depth bias for the shadow depth pass. A modest slope-scaled bias
/// combats acne on surfaces grazing the light; the fragment shader adds a small
/// constant compare bias plus receiver-plane depth bias. (Contacts stay attached
/// because the per-light near/far planes are fit to the casters — see
/// [`light_near_far`] — keeping the depth precision high.)
fn shadow_depth_bias() -> wgpu::DepthBiasState {
    wgpu::DepthBiasState {
        constant: 1,
        slope_scale: 1.75,
        clamp: 0.0,
    }
}

/// Tight perspective near/far planes for a point or spot light, fit to the
/// shadow-caster world AABB so the limited depth range isn't wasted on empty space
/// between the light's tiny default near plane and a far plane at its full
/// attenuation radius (which collapses all geometry to ~1.0 in NDC and destroys the
/// depth precision contact shadows need). Returns `(near, far)` with a little
/// headroom so casters at the bounds aren't clipped.
fn light_near_far(light_pos: Vec3, aabb_min: Vec3, aabb_max: Vec3) -> (f32, f32) {
    // Nearest point on the AABB to the light (the light may be inside it).
    let nearest = light_pos.clamp(aabb_min, aabb_max);
    let near_d = (nearest - light_pos).length();
    // Farthest AABB corner.
    let mut far_d: f32 = 0.0;
    for &cx in &[aabb_min.x, aabb_max.x] {
        for &cy in &[aabb_min.y, aabb_max.y] {
            for &cz in &[aabb_min.z, aabb_max.z] {
                far_d = far_d.max((Vec3::new(cx, cy, cz) - light_pos).length());
            }
        }
    }
    let near = (near_d * 0.9).max(0.02);
    let far = (far_d * 1.1).max(near + 0.1);
    (near, far)
}

/// Picks the point/spot shadow near/far planes: tight bounds from the caster AABB
/// when available (far never exceeds the light's attenuation `radius`, since
/// geometry beyond it is unlit), else the old wide default.
fn fit_near_far(caster_aabb: Option<(Vec3, Vec3)>, light_pos: Vec3, radius: f32) -> (f32, f32) {
    match caster_aabb {
        Some((min, max)) => {
            let (near, far) = light_near_far(light_pos, min, max);
            (near, far.min(radius))
        }
        None => (0.05, radius),
    }
}

/// Factor to scale a *radial* near plane down to the axial near at the CORNER of a
/// square perspective frustum of full angle `fov`. A point on the frustum's corner
/// ray sits at `1/sqrt(1 + 2·tan²(fov/2))` of its radial distance along the view
/// axis; without this, the radial (AABB-fit) near plane clips off-axis occluders
/// and drops part of their shadow. For a 90° cube face this is `1/√3 ≈ 0.577`.
fn near_corner_scale(fov: f32) -> f32 {
    let t = (fov * 0.5).tan();
    1.0 / (1.0 + 2.0 * t * t).sqrt()
}

/// Per-light shadow metadata consumed by `default.wgsl`.
///
/// `view_proj` matrices for every view are stored separately in a flat array;
/// this record points into it via `base_view`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuLightShadow {
    /// First layer/view index in the atlas for this light (`-1` as `u32::MAX` = none).
    base_view: u32,
    /// Number of views this light occupies (1 for directional/spot, 6 for point).
    num_views: u32,
    /// `0`=point, `1`=directional, `2`=spot — mirrors the lighting shader.
    light_type: u32,
    /// `1.0` if this light casts shadows this frame, `0.0` otherwise.
    enabled: f32,
    /// Light world position (used by point lights to pick the cube face).
    light_pos: [f32; 3],
    /// Far plane used to normalize point-light distances (unused otherwise).
    far_plane: f32,
}

impl Default for GpuLightShadow {
    fn default() -> Self {
        Self {
            base_view: u32::MAX,
            num_views: 0,
            light_type: 0,
            enabled: 0.0,
            light_pos: [0.0; 3],
            far_plane: 1.0,
        }
    }
}

/// Frame-level shadow uniforms (binding 2 of the shadow bind group).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ShadowUniforms {
    /// Light-space view-projection matrix for every atlas view.
    view_proj: [[[f32; 4]; 4]; MAX_SHADOW_VIEWS],
    /// Per-light shadow metadata. Slots `0..MAX_LIGHTS` are the primary tier
    /// (indexed by uniform slot, read via `compute_shadow(i)`); slots
    /// `MAX_LIGHTS..` are clustered shadow casters (referenced by each clustered
    /// light's `shadow_slot`).
    lights: [GpuLightShadow; MAX_SHADOW_LIGHTS],
    /// `1.0` when shadow mapping is globally enabled this frame, else `0.0`.
    shadows_enabled: f32,
    /// Texel size (`1.0 / resolution`) for PCF tap spacing.
    texel_size: f32,
    /// Depth bias applied when comparing, mitigating shadow acne.
    depth_bias: f32,
    /// `1.0` when at least one translucent caster wrote into the colored
    /// transmittance atlas this frame (so the lighting shader samples and tints
    /// by it); `0.0` keeps the cheap all-opaque path (atlas left untouched).
    transmittance_enabled: f32,
    /// PCF kernel scale = shadow-edge softness/blur. `1.0` is the default ~5x5
    /// penumbra; larger spreads the taps wider (softer), `0.0` collapses them to
    /// a hard edge. Followed by padding to a 16-byte block.
    softness: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    /// Directional cascade boundaries: the far view-space distance of each cascade
    /// (`[0..num_cascades)`), so the shader can pick the cascade by fragment depth
    /// and blend across boundaries. Same for every directional light (camera-based).
    cascade_splits: [f32; 4],
}

/// Per-view uniform written during the depth pre-pass (one light-space matrix).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ShadowViewUniforms {
    view_proj: [[f32; 4]; 4],
}

/// Per-object model uniform written during the depth pre-pass.
///
/// Mirrors the position transform used by `default.wgsl` so shadow geometry
/// matches the lit geometry exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ShadowModelUniforms {
    transform: [[f32; 4]; 4],
    scale: [[f32; 4]; 3], // mat3x3 padded to mat3x4 for alignment
    /// Base color (RGBA). Only read by the transmittance pass (to tint the
    /// shadow by translucent occluders); the depth pass ignores it.
    color: [f32; 4],
}

/// Aligned stride of the dynamic per-view/per-object buffers (256 satisfies the
/// minimum uniform-buffer-offset alignment on all wgpu backends).
const SHADOW_VIEW_STRIDE: u64 = 256;

/// A caster whose color alpha is below this counts as translucent: it is kept out
/// of the opaque depth map and instead tints the colored transmittance atlas.
const OPAQUE_ALPHA_THRESHOLD: f32 = 0.999;

/// Format of the colored transmittance atlas (RGB transmittance in `[0, 1]`).
const TRANSMITTANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A single shadow view scheduled for the depth pre-pass.
struct ShadowView {
    /// Atlas layer to render into.
    layer: u32,
    /// Light-space view-projection matrix.
    view_proj: Mat4,
}

/// Renders the scene depth from each shadow-casting light into a shared atlas
/// and provides the resources the lighting shader binds as group 4.
pub struct ShadowMapper {
    /// Whether shadow mapping is enabled globally.
    enabled: bool,
    /// Atlas resolution (per layer, square).
    resolution: u32,
    /// Slope-scaled-ish constant depth bias used by the PCF comparison.
    depth_bias: f32,
    /// Shadow-edge softness: scales the PCF tap spacing (`1.0` = default ~5x5
    /// penumbra, larger = blurrier, `0.0` = hard edges).
    softness: f32,
    /// Number of cascades a directional light splits its view frustum into
    /// (cascaded shadow maps). Clamped to `1..=MAX_CASCADES`.
    num_cascades: u32,
    /// Hard cap on how far (world units, along the view) directional shadows reach.
    /// The cascade range runs from the camera near plane to `min(far plane, this)`.
    /// `INFINITY` (default) = use the camera far plane.
    shadow_distance: f32,
    /// Far view distance of the FIRST (highest-resolution) directional cascade. It
    /// covers `[near, this]`; larger means the crisp near cascade reaches further (so
    /// the camera needn't be as close) at the cost of some near detail.
    first_cascade_far_bound: f32,
    /// The depth atlas texture (`Depth32Float`, 2D array).
    atlas: wgpu::Texture,
    /// One depth view per atlas layer, for the pre-pass render targets.
    layer_views: Vec<wgpu::TextureView>,
    /// Array view of the whole atlas, sampled by the lighting shader.
    array_view: wgpu::TextureView,
    /// Comparison sampler used for hardware PCF.
    compare_sampler: wgpu::Sampler,
    /// Colored transmittance atlas (`Rgba8Unorm`, 2D array): per shadow texel, the
    /// accumulated RGB transmittance of translucent occluders in front of the
    /// nearest opaque surface. White where nothing translucent occludes.
    transmittance_atlas: wgpu::Texture,
    /// One color view per transmittance-atlas layer, for the transmittance pass.
    transmittance_layer_views: Vec<wgpu::TextureView>,
    /// Array view of the transmittance atlas, sampled by the lighting shader.
    transmittance_array_view: wgpu::TextureView,
    /// Filtering sampler for the transmittance atlas (bilinear).
    transmittance_sampler: wgpu::Sampler,
    /// Pipeline that rasterizes translucent casters into the transmittance atlas.
    transmittance_pipeline: wgpu::RenderPipeline,
    /// Layout for the per-object albedo-texture bind group sampled by the
    /// transmittance pass (so colored shadows follow the occluder's texture).
    transmittance_tex_bgl: wgpu::BindGroupLayout,
    /// Frame-level shadow uniform buffer (matrices + metadata).
    uniform_buffer: wgpu::Buffer,
    /// Bind group layout for group 4 of the lighting pipeline.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group bound by `ObjectMaterial` (group 4).
    bind_group: wgpu::BindGroup,
    /// Depth-only pipeline used by the pre-pass.
    depth_pipeline: wgpu::RenderPipeline,
    /// Deformed depth pipeline (GPU skinning + morph). `None` on web (and any adapter
    /// without a free bind group), where deformable casters fall back to the rest
    /// shape.
    deform_depth_pipeline: Option<wgpu::RenderPipeline>,
    /// Deformed colored-transmittance pipeline (translucent deformable casters).
    /// `None` on web, where they fall back to the rest shape.
    deform_transmittance_pipeline: Option<wgpu::RenderPipeline>,
    /// Layout for the per-view bind group (group 0 of the depth pipeline).
    view_bind_group_layout: wgpu::BindGroupLayout,
    /// Dynamic per-view uniform buffer (one entry per scheduled view).
    view_uniform_buffer: wgpu::Buffer,
    /// Capacity (in views) of `view_uniform_buffer`.
    view_capacity: u64,
    /// Bind group over `view_uniform_buffer` (uses a dynamic offset).
    view_bind_group: wgpu::BindGroup,
    /// Layout for the per-object model bind group (group 1 of the depth pipeline).
    model_bind_group_layout: wgpu::BindGroupLayout,
    /// Dynamic per-object model uniform buffer (one entry per drawn object).
    model_uniform_buffer: wgpu::Buffer,
    /// Capacity (in objects) of `model_uniform_buffer`.
    model_capacity: u64,
    /// Bind group over `model_uniform_buffer` (uses a dynamic offset).
    model_bind_group: wgpu::BindGroup,
    /// Per-collected-light shadow-metadata slot from the last [`render`](Self::render):
    /// `shadow_slots[i]` is the `ShadowUniforms.lights` index for `lights.lights[i]`,
    /// or `u32::MAX` if it casts no shadow this frame. The clustered tier reads this
    /// to stamp each clustered light's `shadow_slot`.
    last_shadow_slots: Vec<u32>,
}

impl ShadowMapper {
    /// Creates a shadow mapper with the given per-layer resolution.
    pub fn new(resolution: u32) -> Self {
        let ctxt = Context::get();
        let resolution = resolution.max(1);

        let (atlas, layer_views, array_view) = Self::create_atlas(&ctxt, resolution);
        let (transmittance_atlas, transmittance_layer_views, transmittance_array_view) =
            Self::create_transmittance_atlas(&ctxt, resolution);

        // Comparison sampler: hardware does the depth test and (with linear
        // filtering) bilinear PCF across the 2x2 neighborhood per tap.
        let compare_sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_compare_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Plain bilinear sampler for the colored transmittance atlas.
        let transmittance_sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_transmittance_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_uniform_buffer"),
            size: std::mem::size_of::<ShadowUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = shadow_bind_group_layout(&ctxt);

        let bind_group = Self::create_bind_group(
            &ctxt,
            &bind_group_layout,
            &array_view,
            &compare_sampler,
            &uniform_buffer,
            &transmittance_array_view,
            &transmittance_sampler,
        );

        // === Depth-only pre-pass pipeline ===
        let view_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_view_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                            ShadowViewUniforms,
                        >()
                            as u64),
                    },
                    count: None,
                }],
            });

        let model_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_model_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // The transmittance pass reads the per-object color in its
                    // fragment stage, so the model uniform is visible to both.
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                            ShadowModelUniforms,
                        >()
                            as u64),
                    },
                    count: None,
                }],
            });

        // Albedo-texture bind group layout for the transmittance pass.
        let transmittance_tex_bgl =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_transmittance_tex_bgl"),
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

        let depth_pipeline =
            Self::create_depth_pipeline(&ctxt, &view_bind_group_layout, &model_bind_group_layout);
        let transmittance_pipeline = Self::create_transmittance_pipeline(
            &ctxt,
            &view_bind_group_layout,
            &model_bind_group_layout,
            &transmittance_tex_bgl,
        );

        // Deformed depth/transmittance pipelines (native only): the deform data needs
        // a 3rd bind group; we gate it on native so web stays uniformly in the rest
        // shape (the color pass has no deformed pipeline there either). The deform
        // bind-group layout is the shared one from `builtin::deform`, so each object's
        // deform bind group works in both the color and shadow passes.
        #[cfg(not(target_arch = "wasm32"))]
        let (deform_depth_pipeline, deform_transmittance_pipeline) = {
            let deform_layout = crate::builtin::deform::deform_bind_group_layout();
            let depth = Self::create_depth_pipeline_deform(
                &ctxt,
                &view_bind_group_layout,
                &model_bind_group_layout,
                &deform_layout,
            );
            let transmittance = Self::create_transmittance_pipeline_deform(
                &ctxt,
                &view_bind_group_layout,
                &model_bind_group_layout,
                &deform_layout,
                &transmittance_tex_bgl,
            );
            (Some(depth), Some(transmittance))
        };
        #[cfg(target_arch = "wasm32")]
        let (deform_depth_pipeline, deform_transmittance_pipeline): (
            Option<wgpu::RenderPipeline>,
            Option<wgpu::RenderPipeline>,
        ) = (None, None);

        let view_capacity = MAX_SHADOW_VIEWS as u64;
        let view_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_view_uniform_buffer"),
            size: SHADOW_VIEW_STRIDE * view_capacity,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let view_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_view_bind_group"),
            layout: &view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &view_uniform_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(
                        std::mem::size_of::<ShadowViewUniforms>() as u64
                    ),
                }),
            }],
        });

        let model_capacity = 64u64;
        let model_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_model_uniform_buffer"),
            size: SHADOW_VIEW_STRIDE * model_capacity,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let model_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_model_bind_group"),
            layout: &model_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &model_uniform_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(
                        std::mem::size_of::<ShadowModelUniforms>() as u64
                    ),
                }),
            }],
        });

        Self {
            enabled: true,
            resolution,
            depth_bias: 0.0012,
            softness: 1.0,
            num_cascades: 4,
            shadow_distance: f32::INFINITY,
            first_cascade_far_bound: 12.0,
            atlas,
            layer_views,
            array_view,
            compare_sampler,
            transmittance_atlas,
            transmittance_layer_views,
            transmittance_array_view,
            transmittance_sampler,
            transmittance_pipeline,
            transmittance_tex_bgl,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            depth_pipeline,
            deform_depth_pipeline,
            deform_transmittance_pipeline,
            view_bind_group_layout,
            view_uniform_buffer,
            view_capacity,
            view_bind_group,
            model_bind_group_layout,
            model_uniform_buffer,
            model_capacity,
            model_bind_group,
            last_shadow_slots: Vec::new(),
        }
    }

    /// Per-collected-light shadow-metadata slots from the last [`render`](Self::render).
    /// `slots[i]` is the `ShadowUniforms.lights` index for the scene's `i`-th collected
    /// light, or `u32::MAX` if it casts no shadow this frame. Consumed by the clustered
    /// lighting pass to stamp each clustered light's `shadow_slot`.
    pub(crate) fn shadow_slots(&self) -> &[u32] {
        &self.last_shadow_slots
    }

    /// Number of atlas views a light occupies (directional = cascades, spot = 1,
    /// point = 6).
    fn shadow_view_count(&self, light: &CollectedLight) -> usize {
        match light.light_type {
            LightType::Point { .. } => 6,
            LightType::Directional(_) => self.num_cascades as usize,
            LightType::Spot { .. } => 1,
        }
    }

    /// Computes the light-space matrices for a shadow-casting light, writes them
    /// into `view_proj` (and queues the views for the depth pre-pass), and returns
    /// the light's `GpuLightShadow` metadata. `base_view` is the light's first atlas
    /// layer; the caller must have checked it fits within [`MAX_SHADOW_VIEWS`].
    #[allow(clippy::too_many_arguments)]
    fn build_light_shadow(
        &self,
        light: &CollectedLight,
        camera: &dyn Camera3d,
        base_view: u32,
        splits: &[f32],
        caster_aabb: Option<(Vec3, Vec3)>,
        view_proj: &mut [[[f32; 4]; 4]; MAX_SHADOW_VIEWS],
        views: &mut Vec<ShadowView>,
    ) -> GpuLightShadow {
        let needed = self.shadow_view_count(light);
        let light_type;
        let mut far_plane = 1.0;

        match light.light_type {
            LightType::Directional(_) => {
                light_type = 1;
                let dir = light.world_direction.normalize_or(Vec3::NEG_Z);
                // Cascaded shadow maps: fit each cascade to its frustum slice
                // (the `splits` boundaries computed above), so near geometry gets
                // a dedicated high-resolution map and far geometry coarser ones.
                for c in 0..self.num_cascades {
                    let vp = calculate_cascade(
                        dir,
                        camera,
                        splits[c as usize],
                        splits[c as usize + 1],
                        self.resolution,
                    );
                    let layer = base_view + c;
                    view_proj[layer as usize] = vp.to_cols_array_2d();
                    views.push(ShadowView {
                        layer,
                        view_proj: vp,
                    });
                }
            }
            LightType::Spot {
                outer_cone_angle,
                attenuation_radius,
                ..
            } => {
                light_type = 2;
                let radius = attenuation_radius.max(1.0);
                let (near, far) = fit_near_far(caster_aabb, light.world_position, radius);
                far_plane = far;
                let dir = light.world_direction.normalize_or(Vec3::NEG_Z);
                let fov = (outer_cone_angle * 2.0).clamp(0.1, std::f32::consts::PI - 0.05);
                // Shrink the radial near plane to the (square) frustum's corner so
                // off-axis occluders aren't clipped (see the point-light case).
                let near = near * near_corner_scale(fov);
                let vp = perspective_view_proj(
                    light.world_position,
                    light.world_position + dir,
                    fov,
                    near,
                    far,
                );
                view_proj[base_view as usize] = vp.to_cols_array_2d();
                views.push(ShadowView {
                    layer: base_view,
                    view_proj: vp,
                });
            }
            LightType::Point { attenuation_radius } => {
                light_type = 0;
                let radius = attenuation_radius.max(1.0);
                let (near, far) = fit_near_far(caster_aabb, light.world_position, radius);
                far_plane = far;
                // Each cube face is a 90° frustum. The AABB-fit near plane is a
                // RADIAL distance, but an occluder near a face corner lies at only
                // ~1/√3 of that distance along the face axis, so the radial near
                // plane would clip the off-axis part of a caster (dropping half its
                // shadow). Shrink it to the corner's axial near.
                let near = near * near_corner_scale(std::f32::consts::FRAC_PI_2);
                // Six perspective views unrolling a cube map: +X,-X,+Y,-Y,+Z,-Z.
                let faces = cube_face_view_projs(light.world_position, near, far);
                for (face_idx, vp) in faces.iter().enumerate() {
                    let layer = base_view + face_idx as u32;
                    view_proj[layer as usize] = vp.to_cols_array_2d();
                    views.push(ShadowView {
                        layer,
                        view_proj: *vp,
                    });
                }
            }
        }

        GpuLightShadow {
            base_view,
            num_views: needed as u32,
            light_type,
            enabled: 1.0,
            light_pos: light.world_position.into(),
            far_plane,
        }
    }

    fn create_atlas(
        ctxt: &Context,
        resolution: u32,
    ) -> (wgpu::Texture, Vec<wgpu::TextureView>, wgpu::TextureView) {
        let atlas = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_atlas"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: MAX_SHADOW_VIEWS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let layer_views = (0..MAX_SHADOW_VIEWS as u32)
            .map(|layer| {
                atlas.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow_atlas_layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        let array_view = atlas.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow_atlas_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        (atlas, layer_views, array_view)
    }

    /// Creates the colored transmittance atlas (color counterpart of the depth
    /// atlas), returning the texture, one render view per layer, and the array
    /// view sampled by the lighting shader.
    fn create_transmittance_atlas(
        ctxt: &Context,
        resolution: u32,
    ) -> (wgpu::Texture, Vec<wgpu::TextureView>, wgpu::TextureView) {
        let atlas = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_transmittance_atlas"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: MAX_SHADOW_VIEWS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TRANSMITTANCE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let layer_views = (0..MAX_SHADOW_VIEWS as u32)
            .map(|layer| {
                atlas.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow_transmittance_layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        let array_view = atlas.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow_transmittance_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        (atlas, layer_views, array_view)
    }

    #[allow(clippy::too_many_arguments)]
    fn create_bind_group(
        ctxt: &Context,
        layout: &wgpu::BindGroupLayout,
        array_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        uniform_buffer: &wgpu::Buffer,
        transmittance_view: &wgpu::TextureView,
        transmittance_sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(array_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(transmittance_sampler),
                },
            ],
        })
    }

    fn create_depth_pipeline(
        ctxt: &Context,
        view_bind_group_layout: &wgpu::BindGroupLayout,
        model_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = ctxt.create_shader_module(
            Some("shadow_depth_shader"),
            include_str!("shadow_depth.wgsl"),
        );

        let layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_depth_pipeline_layout"),
            bind_group_layouts: &[Some(view_bind_group_layout), Some(model_bind_group_layout)],
            immediate_size: 0,
        });

        // Matches `ObjectMaterial`'s mesh + instance vertex layout, but the depth
        // pass only consumes the position stream (0) and instance translation (1)
        // and the instance deformation columns (2..4).
        let vertex_buffer_layouts = [
            // Buffer 0: vertex positions.
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            // Buffer 1: instance translations.
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            // Buffer 2: instance deformations (3 vec3 columns).
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 9]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 24,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            },
        ];

        ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_depth_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffer_layouts,
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Render every face (no culling). Front-face culling stores the
                // occluder's *far* surface, which detaches the shadow from an
                // object's contact (peter-panning — e.g. a bright ring around a
                // cone's base). Storing the nearest surface keeps shadows attached;
                // acne is handled by the modest slope-scaled bias here plus the
                // small constant depth bias the shader applies when comparing.
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: shadow_depth_bias(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }

    /// Builds the deformed depth pipeline: same as [`create_depth_pipeline`] but with
    /// a 3rd bind group (the shared deform group: joint palette + skin streams + morph
    /// deltas + control), deforming the mesh by skinning and/or morph targets so
    /// animated/morphed casters cast correctly-posed shadows. The vertex layout is
    /// identical to the non-deformed pass — deform data is read from the storage
    /// buffers by vertex index, not vertex attributes.
    #[cfg(not(target_arch = "wasm32"))]
    fn create_depth_pipeline_deform(
        ctxt: &Context,
        view_bind_group_layout: &wgpu::BindGroupLayout,
        model_bind_group_layout: &wgpu::BindGroupLayout,
        deform_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = ctxt.create_shader_module(
            Some("shadow_depth_deform_shader"),
            include_str!("shadow_depth_skinned.wgsl"),
        );

        let layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_depth_deform_pipeline_layout"),
            bind_group_layouts: &[
                Some(view_bind_group_layout),
                Some(model_bind_group_layout),
                Some(deform_bind_group_layout),
            ],
            immediate_size: 0,
        });

        // Same buffers 0..2 as the non-deformed pass (position + instance streams).
        let vertex_buffer_layouts = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 9]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 24,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            },
        ];

        ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_depth_deform_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffer_layouts,
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: shadow_depth_bias(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }

    /// Deformed variant of [`create_transmittance_pipeline`]: GPU skinning + morph
    /// (the shared deform group at group 2) plus the same transmittance fragment
    /// stage, so translucent deformable casters tint shadows in their animated/morphed
    /// pose. The vertex layout matches the non-deformed pass plus UVs; deform data is
    /// read from storage by vertex index.
    #[cfg(not(target_arch = "wasm32"))]
    fn create_transmittance_pipeline_deform(
        ctxt: &Context,
        view_bind_group_layout: &wgpu::BindGroupLayout,
        model_bind_group_layout: &wgpu::BindGroupLayout,
        deform_bind_group_layout: &wgpu::BindGroupLayout,
        tex_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = ctxt.create_shader_module(
            Some("shadow_transmittance_deform_shader"),
            include_str!("shadow_transmittance_skinned.wgsl"),
        );

        let layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_transmittance_deform_pipeline_layout"),
            bind_group_layouts: &[
                Some(view_bind_group_layout),
                Some(model_bind_group_layout),
                Some(deform_bind_group_layout),
                Some(tex_bind_group_layout),
            ],
            immediate_size: 0,
        });

        // Same buffers 0..2 as the non-deformed pass plus UVs (buffer 3, location 5)
        // for the albedo texture lookup.
        let vertex_buffer_layouts = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 9]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 24,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            },
            // Buffer 3: UVs (for the albedo texture lookup).
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x2,
                }],
            },
        ];

        let mult_blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Dst,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        };

        ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_transmittance_skinned_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffer_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TRANSMITTANCE_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: mult_blend,
                        alpha: mult_blend,
                    }),
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }

    /// Builds the pipeline that rasterizes translucent casters into the colored
    /// transmittance atlas: same geometry transform as the depth pass, but with a
    /// fragment stage writing `1 - a*(1-rgb)`, multiplicative blending, and a
    /// read-only depth test against the opaque depth atlas (so only occluders in
    /// front of the nearest opaque surface tint the light).
    fn create_transmittance_pipeline(
        ctxt: &Context,
        view_bind_group_layout: &wgpu::BindGroupLayout,
        model_bind_group_layout: &wgpu::BindGroupLayout,
        tex_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = ctxt.create_shader_module(
            Some("shadow_transmittance_shader"),
            include_str!("shadow_transmittance.wgsl"),
        );

        let layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_transmittance_pipeline_layout"),
            bind_group_layouts: &[
                Some(view_bind_group_layout),
                Some(model_bind_group_layout),
                Some(tex_bind_group_layout),
            ],
            immediate_size: 0,
        });

        // Same vertex streams as the depth pass: position (0), instance
        // translation (1), instance deformation columns (2..4), plus UVs (slot 3,
        // location 7) for the albedo texture lookup.
        let vertex_buffer_layouts = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 9]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 24,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            },
            // Buffer 3: UVs.
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x2,
                }],
            },
        ];

        // Multiplicative blend: result = src * dst. The atlas is cleared to white,
        // so overlapping translucent occluders compose order-independently.
        let mult_blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Dst,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        };

        ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_transmittance_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffer_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TRANSMITTANCE_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: mult_blend,
                        alpha: mult_blend,
                    }),
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
            // Read-only depth test against the opaque depth map: a translucent
            // fragment tints only where it is in front of the nearest opaque
            // surface. Depth writes are off so translucent occluders don't hide
            // one another (their transmittances multiply commutatively).
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }

    /// The bind group layout consumed by the lighting pipeline as group 4.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// The bind group bound as group 4 during the color pass.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Whether shadow mapping is enabled globally.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enables or disables shadow mapping globally.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// The current shadow-edge softness (PCF kernel scale; `1.0` is the default).
    pub fn softness(&self) -> f32 {
        self.softness
    }

    /// Sets the shadow-edge softness: the PCF tap-spacing scale. `1.0` is the
    /// default ~5x5 penumbra; larger values blur the shadow edges more, `0.0`
    /// gives hard edges. Clamped to `>= 0`.
    pub fn set_softness(&mut self, softness: f32) {
        self.softness = softness.max(0.0);
    }

    /// The current shadow atlas per-layer resolution.
    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    /// Number of cascades each directional light uses (cascaded shadow maps).
    pub fn num_cascades(&self) -> u32 {
        self.num_cascades
    }

    /// Sets the number of cascades for directional lights (clamped to
    /// `1..=MAX_CASCADES`). More cascades = higher near-field resolution over a
    /// larger range, at the cost of one extra depth pass + atlas layer each.
    pub fn set_num_cascades(&mut self, num_cascades: u32) {
        self.num_cascades = num_cascades.clamp(1, MAX_CASCADES);
    }

    /// The maximum directional-shadow distance cap (world units along the view).
    pub fn shadow_distance(&self) -> f32 {
        self.shadow_distance
    }

    /// Caps how far directional shadows reach (camera near plane to `min(far, this)`).
    /// `INFINITY` (default) uses the camera far plane.
    pub fn set_shadow_distance(&mut self, distance: f32) {
        self.shadow_distance = distance.max(0.0);
    }

    /// Far view distance of the highest-resolution directional cascade (it covers
    /// `[near, this]`). Increase it so crisp shadows reach further from the camera
    /// (less need to move in close); decrease it for more near detail. Default 12.
    pub fn set_first_cascade_far_bound(&mut self, bound: f32) {
        self.first_cascade_far_bound = bound.max(0.01);
    }

    /// Sets the shadow atlas per-layer resolution, reallocating the atlas.
    pub fn set_resolution(&mut self, resolution: u32) {
        let resolution = resolution.max(1);
        if resolution == self.resolution {
            return;
        }
        let ctxt = Context::get();
        self.resolution = resolution;
        let (atlas, layer_views, array_view) = Self::create_atlas(&ctxt, resolution);
        self.atlas = atlas;
        self.layer_views = layer_views;
        self.array_view = array_view;
        let (t_atlas, t_layer_views, t_array_view) =
            Self::create_transmittance_atlas(&ctxt, resolution);
        self.transmittance_atlas = t_atlas;
        self.transmittance_layer_views = t_layer_views;
        self.transmittance_array_view = t_array_view;
        self.bind_group = Self::create_bind_group(
            &ctxt,
            &self.bind_group_layout,
            &self.array_view,
            &self.compare_sampler,
            &self.uniform_buffer,
            &self.transmittance_array_view,
            &self.transmittance_sampler,
        );
    }

    /// Renders the shadow depth pre-pass and uploads the shadow uniforms.
    ///
    /// For every shadow-casting light this assigns a run of atlas layers, computes
    /// the light-space matrices, renders the scene depth into those layers with the
    /// depth-only pipeline, and writes `ShadowUniforms` for the lighting shader.
    ///
    /// When shadows are disabled or no light casts shadows this still writes a
    /// uniform with `shadows_enabled = 0`, so the lighting shader behaves exactly
    /// as if shadows were absent.
    pub(crate) fn render(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &dyn Camera3d,
        lights: &LightCollection,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &mut crate::renderer::timings::GpuTimer,
    ) {
        let ctxt = Context::get();

        let mut uniforms = ShadowUniforms {
            view_proj: [[[0.0; 4]; 4]; MAX_SHADOW_VIEWS],
            lights: [GpuLightShadow::default(); MAX_SHADOW_LIGHTS],
            shadows_enabled: 0.0,
            texel_size: 1.0 / self.resolution as f32,
            depth_bias: self.depth_bias,
            transmittance_enabled: 0.0,
            softness: self.softness,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            cascade_splits: [f32::MAX; 4],
        };

        if !self.enabled {
            ctxt.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
            return;
        }

        // Directional cascade boundaries (camera-based, shared by all directional
        // lights). Computed once; `cascade_splits[c]` is the far view distance of
        // cascade c so the shader can pick a cascade by fragment depth and blend.
        let splits = cascade_splits(
            camera,
            self.shadow_distance,
            self.first_cascade_far_bound,
            self.num_cascades,
        );
        let n = self.num_cascades.min(4) as usize;
        uniforms.cascade_splits[..n].copy_from_slice(&splits[1..=n]);

        let mut views: Vec<ShadowView> = Vec::new();
        let mut next_layer = 0u32;

        // World AABB of every shadow caster (skinned meshes included), used to fit
        // tight near/far planes for point and spot lights so the depth range isn't
        // wasted between a tiny near plane and a far plane at the full attenuation
        // radius — which is what makes contact shadows attach. Only point/spot
        // lights use it, so skip the (per-frame, all-caster) scan otherwise.
        let needs_aabb = lights.lights.iter().any(|l| {
            l.casts_shadows
                && matches!(
                    l.light_type,
                    LightType::Point { .. } | LightType::Spot { .. }
                )
        });
        let caster_aabb = if needs_aabb {
            scene.data().shadow_casters_world_aabb()
        } else {
            None
        };

        // Split lights into the primary tier (uniform array, shadows looked up by
        // slot via `compute_shadow(i)`) and the clustered tier. Shadow metadata for
        // the primary tier lives at slots `0..MAX_LIGHTS` (== uniform slot, so it
        // stays in lockstep with `object_material`'s `frame.lights`); clustered
        // shadow casters get slots `MAX_LIGHTS..MAX_SHADOW_LIGHTS` and each clustered
        // light references its slot through its `shadow_slot` field. Atlas views are
        // allocated greedily across both tiers, primary first.
        let (primary, clustered) = lights.split_primary_clustered();
        let mut shadow_slots = vec![u32::MAX; lights.lights.len()];

        for (slot, &li) in primary.iter().enumerate() {
            let light = &lights.lights[li];
            if !light.casts_shadows {
                continue;
            }
            let needed = self.shadow_view_count(light);
            if next_layer as usize + needed > MAX_SHADOW_VIEWS {
                // Out of atlas space: this light lights without shadows.
                continue;
            }
            uniforms.lights[slot] = self.build_light_shadow(
                light,
                camera,
                next_layer,
                &splits,
                caster_aabb,
                &mut uniforms.view_proj,
                &mut views,
            );
            shadow_slots[li] = slot as u32;
            next_layer += needed as u32;
        }

        let mut slot = MAX_LIGHTS;
        for &li in &clustered {
            if slot >= MAX_SHADOW_LIGHTS {
                // No metadata slots left for clustered shadow casters.
                break;
            }
            let light = &lights.lights[li];
            if !light.casts_shadows {
                continue;
            }
            let needed = self.shadow_view_count(light);
            if next_layer as usize + needed > MAX_SHADOW_VIEWS {
                // Out of atlas space; a smaller later light might still fit.
                continue;
            }
            uniforms.lights[slot] = self.build_light_shadow(
                light,
                camera,
                next_layer,
                &splits,
                caster_aabb,
                &mut uniforms.view_proj,
                &mut views,
            );
            shadow_slots[li] = slot as u32;
            next_layer += needed as u32;
            slot += 1;
        }

        self.last_shadow_slots = shadow_slots;

        if views.is_empty() {
            // Nothing casts shadows: behave as if shadows were off.
            ctxt.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
            return;
        }

        uniforms.shadows_enabled = 1.0;

        // Collect per-object world transforms once, in the same traversal order the
        // draw walk uses, so each object maps to a stable dynamic-buffer slot. We
        // must upload all model uniforms *before* opening any render pass, since
        // buffer writes can't be interleaved with an active pass.
        let mut models: Vec<ShadowModelUniforms> = Vec::new();
        let mut has_transparent = false;
        scene
            .data()
            .collect_shadow_models(&mut |transform, scale, color| {
                has_transparent |= color.a < OPAQUE_ALPHA_THRESHOLD;
                let scale_mat = glamx::Mat3::from_diagonal(scale).to_cols_array_2d();
                models.push(ShadowModelUniforms {
                    transform: transform.to_mat4().to_cols_array_2d(),
                    scale: [
                        [scale_mat[0][0], scale_mat[0][1], scale_mat[0][2], 0.0],
                        [scale_mat[1][0], scale_mat[1][1], scale_mat[1][2], 0.0],
                        [scale_mat[2][0], scale_mat[2][1], scale_mat[2][2], 0.0],
                    ],
                    color: [color.r, color.g, color.b, color.a],
                });
            });

        if models.is_empty() {
            // No surface geometry to occlude: nothing to render, behave as off.
            uniforms.shadows_enabled = 0.0;
            ctxt.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
            return;
        }

        // Upload per-view matrices into the dynamic buffer (one aligned slot each).
        self.ensure_view_capacity(views.len() as u64);
        for view in &views {
            let view_uniforms = ShadowViewUniforms {
                view_proj: view.view_proj.to_cols_array_2d(),
            };
            ctxt.write_buffer(
                &self.view_uniform_buffer,
                view.layer as u64 * SHADOW_VIEW_STRIDE,
                bytemuck::bytes_of(&view_uniforms),
            );
        }

        // Upload per-object model matrices into the dynamic buffer.
        self.ensure_model_capacity(models.len() as u64);
        for (idx, model) in models.iter().enumerate() {
            ctxt.write_buffer(
                &self.model_uniform_buffer,
                idx as u64 * SHADOW_VIEW_STRIDE,
                bytemuck::bytes_of(model),
            );
        }

        // Render scene depth into each scheduled atlas layer.
        for view in &views {
            let depth_view = &self.layer_views[view.layer as usize];
            let offset = (view.layer as u64 * SHADOW_VIEW_STRIDE) as u32;

            // Pass 1: opaque casters write depth (the binary-visibility map).
            {
                let depth_ts = gpu.render_scope("shadows");
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow_depth_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: depth_ts,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                pass.set_pipeline(&self.depth_pipeline);
                pass.set_bind_group(0, &self.view_bind_group, &[offset]);

                // Deformable casters (native only) use the deformed depth pipeline so
                // their shadow tracks the animated/morphed pose.
                let deform = self.deform_depth_pipeline.as_ref();

                // Re-traverse in collection order, binding each object's slot.
                let mut object_index = 0u32;
                scene.data_mut().render_shadow_casters(
                    &mut pass,
                    &self.depth_pipeline,
                    deform,
                    None, // depth pass: no albedo texture / UVs
                    &self.model_bind_group,
                    SHADOW_VIEW_STRIDE as u32,
                    &mut object_index,
                    false,
                    OPAQUE_ALPHA_THRESHOLD,
                );
            }

            // Pass 2: translucent casters accumulate colored transmittance, depth-
            // tested (read-only) against the opaque depth just written. Skipped
            // entirely when nothing translucent casts, leaving the all-opaque path
            // untouched.
            if has_transparent {
                let transmittance_view = &self.transmittance_layer_views[view.layer as usize];
                let transmittance_ts = gpu.render_scope("shadows");
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow_transmittance_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: transmittance_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // White = fully transmitting; occluders multiply into it.
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        // Read-only: keep the opaque depths, don't write.
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: transmittance_ts,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                pass.set_pipeline(&self.transmittance_pipeline);
                pass.set_bind_group(0, &self.view_bind_group, &[offset]);

                // Deformable translucent casters (native) use the deformed
                // transmittance pipeline so their tinted shadow tracks the pose.
                let deform = self.deform_transmittance_pipeline.as_ref();

                let mut object_index = 0u32;
                scene.data_mut().render_shadow_casters(
                    &mut pass,
                    &self.transmittance_pipeline,
                    deform,
                    Some(&self.transmittance_tex_bgl), // tints the shadow by the albedo texture
                    &self.model_bind_group,
                    SHADOW_VIEW_STRIDE as u32,
                    &mut object_index,
                    true,
                    OPAQUE_ALPHA_THRESHOLD,
                );
            }
        }

        uniforms.transmittance_enabled = if has_transparent { 1.0 } else { 0.0 };
        ctxt.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn ensure_model_capacity(&mut self, needed: u64) {
        if needed <= self.model_capacity {
            return;
        }
        let ctxt = Context::get();
        let new_capacity = needed.next_power_of_two();
        self.model_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_model_uniform_buffer"),
            size: SHADOW_VIEW_STRIDE * new_capacity,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.model_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_model_bind_group"),
            layout: &self.model_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.model_uniform_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(
                        std::mem::size_of::<ShadowModelUniforms>() as u64
                    ),
                }),
            }],
        });
        self.model_capacity = new_capacity;
    }

    fn ensure_view_capacity(&mut self, needed: u64) {
        if needed <= self.view_capacity {
            return;
        }
        let ctxt = Context::get();
        let new_capacity = needed.next_power_of_two();
        self.view_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_view_uniform_buffer"),
            size: SHADOW_VIEW_STRIDE * new_capacity,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.view_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_view_bind_group"),
            layout: &self.view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.view_uniform_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(
                        std::mem::size_of::<ShadowViewUniforms>() as u64
                    ),
                }),
            }],
        });
        self.view_capacity = new_capacity;
    }
}

/// The size in bytes of the shadow uniform buffer (`ShadowUniforms`).
///
/// Exposed so materials can allocate a matching neutral fallback buffer.
pub fn shadow_uniforms_size() -> u64 {
    std::mem::size_of::<ShadowUniforms>() as u64
}

/// Creates the bind group layout for the shadow resources (group 4 of the
/// lighting pipeline): depth atlas, comparison sampler, and shadow uniforms.
///
/// Both [`ShadowMapper`] and [`ObjectMaterial`] build this layout; wgpu treats
/// structurally identical layouts as compatible, so the mapper's bind group can
/// be bound against the material's pipeline.
///
/// [`ObjectMaterial`]: crate::builtin::ObjectMaterial
pub fn shadow_bind_group_layout(ctxt: &Context) -> wgpu::BindGroupLayout {
    ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shadow_bind_group_layout"),
        entries: &[
            // Depth atlas (2D array, sampled for comparison).
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            // Comparison sampler.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
            // Shadow uniforms.
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
            // Colored transmittance atlas (2D array, filtered).
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            // Filtering sampler for the transmittance atlas.
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// World-space camera position + the 4 frustum-corner ray directions + the forward
/// (centre) direction, derived from the camera. A unit viewport size maps the
/// unit-square corners to the NDC corners, so this needs no real framebuffer size.
fn frustum_rays(camera: &dyn Camera3d) -> (Vec3, [Vec3; 4], Vec3) {
    let unit = glamx::Vec2::ONE;
    let dirs = [
        camera.unproject(glamx::Vec2::new(0.0, 0.0), unit).1,
        camera.unproject(glamx::Vec2::new(1.0, 0.0), unit).1,
        camera.unproject(glamx::Vec2::new(0.0, 1.0), unit).1,
        camera.unproject(glamx::Vec2::new(1.0, 1.0), unit).1,
    ];
    let forward = camera.unproject(glamx::Vec2::splat(0.5), unit).1;
    (camera.eye(), dirs, forward)
}

/// Computes the cascaded-shadow-map split distances (forward depths) for a
/// directional light: `num_cascades + 1` boundaries, from the nearest to the
/// farthest visible shadow caster, distributed logarithmically (the standard PSSM
/// scheme — near cascades stay small/high-resolution, far ones grow).
///
/// The range is derived from the camera's clip planes, NOT from the scene bounds.
/// This is deliberate: a merged caster AABB can't give
/// a reliable nearest-geometry distance (its 8 corners are the extremes, so once a
/// body falls far into the void the relevant corner is a meaningless synthetic one),
/// and any scene-derived range couples the near cascades to far/fallen geometry —
/// causing the shadows to flicker and degrade. With a fixed range the logarithmic
/// split automatically places a tight cascade on whatever depth the objects are at,
/// so quality is governed purely by the camera, never by where things fall.
fn cascade_splits(
    camera: &dyn Camera3d,
    distance_cap: f32,
    first_bound: f32,
    num_cascades: u32,
) -> Vec<f32> {
    let (znear, zfar) = camera.clip_planes();
    let near_d = znear.max(1e-3);
    let far_d = zfar.min(distance_cap).max(near_d * 2.0);

    if num_cascades <= 1 {
        return vec![near_d, far_d];
    }
    // Cascade 0 spans `[near, first_bound]` (a generous, useful high-res range — a
    // pure log split from the tiny `znear` would make it almost nothing); cascades
    // `1..N` then grow geometrically from `first_bound` to `far` (a
    // `first_cascade_far_bound` plus log-split-of-the-remainder scheme).
    let first = first_bound.clamp(near_d * 2.0, far_d);
    let ratio = far_d / first;
    let n1 = (num_cascades - 1) as f32;
    let mut splits = Vec::with_capacity(num_cascades as usize + 1);
    splits.push(near_d);
    splits.push(first);
    for i in 1..num_cascades {
        splits.push(first * ratio.powf(i as f32 / n1));
    }
    splits
}

/// Builds the orthographic view-projection for one directional-light cascade,
/// fitting it to the camera frustum slice between forward depths `[near_depth,
/// far_depth]`.
///
/// The slice's bounding-sphere diameter is rounded up to an integer (`ceil`) so the
/// world-space texel size is a stable value invariant to camera orientation; the
/// projection is then texel-snapped. Together these keep the cascade temporally
/// stable (no shimmer). The depth range is a fixed function of the cascade size (not
/// of the scene bounds), so distant/fallen geometry can't corrupt it.
fn calculate_cascade(
    dir: Vec3,
    camera: &dyn Camera3d,
    near_depth: f32,
    far_depth: f32,
    resolution: u32,
) -> Mat4 {
    let dir = dir.normalize_or(Vec3::NEG_Z);
    let (eye, dirs, forward) = frustum_rays(camera);

    // 8 world-space corners of the frustum slice. `d` is a ray direction; dividing
    // the forward depth by `cos` gives the ray distance reaching that depth.
    let mut corners = [Vec3::ZERO; 8];
    for (i, d) in dirs.iter().enumerate() {
        let cos = d.dot(forward).max(1e-3);
        corners[i] = eye + *d * (near_depth / cos);
        corners[i + 4] = eye + *d * (far_depth / cos);
    }

    // Bounding-sphere diameter of the slice, rounded up to an integer for a stable
    // texel size (the corner set is rigid under camera motion, so the diameter is
    // orientation-invariant).
    let center = corners.iter().copied().fold(Vec3::ZERO, |a, c| a + c) / 8.0;
    let mut max_r = 0.05_f32;
    for c in &corners {
        max_r = max_r.max((*c - center).length());
    }
    let radius = (2.0 * max_r).ceil() * 0.5;

    let up = if dir.abs().dot(Vec3::Y) > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    // Push the light back by a fixed margin beyond the slice's bounding sphere so the
    // ortho depth range covers the whole slice (radius `radius` around `center`, i.e.
    // `[0, 2*radius]` from the light) plus a little headroom for casters sitting just
    // above it. The range is deliberately a FIXED function of the cascade size, NOT
    // of the caster bounds: deriving it from the casters let a far object (e.g. one
    // fallen into the void) corrupt the depth range and make shadows flicker/degrade.
    let margin = 0.5 * radius;
    let light_eye = center - dir * (radius + margin);
    let view = Mat4::look_at_rh(light_eye, center, up);
    let near = 0.0_f32;
    let far = 2.0 * radius + 2.0 * margin;
    let proj = Mat4::orthographic_rh(-radius, radius, -radius, radius, near, far);
    let view_proj = proj * view;

    // Texel snap: round the projected world origin to the texel grid so a fixed
    // world point always lands on the same texel (world-locked grid).
    let half_res = resolution.max(1) as f32 * 0.5;
    let origin = view_proj * glamx::Vec4::new(0.0, 0.0, 0.0, 1.0);
    let rounded_x = (origin.x * half_res).round() / half_res;
    let rounded_y = (origin.y * half_res).round() / half_res;
    let snap = Mat4::from_translation(Vec3::new(rounded_x - origin.x, rounded_y - origin.y, 0.0));
    snap * view_proj
}

/// Builds a perspective view-projection looking from `eye` toward `target`.
fn perspective_view_proj(eye: Vec3, target: Vec3, fov: f32, near: f32, far: f32) -> Mat4 {
    let fwd = (target - eye).normalize_or(Vec3::NEG_Z);
    let up = if fwd.abs().dot(Vec3::Y) > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(eye, eye + fwd, up);
    let proj = Mat4::perspective_rh(fov, 1.0, near, far);
    proj * view
}

/// Builds the six 90°-FOV perspective view-projections of a point-light cube map.
///
/// Face order: +X, -X, +Y, -Y, +Z, -Z, matching the selection logic in the shader.
fn cube_face_view_projs(eye: Vec3, near: f32, far: f32) -> [Mat4; 6] {
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, near, far);
    let dirs_ups = [
        (Vec3::X, Vec3::NEG_Y),
        (Vec3::NEG_X, Vec3::NEG_Y),
        (Vec3::Y, Vec3::Z),
        (Vec3::NEG_Y, Vec3::NEG_Z),
        (Vec3::Z, Vec3::NEG_Y),
        (Vec3::NEG_Z, Vec3::NEG_Y),
    ];
    let mut out = [Mat4::IDENTITY; 6];
    for (i, (dir, up)) in dirs_ups.iter().enumerate() {
        let view = Mat4::look_at_rh(eye, eye + *dir, *up);
        out[i] = proj * view;
    }
    out
}
