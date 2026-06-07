use crate::camera::Camera3d;
use crate::context::Context;
use crate::light::{LightCollection, LightType, MAX_LIGHTS};
use crate::post_processing::{OIT_ACCUM_FORMAT, OIT_REVEAL_FORMAT};
use crate::resource::vertex_index::VERTEX_INDEX_FORMAT;
use crate::resource::{
    multisample_state, DynamicUniformBuffer, GpuData, GpuMesh3d, Material3d, PipelineCache,
    RenderContext, Texture,
};
use crate::scene::{InstancesBuffer3d, ObjectData3d};
use bytemuck::{Pod, Zeroable};
use glamx::{Mat3, Pose3, Vec3};
use std::any::Any;
use std::cell::Cell;

/// GPU representation of a single light.
///
/// This 64-byte layout is shared by the fixed primary-light uniform array
/// ([`FrameUniforms::lights`]) and the clustered forward+ storage buffer
/// (`crate::builtin::clustered`), so both shading paths read identical packing.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct GpuLight {
    position: [f32; 3],
    light_type: u32, // 0=point, 1=directional, 2=spot
    direction: [f32; 3],
    intensity: f32,
    color: [f32; 3],
    inner_cone_cos: f32,
    outer_cone_cos: f32,
    attenuation_radius: f32,
    /// Index into `ShadowUniforms.lights` for this light's shadow metadata, or
    /// `u32::MAX` when it casts no shadow. Used by the clustered tier; the primary
    /// tier indexes shadows by its own uniform-array slot instead.
    shadow_slot: u32,
    _padding: f32,
}

impl Default for GpuLight {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            light_type: 0,
            direction: [0.0, 0.0, -1.0],
            intensity: 0.0,
            color: [1.0, 1.0, 1.0],
            inner_cone_cos: 1.0,
            outer_cone_cos: 0.0,
            attenuation_radius: 100.0,
            shadow_slot: u32::MAX,
            _padding: 0.0,
        }
    }
}

impl GpuLight {
    /// Packs a scene-collected light into its GPU representation. The light
    /// type is encoded as 0=point, 1=directional, 2=spot and the spot cone
    /// angles are pre-converted to cosines for the shader.
    pub(crate) fn from_collected(light: &crate::light::CollectedLight) -> GpuLight {
        let (light_type, inner_cone_cos, outer_cone_cos, attenuation_radius) =
            match &light.light_type {
                LightType::Point { attenuation_radius } => (0u32, 1.0, 0.0, *attenuation_radius),
                LightType::Directional(_) => (1u32, 1.0, 0.0, 0.0),
                LightType::Spot {
                    inner_cone_angle,
                    outer_cone_angle,
                    attenuation_radius,
                } => (
                    2u32,
                    inner_cone_angle.cos(),
                    outer_cone_angle.cos(),
                    *attenuation_radius,
                ),
            };

        GpuLight {
            position: light.world_position.into(),
            light_type,
            direction: light.world_direction.into(),
            intensity: light.intensity,
            color: light.color.into(),
            inner_cone_cos,
            outer_cone_cos,
            attenuation_radius,
            shadow_slot: u32::MAX,
            _padding: 0.0,
        }
    }
}

impl GpuLight {
    /// Sets the shadow-metadata slot (see [`GpuLight::shadow_slot`]).
    pub(crate) fn set_shadow_slot(&mut self, slot: u32) {
        self.shadow_slot = slot;
    }
}

/// Maximum number of reflection probes packed into the frame uniform. Must match
/// `MAX_PROBES` in `builtin/default.wgsl` and `renderer::reflection_probe`.
pub(crate) const MAX_PROBES: usize = crate::renderer::reflection_probe::MAX_PROBES;

/// GPU representation of a single reflection probe (64 bytes), packed into the
/// frame uniform's fixed-size probe array. Mirrors `Probe` in `default.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct GpuProbe {
    // xyz: world center; w: 1.0 if this slot is active, else 0.0.
    center_active: [f32; 4],
    // xyz: parallax-box min (world); w: array layer index.
    box_min_layer: [f32; 4],
    // xyz: parallax-box max (world); w: intensity.
    box_max_intensity: [f32; 4],
    // x: rotation (radians); y: falloff (world units); z: max LOD; w: unused.
    params: [f32; 4],
}

/// Frame-level uniforms (view, projection, lights).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FrameUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    lights: [GpuLight; MAX_LIGHTS],
    num_lights: u32,
    ambient_intensity: f32,
    _padding: [f32; 2],
    // Global ambient light color (rgb); a is unused.
    ambient_color: [f32; 4],
    // Distance fog color (rgb) + max fog opacity (a).
    fog_color: [f32; 4],
    // Fog params: (mode, param_a, param_b, height_falloff). See `Fog::params`.
    fog_params: [f32; 4],
    // Camera world position (xyz) + unused, for image-based lighting.
    camera_pos: [f32; 4],
    // IBL params: (has_ibl, max_lod, intensity, env_rotation_radians).
    ibl_params: [f32; 4],
    // Clustered forward+ grid: (grid_x, grid_y, grid_z, num_clustered_lights).
    cluster_grid_dims: [f32; 4],
    // Clustered depth slicing: (z_near, z_far, ln(z_far/z_near), unused).
    cluster_depth: [f32; 4],
    // Clustered tile size in pixels: (tile_w, tile_h, unused, unused).
    cluster_tile: [f32; 4],
    // Reflection probes: x = active probe count; yzw unused (keeps the following
    // array 16-byte aligned).
    probe_count: [u32; 4],
    // World-space clip plane (a,b,c,d). When xyz != 0, fragments with
    // dot(xyz, world_pos) + w < 0 are discarded (reflector capture clips geometry
    // behind the mirror). All-zero = inactive.
    clip_plane: [f32; 4],
    // Fixed-size reflection-probe array (only the first `probe_count.x` are live).
    probes: [GpuProbe; MAX_PROBES],
}

/// Object-level uniforms (transform, scale, color, PBR properties).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ObjectUniforms {
    transform: [[f32; 4]; 4],
    ntransform: [[f32; 4]; 3], // mat3x3 padded to mat3x4 for alignment
    scale: [[f32; 4]; 3],      // mat3x3 padded to mat3x4 for alignment
    color: [f32; 4],
    metallic: f32,
    roughness: f32,
    _pad0: [f32; 2],
    emissive: [f32; 4],
    // Texture presence flags (0.0 or 1.0 - WGSL doesn't support bool in uniforms)
    has_normal_map: f32,
    has_metallic_roughness_map: f32,
    has_ao_map: f32,
    has_emissive_map: f32,
    // Extended PBR surface properties (clearcoat, anisotropy, transmission, ...).
    reflectance: f32,
    clearcoat: f32,
    clearcoat_roughness: f32,
    anisotropy: f32,
    anisotropy_rotation: f32,
    transmission: f32,
    // Alpha mode code (0 opaque / 1 mask / 2 blend / 3 premultiplied) + cutoff.
    alpha_mode: f32,
    alpha_cutoff: f32,
    specular_tint: [f32; 4],
    // (has_height_map, parallax_scale, unused, unused).
    parallax: [f32; 4],
    // Per-object SSR: (intensity, infinite_thick, distance_attenuation, fresnel);
    // intensity 0 means the object receives no SSR.
    ssr: [f32; 4],
    // Per-object planar reflector: world -> reflection-texture clip transform.
    reflector_view_proj: [[f32; 4]; 4],
    // (reflection_intensity, has_reflector, normal_falloff, unused).
    reflection_params: [f32; 4],
    // Reflector world-space plane normal (xyz); w unused. Used for the
    // normal-alignment falloff.
    reflector_normal: [f32; 4],
}

/// View uniforms for wireframe rendering (includes viewport).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WireframeViewUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    viewport: [f32; 4], // x, y, width, height
}

/// Model uniforms for wireframe rendering.
/// Layout must match WGSL struct:
/// - transform: mat4x4<f32> at offset 0 (64 bytes)
/// - scale: vec3<f32> at offset 64 (12 bytes, aligned to 16)
/// - num_edges: u32 at offset 76 (4 bytes)
/// - default_color: vec4<f32> at offset 80 (16 bytes, aligned to 16)
/// - default_width: f32 at offset 96 (4 bytes)
/// - use_perspective: u32 at offset 100 (4 bytes)
/// - _padding: vec2<f32> at offset 104 (8 bytes)
///
/// Total: 112 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WireframeModelUniforms {
    transform: [[f32; 4]; 4], // 64 bytes at offset 0
    scale: [f32; 3],          // 12 bytes at offset 64
    num_edges: u32,           // 4 bytes at offset 76
    default_color: [f32; 4],  // 16 bytes at offset 80
    default_width: f32,       // 4 bytes at offset 96
    use_perspective: u32,     // 4 bytes at offset 100
    _padding: [f32; 2],       // 8 bytes at offset 104 to align to 16-byte boundary
}

/// Edge data in GPU format (matches shader struct).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuEdge {
    point_a: [f32; 3],
    _pad_a: f32,
    point_b: [f32; 3],
    _pad_b: f32,
}

/// Model uniforms for point rendering.
/// Layout must match wireframe_points.wgsl ModelUniforms struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PointsModelUniforms {
    transform: [[f32; 4]; 4], // 64 bytes at offset 0
    scale: [f32; 3],          // 12 bytes at offset 64
    num_vertices: u32,        // 4 bytes at offset 76
    default_color: [f32; 4],  // 16 bytes at offset 80
    default_size: f32,        // 4 bytes at offset 96
    use_perspective: u32,     // 4 bytes at offset 100
    _padding: [f32; 2],       // 8 bytes at offset 104 to align to 16-byte boundary
}

/// Vertex data in GPU format for points (matches shader struct).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuVertex {
    position: [f32; 3],
    _pad: f32,
}

/// Per-object GPU data for ObjectMaterial.
///
/// This struct now only contains wireframe and points rendering data.
/// The main uniform buffers and shared view uniforms are managed by ObjectMaterial.
pub struct ObjectMaterialGpuData {
    // Cached combined material-texture bind group (albedo + PBR maps). The cached
    // pointers below detect when any of the source textures change so it is rebuilt.
    texture_bind_group: Option<wgpu::BindGroup>,
    cached_texture_ptr: usize,
    /// Offset into the dynamic object uniform buffer, set during prepare() phase.
    object_uniform_offset: Option<u32>,
    cached_normal_map_ptr: usize,
    cached_metallic_roughness_map_ptr: usize,
    cached_ao_map_ptr: usize,
    cached_emissive_map_ptr: usize,
    cached_height_map_ptr: usize,
    /// Reflection texture view bound last (the reflector target, or fallback during
    /// capture / when not a reflector). Detects when the bind group must rebuild.
    cached_reflection_ptr: usize,
    /// Reflector generation bound last. The reflector's `color_view` lives in a fixed
    /// struct slot (stable address), so a resize that replaces the underlying texture
    /// isn't caught by `cached_reflection_ptr` alone — the generation catches it.
    cached_reflection_gen: u64,
    // Wireframe rendering data (model uniforms are per-object)
    wireframe_model_uniform_buffer: wgpu::Buffer,
    wireframe_edge_buffer: wgpu::Buffer,
    wireframe_edge_capacity: usize,
    wireframe_model_bind_group: Option<wgpu::BindGroup>,
    /// Cached wireframe edges in local coordinates (built lazily from mesh).
    wireframe_edges: Option<Vec<(Vec3, Vec3)>>,
    /// Hash of mesh faces to detect when edges need rebuilding.
    wireframe_edges_mesh_hash: u64,
    /// Cached wireframe model uniforms (written during prepare).
    wireframe_model_uniforms: WireframeModelUniforms,
    // Point rendering data (model uniforms are per-object)
    points_model_uniform_buffer: wgpu::Buffer,
    points_vertex_buffer: wgpu::Buffer,
    points_vertex_capacity: usize,
    points_model_bind_group: Option<wgpu::BindGroup>,
    /// Cached vertices for point rendering (built lazily from mesh).
    points_vertices: Option<Vec<Vec3>>,
    /// Hash of mesh coords to detect when vertices need rebuilding.
    points_vertices_mesh_hash: u64,
    /// Cached points model uniforms (written during prepare).
    points_model_uniforms: PointsModelUniforms,
}

impl ObjectMaterialGpuData {
    /// Creates new per-object GPU data.
    pub fn new() -> Self {
        let ctxt = Context::get();

        // Wireframe model uniform buffer (per-object)
        let wireframe_model_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wireframe_model_uniform_buffer"),
            size: std::mem::size_of::<WireframeModelUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initial edge storage buffer (will grow as needed)
        let wireframe_edge_capacity = 1024;
        let wireframe_edge_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wireframe_edge_buffer"),
            size: (std::mem::size_of::<GpuEdge>() * wireframe_edge_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Point model uniform buffer (per-object)
        let points_model_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("points_model_uniform_buffer"),
            size: std::mem::size_of::<PointsModelUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initial vertex storage buffer for points (will grow as needed)
        let points_vertex_capacity = 1024;
        let points_vertex_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("points_vertex_buffer"),
            size: (std::mem::size_of::<GpuVertex>() * points_vertex_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            texture_bind_group: None,
            cached_texture_ptr: 0,
            object_uniform_offset: None,
            // Material-texture caching (albedo + PBR maps)
            cached_normal_map_ptr: 0,
            cached_metallic_roughness_map_ptr: 0,
            cached_ao_map_ptr: 0,
            cached_emissive_map_ptr: 0,
            cached_height_map_ptr: 0,
            cached_reflection_ptr: 0,
            cached_reflection_gen: 0,
            // Wireframe rendering
            wireframe_model_uniform_buffer,
            wireframe_edge_buffer,
            wireframe_edge_capacity,
            wireframe_model_bind_group: None,
            wireframe_edges: None,
            wireframe_edges_mesh_hash: 0,
            wireframe_model_uniforms: WireframeModelUniforms {
                transform: [[0.0; 4]; 4],
                scale: [0.0; 3],
                num_edges: 0,
                default_color: [0.0; 4],
                default_width: 0.0,
                use_perspective: 0,
                _padding: [0.0; 2],
            },
            points_model_uniform_buffer,
            points_vertex_buffer,
            points_vertex_capacity,
            points_model_bind_group: None,
            points_vertices: None,
            points_vertices_mesh_hash: 0,
            points_model_uniforms: PointsModelUniforms {
                transform: [[0.0; 4]; 4],
                scale: [0.0; 3],
                num_vertices: 0,
                default_color: [0.0; 4],
                default_size: 0.0,
                use_perspective: 0,
                _padding: [0.0; 2],
            },
        }
    }

    /// Ensures the edge buffer has enough capacity, growing if needed.
    fn ensure_edge_buffer_capacity(&mut self, needed: usize) {
        if needed > self.wireframe_edge_capacity {
            let ctxt = Context::get();
            let new_capacity = needed.next_power_of_two();
            self.wireframe_edge_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wireframe_edge_buffer"),
                size: (std::mem::size_of::<GpuEdge>() * new_capacity) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.wireframe_edge_capacity = new_capacity;
            // Invalidate bind group since buffer changed
            self.wireframe_model_bind_group = None;
        }
    }

    /// Ensures the vertex buffer for points has enough capacity, growing if needed.
    fn ensure_vertex_buffer_capacity(&mut self, needed: usize) {
        if needed > self.points_vertex_capacity {
            let ctxt = Context::get();
            let new_capacity = needed.next_power_of_two();
            self.points_vertex_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
                label: Some("points_vertex_buffer"),
                size: (std::mem::size_of::<GpuVertex>() * new_capacity) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.points_vertex_capacity = new_capacity;
            // Invalidate bind group since buffer changed
            self.points_model_bind_group = None;
        }
    }
}

impl Default for ObjectMaterialGpuData {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuData for ObjectMaterialGpuData {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// The default material used to draw objects.
///
/// This struct holds shared resources (pipeline, bind group layouts, dynamic buffers)
/// that are used by all objects. Per-object resources for wireframe/points are stored
/// in `ObjectMaterialGpuData` instances.
///
/// ## Performance Optimization
///
/// This material uses dynamic uniform buffers to batch uniform data writes:
/// - Frame uniforms (view, projection, light) are written once per frame
/// - Object uniforms are accumulated in a dynamic buffer and flushed once
/// - Wireframe/points view uniforms (view, proj, viewport) are shared and written once per frame
/// - This significantly reduces the number of `write_buffer` calls per frame
pub struct ObjectMaterial {
    /// Pipeline with backface culling enabled (lazily built per MSAA sample count)
    pipeline_cull: PipelineCache,
    /// Pipeline with backface culling disabled (lazily built per MSAA sample count)
    pipeline_no_cull: PipelineCache,
    /// Weighted-blended OIT pipeline (backface culling), writing the accum +
    /// revealage targets with no depth write. Used in the transparent phase.
    oit_pipeline_cull: PipelineCache,
    /// Weighted-blended OIT pipeline (no culling).
    oit_pipeline_no_cull: PipelineCache,
    /// Depth + view-position prepass pipeline (single target), for SSAO.
    prepass_pipeline: PipelineCache,
    object_bind_group_layout: wgpu::BindGroupLayout,
    /// Combined material-texture bind group layout (albedo + PBR maps, group 2).
    texture_bind_group_layout: wgpu::BindGroupLayout,
    /// Default PBR textures for when user hasn't set any
    default_normal_map: std::sync::Arc<crate::resource::Texture>,
    default_metallic_roughness_map: std::sync::Arc<crate::resource::Texture>,
    default_ao_map: std::sync::Arc<crate::resource::Texture>,
    default_emissive_map: std::sync::Arc<crate::resource::Texture>,
    default_height_map: std::sync::Arc<crate::resource::Texture>,
    /// Clamp+linear sampler for the per-object planar-reflection texture (binding 13).
    reflection_sampler: wgpu::Sampler,
    // Wireframe rendering resources
    wireframe_pipeline: PipelineCache,
    wireframe_model_bind_group_layout: wgpu::BindGroupLayout,
    // Point rendering resources
    points_pipeline: PipelineCache,
    points_model_bind_group_layout: wgpu::BindGroupLayout,

    // === Dynamic uniform buffer system ===
    /// Shared frame uniform buffer (view, projection, light)
    frame_uniform_buffer: wgpu::Buffer,
    /// Shared bind group for frame uniforms (+ the IBL environment at bindings 1/2)
    frame_bind_group: wgpu::BindGroup,
    /// Frame bind group layout, kept so the group can be rebuilt when the IBL
    /// environment or SSAO texture changes.
    frame_bind_group_layout: wgpu::BindGroupLayout,
    /// Whether this material uses the clustered forward+ pipeline variant (group 0
    /// has storage bindings 4/5/6 and the fragment shader has the clustered loop).
    clustered: bool,
    /// Currently-bound clustered storage buffers (group 0 bindings 4/5/6). Start as
    /// tiny placeholders; swapped for the renderer's real buffers by
    /// [`set_clustered_buffers`](Self::set_clustered_buffers).
    clustered_lights_buf: wgpu::Buffer,
    cluster_grid_buf: wgpu::Buffer,
    cluster_index_buf: wgpu::Buffer,
    /// Whether the real clustered buffers have been bound yet (false = placeholders).
    clustered_bound: bool,
    // === Per-view textures in group 0: IBL env (1/2) + SSAO (3). ===
    /// 1x1 black fallback env bound when no IBL environment is set.
    _ibl_fallback_texture: wgpu::Texture,
    ibl_fallback_view: wgpu::TextureView,
    ibl_fallback_sampler: wgpu::Sampler,
    /// 1x1 white fallback AO (no occlusion) bound when SSAO is off.
    _ao_fallback_texture: wgpu::Texture,
    ao_fallback_view: wgpu::TextureView,
    /// Currently-bound views (clones; default to the fallbacks).
    cur_ibl_view: wgpu::TextureView,
    cur_ibl_sampler: wgpu::Sampler,
    cur_ao_view: wgpu::TextureView,
    /// Identities of the bound views (`0` = fallback) to avoid per-frame rebuilds.
    ibl_bound_ptr: usize,
    ao_bound_ptr: usize,
    /// Current IBL parameters, written into the frame uniform each frame.
    ibl_has: Cell<bool>,
    ibl_max_lod: Cell<f32>,
    ibl_intensity: Cell<f32>,
    ibl_rotation: Cell<f32>,
    /// Whether SSAO is active this frame (gates the AO sample in the shader).
    ssao_has: Cell<bool>,
    /// Active while a reflection probe is being captured. The capture's per-face
    /// views have no clustered cull data (and the clustered buffers may still be
    /// placeholders), so it forces the fixed-light path; it also disables
    /// reflection probes so captured surfaces don't sample the probe being
    /// captured (which would create a hall-of-mirrors feedback loop).
    capture_mode: Cell<bool>,
    // === Reflection probes (group 0 binding 7; data in the frame uniform). ===
    /// 1x1x1 black fallback probe array, bound when no probes are set.
    _probe_fallback_texture: wgpu::Texture,
    probe_fallback_view: wgpu::TextureView,
    /// Currently-bound probe array view (clone; defaults to the fallback).
    cur_probe_view: wgpu::TextureView,
    /// Identity of the bound probe view (`0` = fallback) to avoid rebuilds.
    probe_bound_ptr: usize,
    /// Packed probe records + count + max LOD, written into the frame uniform.
    probe_records: Cell<[GpuProbe; MAX_PROBES]>,
    probe_count: Cell<u32>,
    /// World-space clip plane (a,b,c,d), set during reflector capture; all-zero off.
    clip_plane: Cell<[f32; 4]>,
    /// Dynamic buffer for object uniforms
    object_uniform_buffer: DynamicUniformBuffer<ObjectUniforms>,
    /// Bind group for object uniforms (recreated when buffer grows)
    object_bind_group: Option<wgpu::BindGroup>,
    /// Capacity when bind group was last created (to detect regrowth)
    object_bind_group_capacity: u64,
    /// Frame counter for detecting new frames
    frame_counter: Cell<u64>,
    /// Last frame we processed (to detect new frame)
    last_frame: Cell<u64>,

    // === Shared wireframe/points view uniforms ===
    /// Shared wireframe view uniform buffer (view, proj, viewport - same for all objects)
    wireframe_view_uniform_buffer: wgpu::Buffer,
    /// Shared wireframe view bind group
    wireframe_view_bind_group: wgpu::BindGroup,
    /// Shared points view uniform buffer (view, proj, viewport - same for all objects)
    points_view_uniform_buffer: wgpu::Buffer,
    /// Shared points view bind group
    points_view_bind_group: wgpu::BindGroup,

    // === Shadow mapping (group 4) ===
    /// Neutral fallback bind group used when no shadow mapper is supplied (or when
    /// shadows are disabled). It points at a 1x1 dummy atlas with a uniform whose
    /// `shadows_enabled` flag is `0`, so the shader skips all shadow sampling.
    default_shadow_bind_group: wgpu::BindGroup,
    /// Keeps the dummy atlas/sampler/buffer alive for `default_shadow_bind_group`.
    _default_shadow_resources: DefaultShadowResources,

    // === GPU vertex deformation: skinning + morph targets (native only) ===
    /// Deformed pipeline variants. `None` on web (and any adapter without a 5th bind
    /// group), where skinned/morphed meshes fall back to the plain pipelines (base
    /// rest shape). The per-object deform bind group (group 4) is owned by the object
    /// (see [`crate::builtin::deform`]); these are only the pipelines.
    deform: Option<DeformResources>,
}

/// Deformed pipeline variants. Mirror the plain opaque/OIT/prepass pipelines but
/// bind the deform group (group 4: joint palette + skin joints/weights + morph
/// deltas + control uniform) and use the deformed vertex entry. The vertex *layout*
/// is identical to the plain one — all deform data is read from storage by vertex
/// index, not vertex attributes.
struct DeformResources {
    pipeline_cull: PipelineCache,
    pipeline_no_cull: PipelineCache,
    oit_pipeline_cull: PipelineCache,
    oit_pipeline_no_cull: PipelineCache,
    prepass_pipeline: PipelineCache,
}

/// Owns the GPU resources backing [`ObjectMaterial`]'s neutral shadow bind group.
struct DefaultShadowResources {
    _atlas: wgpu::Texture,
    _view: wgpu::TextureView,
    _sampler: wgpu::Sampler,
    _uniform: wgpu::Buffer,
    _transmittance_atlas: wgpu::Texture,
    _transmittance_view: wgpu::TextureView,
    _transmittance_sampler: wgpu::Sampler,
}

// Non-skinned vertex input: injected into `default.wgsl` at `//__VERTEX_INPUT__`.
// Must reproduce the original struct exactly so the non-skinned shader is
// byte-identical to before this split.
const PLAIN_VERTEX_INPUT: &str = "struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coord: vec2<f32>,
    @location(2) normal: vec3<f32>,
}";

// Deformed vertex input: identical vertex attributes to the plain variant (skin
// joints/weights and morph deltas are read from the group-4 storage buffers by
// vertex index, not as vertex attributes), plus the deform bind group at group 4.
const DEFORM_VERTEX_INPUT: &str = "struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coord: vec2<f32>,
    @location(2) normal: vec3<f32>,
}
struct DeformControl {
    num_targets: u32,
    num_vertices: u32,
    has_skin: u32,
    has_morph_normals: u32,
    weights: array<vec4<f32>, 16>,
}
@group(4) @binding(0) var<storage, read> joint_palette: array<mat4x4<f32>>;
@group(4) @binding(1) var<storage, read> skin_joints: array<vec4<u32>>;
@group(4) @binding(2) var<storage, read> skin_weights: array<vec4<f32>>;
@group(4) @binding(3) var<storage, read> morph_pos: array<vec4<f32>>;
@group(4) @binding(4) var<storage, read> morph_nrm: array<vec4<f32>>;
@group(4) @binding(5) var<uniform> deform: DeformControl;";

// Non-skinned vertex entry, injected at `//__VS_MAIN__`. Byte-identical to the
// original `vs_main`.
const PLAIN_VS_MAIN: &str = "@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    // Build deformation matrix from instance data
    let deformation = mat3x3<f32>(
        instance.inst_def_0,
        instance.inst_def_1,
        instance.inst_def_2
    );

    // Transform position
    let scaled_pos = object.scale * vertex.position;
    let deformed_pos = deformation * scaled_pos;
    let model_pos = object.transform * vec4<f32>(deformed_pos, 1.0);
    let world_pos = vec4<f32>(instance.inst_tra, 0.0) + model_pos;

    out.clip_position = frame.proj * frame.view * world_pos;
    out.world_pos = world_pos.xyz;

    // Transform normal to world space
    out.world_normal = normalize(deformation * object.ntransform * vertex.normal);

    // View-space position for lighting calculations
    let view_pos = frame.view * world_pos;
    out.view_pos = view_pos.xyz / view_pos.w;

    out.tex_coord = vertex.tex_coord;
    out.vert_color = instance.inst_color;

    return out;
}";

// Deformed vertex entry. First applies morph targets (Σ weightᵢ·Δ, read from the
// group-4 storage buffers by vertex index), then — when the mesh is skinned — the
// joint-palette blend (ignoring the mesh node's own transform, per the glTF spec),
// or otherwise the ordinary instance/object-transform path. One control uniform
// gates both so skin-only, morph-only, and skin+morph meshes share this entry.
const DEFORM_VS_MAIN: &str = "@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput, @builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;

    // Morph: accumulate weighted position (and optional normal) deltas.
    var pos = vertex.position;
    var nrm = vertex.normal;
    if (deform.num_targets > 0u) {
        for (var t = 0u; t < deform.num_targets; t = t + 1u) {
            let wgt = deform.weights[t >> 2u][t & 3u];
            if (wgt != 0.0) {
                let idx = t * deform.num_vertices + vid;
                pos = pos + wgt * morph_pos[idx].xyz;
                if (deform.has_morph_normals != 0u) {
                    nrm = nrm + wgt * morph_nrm[idx].xyz;
                }
            }
        }
    }

    if (deform.has_skin != 0u) {
        // Skin: blend the joint matrices by their (renormalized) weights.
        var w = skin_weights[vid];
        let j = skin_joints[vid];
        let wsum = w.x + w.y + w.z + w.w;
        if (wsum > 0.0) { w = w / wsum; }
        let skin =
            w.x * joint_palette[j.x] +
            w.y * joint_palette[j.y] +
            w.z * joint_palette[j.z] +
            w.w * joint_palette[j.w];

        let world_pos = skin * vec4<f32>(pos, 1.0);
        out.clip_position = frame.proj * frame.view * world_pos;
        out.world_pos = world_pos.xyz;
        let skin3 = mat3x3<f32>(skin[0].xyz, skin[1].xyz, skin[2].xyz);
        out.world_normal = normalize(skin3 * nrm);
        let view_pos = frame.view * world_pos;
        out.view_pos = view_pos.xyz / view_pos.w;
    } else {
        // Morph-only (or rigid): the usual instance/object-transform path.
        let deformation = mat3x3<f32>(
            instance.inst_def_0,
            instance.inst_def_1,
            instance.inst_def_2
        );
        let scaled_pos = object.scale * pos;
        let deformed_pos = deformation * scaled_pos;
        let model_pos = object.transform * vec4<f32>(deformed_pos, 1.0);
        let world_pos = vec4<f32>(instance.inst_tra, 0.0) + model_pos;

        out.clip_position = frame.proj * frame.view * world_pos;
        out.world_pos = world_pos.xyz;
        out.world_normal = normalize(deformation * object.ntransform * nrm);
        let view_pos = frame.view * world_pos;
        out.view_pos = view_pos.xyz / view_pos.w;
    }

    out.tex_coord = vertex.tex_coord;
    out.vert_color = instance.inst_color;

    return out;
}";

/// Clustered forward+ storage bindings, injected into group 0 only for the
/// clustered pipeline variant. Omitted on the fixed-light fallback so the shader
/// declares no storage buffers (WebGL2-safe).
const CLUSTERED_BINDINGS: &str =
    "@group(0) @binding(4) var<storage, read> clustered_lights: array<LightData>;
@group(0) @binding(5) var<storage, read> cluster_light_grid: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> cluster_light_index: array<u32>;";

/// The clustered-light shading loop, injected after the primary-light loop in
/// `shade()`. Locates the fragment's cluster, then accumulates the (shadowless)
/// contribution of every light the cull pass recorded for it.
const CLUSTERED_LOOP: &str = "if frame.cluster_grid_dims.w > 0.5 {
        let near = frame.cluster_depth.x;
        let log_ratio = frame.cluster_depth.z;
        let gz = frame.cluster_grid_dims.z;
        let zv = max(-in.view_pos.z, near);
        var fslice = floor(log(zv / near) / log_ratio * gz);
        fslice = clamp(fslice, 0.0, gz - 1.0);
        let slice = u32(fslice);
        let gx = u32(frame.cluster_grid_dims.x);
        let gy = u32(frame.cluster_grid_dims.y);
        let tx = min(u32(in.clip_position.x / frame.cluster_tile.x), gx - 1u);
        let ty = min(u32(in.clip_position.y / frame.cluster_tile.y), gy - 1u);
        let cluster = tx + ty * gx + slice * gx * gy;
        let cell = cluster_light_grid[cluster];
        for (var k = 0u; k < cell.y; k = k + 1u) {
            let li = cluster_light_index[cell.x + k];
            let cl = clustered_lights[li];
            var contrib = shade_light(
                cl, in.view_pos, V, N_view, F0, albedo, metallic,
                alpha, aniso, at, ab, aniso_t, aniso_b, cc_alpha
            );
            // A clustered light with an allocated shadow slot is shadow-mapped too.
            if cl.shadow_slot != 0xffffffffu {
                contrib *= compute_shadow(cl.shadow_slot, in.world_pos, dpos_dx, dpos_dy, receives_transmit);
            }
            Lo += contrib;
        }
    }";

/// Assembles the object shader source, injecting either the plain or the deformed
/// (skinning + morph) vertex input + vertex entry into `default.wgsl`, and the
/// clustered forward+ bindings + loop when `clustered` is set. The whole fragment
/// stage (apart from the injected clustered loop) is shared between all variants.
fn build_object_shader_src(deformed: bool, clustered: bool) -> String {
    let (vertex_input, vs_main) = if deformed {
        (DEFORM_VERTEX_INPUT, DEFORM_VS_MAIN)
    } else {
        (PLAIN_VERTEX_INPUT, PLAIN_VS_MAIN)
    };
    let (cl_bindings, cl_loop) = if clustered {
        (CLUSTERED_BINDINGS, CLUSTERED_LOOP)
    } else {
        ("", "")
    };
    include_str!("default.wgsl")
        .replace("//__VERTEX_INPUT__", vertex_input)
        .replace("//__VS_MAIN__", vs_main)
        .replace("//__CLUSTERED_BINDINGS__", cl_bindings)
        .replace("//__CLUSTERED_LOOP__", cl_loop)
}

/// The vertex buffer layouts shared by the opaque and OIT surface pipelines.
///
/// Returned by value (referencing `const` attribute arrays, hence `'static`) so it
/// can be rebuilt cheaply inside the lazily-built, per-sample-count pipeline
/// builders without borrowing locals.
///
/// We use separate buffers for instance data (positions, colors, deformations)
/// instead of interleaving them, to avoid per-frame data conversion overhead.
fn surface_vertex_buffer_layouts() -> [wgpu::VertexBufferLayout<'static>; 6] {
    // Buffer 0: Vertex positions
    const POSITIONS: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: wgpu::VertexFormat::Float32x3,
    }];
    // Buffer 1: Texture coordinates
    const UVS: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
        offset: 0,
        shader_location: 1,
        format: wgpu::VertexFormat::Float32x2,
    }];
    // Buffer 2: Normals
    const NORMALS: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
        offset: 0,
        shader_location: 2,
        format: wgpu::VertexFormat::Float32x3,
    }];
    // Buffer 3: Instance positions (Point3<f32>)
    const INST_TRA: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
        offset: 0,
        shader_location: 3,
        format: wgpu::VertexFormat::Float32x3,
    }];
    // Buffer 4: Instance colors ([f32; 4])
    const INST_COLOR: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
        offset: 0,
        shader_location: 4,
        format: wgpu::VertexFormat::Float32x4,
    }];
    // Buffer 5: Instance deformations (3x Vector3<f32> = 3 columns of a 3x3 matrix),
    // stored as 3 consecutive vec3s per instance.
    const INST_DEF: [wgpu::VertexAttribute; 3] = [
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 5,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: 12, // 3 * sizeof(f32)
            shader_location: 6,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: 24, // 6 * sizeof(f32)
            shader_location: 7,
            format: wgpu::VertexFormat::Float32x3,
        },
    ];

    [
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &POSITIONS,
        },
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &UVS,
        },
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &NORMALS,
        },
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INST_TRA,
        },
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INST_COLOR,
        },
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 9]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INST_DEF,
        },
    ]
}

impl Default for ObjectMaterial {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectMaterial {
    /// Creates a new `ObjectMaterial`.
    pub fn new() -> ObjectMaterial {
        let ctxt = Context::get();

        // Clustered forward+ needs compute + fragment storage buffers. When the
        // backend supports it (native + WebGPU) the object material uses the
        // clustered pipeline variant (group 0 gains storage bindings 4/5/6);
        // otherwise it falls back to the legacy fixed 8-light path (WebGL2).
        let clustered = ctxt.supports_clustered_lighting();

        // Create bind group layouts. Group 0: frame uniform (0), IBL env (1/2),
        // SSAO (3), and — for the clustered variant — the clustered light list (4),
        // per-cluster light grid (5) and global light-index list (6).
        let mut frame_entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Image-based-lighting environment map (+ sampler).
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
            // Screen-space ambient occlusion (sampled by texel via textureLoad).
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ];
        if clustered {
            for binding in 4..=6 {
                frame_entries.push(wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }
        }
        // Reflection-probe equirectangular array (binding 7, always present). The
        // probe data lives in the frame uniform; this is just the layered texture.
        // Sampled with the IBL sampler (binding 2) — no extra sampler binding.
        frame_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        });
        let frame_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("object_material_frame_bind_group_layout"),
                entries: &frame_entries,
            });

        // Placeholder clustered buffers, bound until the renderer hands over the
        // real ones via `set_clustered_buffers`. Tiny (1 element each); never read
        // while `num_clustered_lights == 0` gates the shader's clustered loop.
        let clustered_lights_buf = ctxt.create_buffer_simple(
            Some("object_material_clustered_lights_placeholder"),
            64,
            wgpu::BufferUsages::STORAGE,
        );
        let cluster_grid_buf = ctxt.create_buffer_simple(
            Some("object_material_cluster_grid_placeholder"),
            8,
            wgpu::BufferUsages::STORAGE,
        );
        let cluster_index_buf = ctxt.create_buffer_simple(
            Some("object_material_cluster_index_placeholder"),
            4,
            wgpu::BufferUsages::STORAGE,
        );

        // 1x1 black fallback environment, bound when no IBL is active.
        let ibl_fallback_texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("object_material_ibl_fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctxt.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ibl_fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0u8; 8], // one Rgba16Float texel = 8 bytes, all zero
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let ibl_fallback_view =
            ibl_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let ibl_fallback_sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("object_material_ibl_fallback_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // 1x1 white fallback AO (= no occlusion), bound when SSAO is off.
        let ao_fallback_texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("object_material_ao_fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctxt.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ao_fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &0x3c00u16.to_le_bytes(), // f16 1.0
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(2),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let ao_fallback_view =
            ao_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 1x1x1 black fallback reflection-probe array, bound when no probes exist
        // (the probe layout binding is always present).
        let probe_fallback_texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("object_material_probe_fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctxt.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &probe_fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0u8; 8],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let probe_fallback_view =
            probe_fallback_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("object_material_probe_fallback_view"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });

        // Object bind group uses dynamic offset for batched uniforms
        let object_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("object_material_object_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true, // Enable dynamic offsets!
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Material-texture bind group layout (group 2): albedo + the four PBR maps
        // (normal, metallic-roughness, ao, emissive), each a texture+sampler pair.
        // Albedo and the PBR maps share one group so the pipeline uses only 4 bind
        // groups total, within WebGPU's `maxBindGroups` limit of 4. Bindings:
        // 0/1 albedo, 2/3 normal, 4/5 metallic-roughness, 6/7 ao, 8/9 emissive.
        // 7 texture+sampler pairs (bindings 0..13): albedo(0/1), normal(2/3),
        // metallic-roughness(4/5), ao(6/7), emissive(8/9), height(10/11), and the
        // per-object planar-reflection texture(12/13).
        let texture_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..7u32)
            .flat_map(|i| {
                [
                    wgpu::BindGroupLayoutEntry {
                        binding: i * 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: i * 2 + 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ]
            })
            .collect();
        let texture_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("object_material_texture_bind_group_layout"),
                entries: &texture_entries,
            });

        // Create default PBR textures
        let default_normal_map = crate::resource::Texture::new_default_normal_map();
        let default_metallic_roughness_map =
            crate::resource::Texture::new_default_metallic_roughness_map();
        let default_ao_map = crate::resource::Texture::new_default_ao_map();
        let default_emissive_map = crate::resource::Texture::new_default_emissive_map();
        let default_height_map = crate::resource::Texture::new_default_height_map();

        // Sampler for the per-object planar reflection (binding 13). Clamp so the
        // projected reflection UV doesn't wrap at the screen edges.
        let reflection_sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("object_material_reflection_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Shadow bind group layout (group 3). Structurally identical to the one the
        // window's `ShadowMapper` builds, so its bind group is compatible here.
        let shadow_bind_group_layout = crate::builtin::shadow::shadow_bind_group_layout(&ctxt);

        // Four bind groups total (frame, object, textures, shadow) — WebGPU caps
        // `maxBindGroups` at 4, so this must not grow.
        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("object_material_pipeline_layout"),
            bind_group_layouts: &[
                Some(&frame_bind_group_layout),
                Some(&object_bind_group_layout),
                Some(&texture_bind_group_layout),
                Some(&shadow_bind_group_layout),
            ],
            immediate_size: 0,
        });

        // Load shader (non-skinned variant). The skinned variant, built below,
        // shares the whole fragment stage and differs only in its vertex input +
        // entry point. When clustered lighting is unsupported,
        // `build_object_shader_src(false, false)` reproduces the original
        // `default.wgsl` byte-for-byte (no storage bindings, WebGL2-safe).
        let shader = ctxt.create_shader_module(
            Some("object_material_shader"),
            &build_object_shader_src(false, clustered),
        );

        // Shared opaque-surface pipeline builder, parameterized by the pipeline
        // layout, shader module, and whether the skinned vertex layout (joints +
        // weights at slots 6/7) is used — so the plain and skinned pipelines reuse
        // one descriptor. Each `PipelineCache` builds lazily per MSAA sample count
        // (the scene is rasterized into the optionally-multisampled HDR film).
        let build_opaque = std::rc::Rc::new(
            |layout: &wgpu::PipelineLayout,
             shader: &wgpu::ShaderModule,
             skinned: bool,
             cull_mode: Option<wgpu::Face>,
             label: &'static str,
             sample_count: u32| {
                let ctxt = Context::get();
                // The deformed pipelines share the plain vertex layout: skin
                // joints/weights and morph deltas come from group-4 storage buffers,
                // not vertex attributes. `skinned` only selects the shader + layout.
                let _ = skinned;
                let plain_layouts = surface_vertex_buffer_layouts();
                let buffers: &[wgpu::VertexBufferLayout] = &plain_layouts;
                ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some("vs_main"),
                        buffers,
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: Context::render_format(), // HDR rasterization target (tonemapped to LDR in the resolve pass)
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: Context::depth_format(),
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: multisample_state(sample_count),
                    multiview_mask: None,
                    cache: None,
                })
            },
        );

        let pipeline_cull = PipelineCache::new({
            let build = build_opaque.clone();
            let l = pipeline_layout.clone();
            let s = shader.clone();
            move |sc| {
                build(
                    &l,
                    &s,
                    false,
                    Some(wgpu::Face::Back),
                    "object_material_pipeline_cull",
                    sc,
                )
            }
        });
        let pipeline_no_cull = PipelineCache::new({
            let build = build_opaque.clone();
            let l = pipeline_layout.clone();
            let s = shader.clone();
            move |sc| build(&l, &s, false, None, "object_material_pipeline_no_cull", sc)
        });

        // Weighted-blended OIT pipelines: same vertex stage and bind groups, but the
        // `fs_oit` entry point writes two targets — an additive premultiplied-weighted
        // color accumulator (Rgba16Float) and a multiplicative revealage (R16Float) —
        // and depth-tests against the opaque depth without writing it. Built lazily per
        // sample count; the OIT geometry targets are multisampled to match the (MSAA)
        // opaque depth buffer, then resolved before compositing.
        let build_oit = std::rc::Rc::new(
            |layout: &wgpu::PipelineLayout,
             shader: &wgpu::ShaderModule,
             skinned: bool,
             cull_mode: Option<wgpu::Face>,
             label: &'static str,
             sample_count: u32| {
                let ctxt = Context::get();
                // The deformed pipelines share the plain vertex layout: skin
                // joints/weights and morph deltas come from group-4 storage buffers,
                // not vertex attributes. `skinned` only selects the shader + layout.
                let _ = skinned;
                let plain_layouts = surface_vertex_buffer_layouts();
                let buffers: &[wgpu::VertexBufferLayout] = &plain_layouts;
                ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some("vs_main"),
                        buffers,
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
                        entry_point: Some("fs_oit"),
                        targets: &[
                            // accum: additive (One, One).
                            Some(wgpu::ColorTargetState {
                                format: OIT_ACCUM_FORMAT,
                                blend: Some(wgpu::BlendState {
                                    color: wgpu::BlendComponent {
                                        src_factor: wgpu::BlendFactor::One,
                                        dst_factor: wgpu::BlendFactor::One,
                                        operation: wgpu::BlendOperation::Add,
                                    },
                                    alpha: wgpu::BlendComponent {
                                        src_factor: wgpu::BlendFactor::One,
                                        dst_factor: wgpu::BlendFactor::One,
                                        operation: wgpu::BlendOperation::Add,
                                    },
                                }),
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            // revealage: multiplicative dst *= (1 - src).
                            Some(wgpu::ColorTargetState {
                                format: OIT_REVEAL_FORMAT,
                                blend: Some(wgpu::BlendState {
                                    color: wgpu::BlendComponent {
                                        src_factor: wgpu::BlendFactor::Zero,
                                        dst_factor: wgpu::BlendFactor::OneMinusSrc,
                                        operation: wgpu::BlendOperation::Add,
                                    },
                                    alpha: wgpu::BlendComponent::REPLACE,
                                }),
                                write_mask: wgpu::ColorWrites::RED,
                            }),
                        ],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: Context::depth_format(),
                        // Test against opaque depth, but do not write (transparent
                        // fragments must not occlude each other).
                        depth_write_enabled: Some(false),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: multisample_state(sample_count),
                    multiview_mask: None,
                    cache: None,
                })
            },
        );
        let oit_pipeline_cull = PipelineCache::new({
            let build = build_oit.clone();
            let l = pipeline_layout.clone();
            let s = shader.clone();
            move |sc| {
                build(
                    &l,
                    &s,
                    false,
                    Some(wgpu::Face::Back),
                    "object_material_oit_pipeline_cull",
                    sc,
                )
            }
        });
        let oit_pipeline_no_cull = PipelineCache::new({
            let build = build_oit.clone();
            let l = pipeline_layout.clone();
            let s = shader.clone();
            move |sc| {
                build(
                    &l,
                    &s,
                    false,
                    None,
                    "object_material_oit_pipeline_no_cull",
                    sc,
                )
            }
        });

        // Depth + view-position prepass pipeline: reuses the surface vertex stage
        // and the full pipeline layout (so the per-object bind calls are
        // unchanged), with a minimal fragment writing view-space position into a
        // single Rgba16Float target. Single-sampled (SSAO runs at 1x).
        let build_prepass = std::rc::Rc::new(
            |layout: &wgpu::PipelineLayout,
             shader: &wgpu::ShaderModule,
             skinned: bool,
             sample_count: u32| {
                let ctxt = Context::get();
                // The deformed pipelines share the plain vertex layout: skin
                // joints/weights and morph deltas come from group-4 storage buffers,
                // not vertex attributes. `skinned` only selects the shader + layout.
                let _ = skinned;
                let plain_layouts = surface_vertex_buffer_layouts();
                let buffers: &[wgpu::VertexBufferLayout] = &plain_layouts;
                ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("object_material_prepass_pipeline"),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some("vs_main"),
                        buffers,
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
                        entry_point: Some("fs_prepass"),
                        // G-buffer MRT: viewpos, world-normal+roughness, F0+metallic,
                        // per-object SSR params. SSAO reads only target 0; the rest
                        // feed SSR.
                        targets: &[
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba16Float,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba16Float,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba16Float,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba16Float,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                        ],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: Context::depth_format(),
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: multisample_state(sample_count),
                    multiview_mask: None,
                    cache: None,
                })
            },
        );
        let prepass_pipeline = PipelineCache::new({
            let build = build_prepass.clone();
            let l = pipeline_layout.clone();
            let s = shader.clone();
            move |sc| build(&l, &s, false, sc)
        });

        // Deformed pipeline variants (native only). The deform bind group is a 5th
        // bind group, which exceeds WebGPU/WebGL2's 4-group cap, so on web (or any
        // adapter without a free group) skinned/morphed objects fall back to the
        // plain pipeline and render in their base rest shape. Built lazily per sample
        // count like the plain pipelines, sharing the same builder closures. The
        // deform bind-group layout is the shared one from `builtin::deform`, so the
        // per-object bind group also works in the shadow pipelines.
        #[cfg(not(target_arch = "wasm32"))]
        let deform = {
            let deform_bind_group_layout = crate::builtin::deform::deform_bind_group_layout();
            let deform_pipeline_layout =
                ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("object_material_deform_pipeline_layout"),
                    bind_group_layouts: &[
                        Some(&frame_bind_group_layout),
                        Some(&object_bind_group_layout),
                        Some(&texture_bind_group_layout),
                        Some(&shadow_bind_group_layout),
                        Some(&deform_bind_group_layout),
                    ],
                    immediate_size: 0,
                });
            let deform_shader = ctxt.create_shader_module(
                Some("object_material_deform_shader"),
                &build_object_shader_src(true, clustered),
            );

            let pipeline_cull = PipelineCache::new({
                let build = build_opaque.clone();
                let l = deform_pipeline_layout.clone();
                let s = deform_shader.clone();
                move |sc| {
                    build(
                        &l,
                        &s,
                        true,
                        Some(wgpu::Face::Back),
                        "object_material_deform_cull",
                        sc,
                    )
                }
            });
            let pipeline_no_cull = PipelineCache::new({
                let build = build_opaque.clone();
                let l = deform_pipeline_layout.clone();
                let s = deform_shader.clone();
                move |sc| build(&l, &s, true, None, "object_material_deform_no_cull", sc)
            });
            let oit_pipeline_cull = PipelineCache::new({
                let build = build_oit.clone();
                let l = deform_pipeline_layout.clone();
                let s = deform_shader.clone();
                move |sc| {
                    build(
                        &l,
                        &s,
                        true,
                        Some(wgpu::Face::Back),
                        "object_material_deform_oit_cull",
                        sc,
                    )
                }
            });
            let oit_pipeline_no_cull = PipelineCache::new({
                let build = build_oit.clone();
                let l = deform_pipeline_layout.clone();
                let s = deform_shader.clone();
                move |sc| build(&l, &s, true, None, "object_material_deform_oit_no_cull", sc)
            });
            let prepass_pipeline = PipelineCache::new({
                let build = build_prepass.clone();
                let l = deform_pipeline_layout.clone();
                let s = deform_shader.clone();
                move |sc| build(&l, &s, true, sc)
            });

            Some(DeformResources {
                pipeline_cull,
                pipeline_no_cull,
                oit_pipeline_cull,
                oit_pipeline_no_cull,
                prepass_pipeline,
            })
        };
        #[cfg(target_arch = "wasm32")]
        let deform: Option<DeformResources> = None;

        // Create wireframe shader and pipelines for lines/points
        // Note: _wireframe_shader, _wireframe_pipeline_layout, and _wireframe_vertex_buffer_layouts
        // were previously used for the old PointList pipeline but are now replaced by the new
        // wireframe_points.wgsl shader. Keeping them here in case they're needed for 1px fallback.
        let _wireframe_shader =
            ctxt.create_shader_module(Some("wireframe_shader"), include_str!("wireframe.wgsl"));

        // Pipeline layout for wireframe (only needs frame and object uniforms, no texture)
        let _wireframe_pipeline_layout =
            ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("wireframe_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&frame_bind_group_layout),
                    Some(&object_bind_group_layout),
                ],
                immediate_size: 0,
            });

        // Vertex buffer layouts for wireframe (position only + instance data)
        let _wireframe_vertex_buffer_layouts = [
            // Buffer 0: Vertex positions
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            // Buffer 1: Instance positions (Point3<f32>)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3, // inst_tra
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            // Buffer 2: Instance colors ([f32; 4])
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 4, // inst_color
                    format: wgpu::VertexFormat::Float32x4,
                }],
            },
            // Buffer 3: Instance deformations (3x Vector3<f32> = 3 columns of 3x3 matrix)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 9]>() as wgpu::BufferAddress, // 3 vec3s
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    // inst_def_0 (column 0)
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 5,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    // inst_def_1 (column 1)
                    wgpu::VertexAttribute {
                        offset: 12, // 3 * sizeof(f32)
                        shader_location: 6,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    // inst_def_2 (column 2)
                    wgpu::VertexAttribute {
                        offset: 24, // 6 * sizeof(f32)
                        shader_location: 7,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            },
        ];

        // Create wireframe bind group layouts
        let wireframe_view_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("wireframe_view_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let wireframe_model_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("wireframe_model_bind_group_layout"),
                entries: &[
                    // Model uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Edge storage buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let wireframe_polyline_pipeline_layout =
            ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("wireframe_polyline_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&wireframe_view_bind_group_layout),
                    Some(&wireframe_model_bind_group_layout),
                ],
                immediate_size: 0,
            });

        // Load wireframe polyline shader
        let wireframe_polyline_shader = ctxt.create_shader_module(
            Some("wireframe_polyline_shader"),
            include_str!("wireframe_polyline3d.wgsl"),
        );

        // Wireframe pipeline, built lazily per MSAA sample count (lines render into
        // the optionally-multisampled HDR film alongside surfaces).
        let wireframe_pipeline = PipelineCache::new(move |sample_count| {
            let ctxt = Context::get();
            // Instance vertex buffer layouts for wireframe (matching InstancesBuffer)
            let wireframe_instance_buffer_layouts = [
                // Buffer 0: positions (Point3<f32>)
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    }],
                },
                // Buffer 1: colors ([f32; 4]) - not used but needed for layout consistency
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x4,
                    }],
                },
                // Buffer 2: deformations - all 3 columns from same buffer with stride = 3*vec3
                // Matrix3 is stored as 3 consecutive Vector3 columns (36 bytes total)
                wgpu::VertexBufferLayout {
                    array_stride: (std::mem::size_of::<[f32; 3]>() * 3) as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        // Column 0 at offset 0
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        // Column 1 at offset 12
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 3]>() as u64,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        // Column 2 at offset 24
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>() * 2) as u64,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                    ],
                },
                // Buffer 3: lines_colors ([f32; 4])
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 5,
                        format: wgpu::VertexFormat::Float32x4,
                    }],
                },
                // Buffer 4: lines_widths (f32)
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 6,
                        format: wgpu::VertexFormat::Float32,
                    }],
                },
            ];

            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("wireframe_polyline_pipeline"),
                layout: Some(&wireframe_polyline_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &wireframe_polyline_shader,
                    entry_point: Some("vs_main"),
                    buffers: &wireframe_instance_buffer_layouts,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &wireframe_polyline_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: Context::render_format(), // HDR rasterization target (tonemapped to LDR in the resolve pass)
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
                    format: Context::depth_format(),
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: multisample_state(sample_count),
                multiview_mask: None,
                cache: None,
            })
        });

        // Create points bind group layouts (same view layout as wireframe, different model layout)
        let points_view_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("points_view_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let points_model_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("points_model_bind_group_layout"),
                entries: &[
                    // Model uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Vertex storage buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let points_pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("points_pipeline_layout"),
            bind_group_layouts: &[
                Some(&points_view_bind_group_layout),
                Some(&points_model_bind_group_layout),
            ],
            immediate_size: 0,
        });

        // Load points shader
        let points_shader = ctxt.create_shader_module(
            Some("wireframe_points_shader"),
            include_str!("wireframe_points3d.wgsl"),
        );

        // Points pipeline, built lazily per MSAA sample count (points render into
        // the optionally-multisampled HDR film alongside surfaces).
        let points_pipeline = PipelineCache::new(move |sample_count| {
            let ctxt = Context::get();
            // Instance vertex buffer layouts for points (similar to wireframe but with points_colors/sizes)
            let points_instance_buffer_layouts = [
                // Buffer 0: positions (Point3<f32>)
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    }],
                },
                // Buffer 1: colors ([f32; 4]) - not used but needed for layout consistency
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x4,
                    }],
                },
                // Buffer 2: deformations - all 3 columns from same buffer with stride = 3*vec3
                wgpu::VertexBufferLayout {
                    array_stride: (std::mem::size_of::<[f32; 3]>() * 3) as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 3]>() as u64,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>() * 2) as u64,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                    ],
                },
                // Buffer 3: points_colors ([f32; 4])
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 5,
                        format: wgpu::VertexFormat::Float32x4,
                    }],
                },
                // Buffer 4: points_sizes (f32)
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 6,
                        format: wgpu::VertexFormat::Float32,
                    }],
                },
            ];

            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("wireframe_points_pipeline"),
                layout: Some(&points_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &points_shader,
                    entry_point: Some("vs_main"),
                    buffers: &points_instance_buffer_layouts,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &points_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: Context::render_format(), // HDR rasterization target (tonemapped to LDR in the resolve pass)
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
                    format: Context::depth_format(),
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: multisample_state(sample_count),
                multiview_mask: None,
                cache: None,
            })
        });

        // === Create shared dynamic buffer resources ===

        // Frame uniform buffer (written once per frame)
        let frame_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shared_frame_uniform_buffer"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create frame bind group
        let mut frame_group_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&ibl_fallback_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&ibl_fallback_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&ao_fallback_view),
            },
        ];
        if clustered {
            frame_group_entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: clustered_lights_buf.as_entire_binding(),
            });
            frame_group_entries.push(wgpu::BindGroupEntry {
                binding: 5,
                resource: cluster_grid_buf.as_entire_binding(),
            });
            frame_group_entries.push(wgpu::BindGroupEntry {
                binding: 6,
                resource: cluster_index_buf.as_entire_binding(),
            });
        }
        frame_group_entries.push(wgpu::BindGroupEntry {
            binding: 7,
            resource: wgpu::BindingResource::TextureView(&probe_fallback_view),
        });
        let frame_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared_frame_bind_group"),
            layout: &frame_bind_group_layout,
            entries: &frame_group_entries,
        });

        // Dynamic buffer for object uniforms
        let object_uniform_buffer =
            DynamicUniformBuffer::<ObjectUniforms>::new("dynamic_object_uniform_buffer");

        // Create initial object bind group
        let object_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dynamic_object_bind_group"),
            layout: &object_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: object_uniform_buffer.buffer(),
                    offset: 0,
                    size: std::num::NonZeroU64::new(object_uniform_buffer.aligned_size()),
                }),
            }],
        });

        // Get capacity before moving into struct
        let object_bind_group_capacity = object_uniform_buffer.capacity();

        // === Shared wireframe/points view uniform buffers ===
        // These contain view, proj, and viewport which are the same for all objects in a frame

        let wireframe_view_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shared_wireframe_view_uniform_buffer"),
            size: std::mem::size_of::<WireframeViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let wireframe_view_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared_wireframe_view_bind_group"),
            layout: &wireframe_view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wireframe_view_uniform_buffer.as_entire_binding(),
            }],
        });

        let points_view_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shared_points_view_uniform_buffer"),
            size: std::mem::size_of::<WireframeViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let points_view_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared_points_view_bind_group"),
            layout: &points_view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: points_view_uniform_buffer.as_entire_binding(),
            }],
        });

        // === Neutral fallback shadow resources ===
        // A 1x1xMAX_SHADOW_VIEWS dummy depth atlas plus a zeroed uniform buffer
        // (`shadows_enabled == 0`). Bound at group 4 whenever no shadow mapper is
        // active, this keeps the lighting shader correct with shadows disabled.
        let dummy_atlas = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("object_material_dummy_shadow_atlas"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: crate::builtin::shadow::MAX_SHADOW_VIEWS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_atlas_view = dummy_atlas.create_view(&wgpu::TextureViewDescriptor {
            label: Some("object_material_dummy_shadow_atlas_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let dummy_shadow_sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("object_material_dummy_shadow_sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        // Size matches `ShadowUniforms`; zero-initialized means `shadows_enabled == 0`.
        let dummy_shadow_uniform = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object_material_dummy_shadow_uniform"),
            size: crate::builtin::shadow::shadow_uniforms_size(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctxt.write_buffer(
            &dummy_shadow_uniform,
            0,
            &vec![0u8; crate::builtin::shadow::shadow_uniforms_size() as usize],
        );
        // Dummy colored transmittance atlas + filtering sampler. The zeroed shadow
        // uniform sets `shadows_enabled == 0`, so the shader never actually samples
        // these — they only need to exist to satisfy the bind group layout.
        let dummy_transmittance_atlas = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("object_material_dummy_transmittance_atlas"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: crate::builtin::shadow::MAX_SHADOW_VIEWS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_transmittance_view =
            dummy_transmittance_atlas.create_view(&wgpu::TextureViewDescriptor {
                label: Some("object_material_dummy_transmittance_view"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
        let dummy_transmittance_sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("object_material_dummy_transmittance_sampler"),
            ..Default::default()
        });
        let default_shadow_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("object_material_default_shadow_bind_group"),
            layout: &shadow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dummy_atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&dummy_shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dummy_shadow_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&dummy_transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&dummy_transmittance_sampler),
                },
            ],
        });
        let default_shadow_resources = DefaultShadowResources {
            _atlas: dummy_atlas,
            _view: dummy_atlas_view,
            _sampler: dummy_shadow_sampler,
            _uniform: dummy_shadow_uniform,
            _transmittance_atlas: dummy_transmittance_atlas,
            _transmittance_view: dummy_transmittance_view,
            _transmittance_sampler: dummy_transmittance_sampler,
        };

        ObjectMaterial {
            pipeline_cull,
            pipeline_no_cull,
            oit_pipeline_cull,
            oit_pipeline_no_cull,
            prepass_pipeline,
            object_bind_group_layout,
            texture_bind_group_layout,
            default_normal_map,
            default_metallic_roughness_map,
            default_ao_map,
            default_emissive_map,
            default_height_map,
            reflection_sampler,
            wireframe_pipeline,
            wireframe_model_bind_group_layout,
            points_pipeline,
            points_model_bind_group_layout,
            frame_uniform_buffer,
            frame_bind_group,
            frame_bind_group_layout,
            clustered,
            clustered_lights_buf,
            cluster_grid_buf,
            cluster_index_buf,
            clustered_bound: false,
            cur_ibl_view: ibl_fallback_view.clone(),
            cur_ibl_sampler: ibl_fallback_sampler.clone(),
            cur_ao_view: ao_fallback_view.clone(),
            _ibl_fallback_texture: ibl_fallback_texture,
            ibl_fallback_view,
            ibl_fallback_sampler,
            _ao_fallback_texture: ao_fallback_texture,
            ao_fallback_view,
            ibl_bound_ptr: 0,
            ao_bound_ptr: 0,
            ibl_has: Cell::new(false),
            ibl_max_lod: Cell::new(0.0),
            ibl_intensity: Cell::new(1.0),
            ibl_rotation: Cell::new(0.0),
            ssao_has: Cell::new(false),
            capture_mode: Cell::new(false),
            cur_probe_view: probe_fallback_view.clone(),
            _probe_fallback_texture: probe_fallback_texture,
            probe_fallback_view,
            probe_bound_ptr: 0,
            probe_records: Cell::new([GpuProbe::default(); MAX_PROBES]),
            probe_count: Cell::new(0),
            clip_plane: Cell::new([0.0; 4]),
            object_uniform_buffer,
            object_bind_group: Some(object_bind_group),
            object_bind_group_capacity,
            frame_counter: Cell::new(0),
            last_frame: Cell::new(u64::MAX),
            wireframe_view_uniform_buffer,
            wireframe_view_bind_group,
            points_view_uniform_buffer,
            points_view_bind_group,
            default_shadow_bind_group,
            _default_shadow_resources: default_shadow_resources,
            deform,
        }
    }

    /// Builds the combined material-texture bind group (group 2): albedo at
    /// bindings 0/1 followed by the four PBR maps at 2/3, 4/5, 6/7, 8/9.
    /// Rebuilds the shared frame bind group (group 0) from the current per-view
    /// resources: the frame uniform, the IBL environment, and the SSAO texture.
    fn rebuild_frame_bind_group(&mut self) {
        let ctxt = Context::get();
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.frame_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&self.cur_ibl_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&self.cur_ibl_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&self.cur_ao_view),
            },
        ];
        if self.clustered {
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: self.clustered_lights_buf.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 5,
                resource: self.cluster_grid_buf.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 6,
                resource: self.cluster_index_buf.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 7,
            resource: wgpu::BindingResource::TextureView(&self.cur_probe_view),
        });
        self.frame_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared_frame_bind_group"),
            layout: &self.frame_bind_group_layout,
            entries: &entries,
        });
    }

    fn create_texture_bind_group(
        &self,
        albedo: &Texture,
        normal_map: &Texture,
        metallic_roughness_map: &Texture,
        ao_map: &Texture,
        emissive_map: &Texture,
        height_map: &Texture,
        reflection_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let ctxt = Context::get();
        let textures = [
            albedo,
            normal_map,
            metallic_roughness_map,
            ao_map,
            emissive_map,
            height_map,
        ];
        let mut entries: Vec<wgpu::BindGroupEntry> = textures
            .iter()
            .enumerate()
            .flat_map(|(i, tex)| {
                let i = i as u32;
                [
                    wgpu::BindGroupEntry {
                        binding: i * 2,
                        resource: wgpu::BindingResource::TextureView(&tex.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: i * 2 + 1,
                        resource: wgpu::BindingResource::Sampler(&tex.sampler),
                    },
                ]
            })
            .collect();
        // Per-object planar reflection (binding 12/13): the reflector's target, or a
        // 1x1 fallback when the object isn't a reflector. Uses the dedicated clamp
        // sampler so the projected UV doesn't wrap.
        entries.push(wgpu::BindGroupEntry {
            binding: 12,
            resource: wgpu::BindingResource::TextureView(reflection_view),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 13,
            resource: wgpu::BindingResource::Sampler(&self.reflection_sampler),
        });
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("object_material_texture_bind_group"),
            layout: &self.texture_bind_group_layout,
            entries: &entries,
        })
    }

    fn create_wireframe_model_bind_group(
        &self,
        model_buffer: &wgpu::Buffer,
        edge_buffer: &wgpu::Buffer,
        edge_size: u64,
    ) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wireframe_model_bind_group"),
            layout: &self.wireframe_model_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: edge_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(edge_size),
                    }),
                },
            ],
        })
    }

    fn create_points_model_bind_group(
        &self,
        model_buffer: &wgpu::Buffer,
        vertex_buffer: &wgpu::Buffer,
        vertex_size: u64,
    ) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("points_model_bind_group"),
            layout: &self.points_model_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: vertex_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(vertex_size),
                    }),
                },
            ],
        })
    }

    /// Signals the start of a new frame.
    ///
    /// This clears the dynamic object uniform buffer and resets the frame counter.
    /// Should be called before rendering any objects for a new frame.
    pub fn begin_frame(&mut self) {
        self.frame_counter
            .set(self.frame_counter.get().wrapping_add(1));
        self.object_uniform_buffer.clear();
    }

    /// Flushes the accumulated object uniforms to the GPU.
    ///
    /// This performs a single `write_buffer` call with all accumulated object data.
    /// Should be called after all objects have been processed for the frame.
    pub fn flush(&mut self) {
        let ctxt = Context::get();

        self.object_uniform_buffer.flush();

        // Recreate bind group if buffer grew
        if self.object_uniform_buffer.capacity() != self.object_bind_group_capacity {
            self.object_bind_group = Some(ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dynamic_object_bind_group"),
                layout: &self.object_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: self.object_uniform_buffer.buffer(),
                        offset: 0,
                        size: std::num::NonZeroU64::new(self.object_uniform_buffer.aligned_size()),
                    }),
                }],
            }));
            self.object_bind_group_capacity = self.object_uniform_buffer.capacity();
        }
    }
}

impl Material3d for ObjectMaterial {
    fn create_gpu_data(&self) -> Box<dyn GpuData> {
        Box::new(ObjectMaterialGpuData::new())
    }

    fn begin_frame(&mut self) {
        self.frame_counter
            .set(self.frame_counter.get().wrapping_add(1));
        self.object_uniform_buffer.clear();
    }

    fn prepare(
        &mut self,
        pass: usize,
        transform: Pose3,
        scale: Vec3,
        camera: &mut dyn Camera3d,
        lights: &LightCollection,
        data: &ObjectData3d,
        gpu_data: &mut dyn GpuData,
        viewport_width: u32,
        viewport_height: u32,
    ) {
        let ctxt = Context::get();

        // Downcast gpu_data to our specific type
        let gpu_data = gpu_data
            .as_any_mut()
            .downcast_mut::<ObjectMaterialGpuData>()
            .expect("ObjectMaterial requires ObjectMaterialGpuData");

        // Check if this is a new frame (first object being prepared)
        let current_frame = self.frame_counter.get();
        let is_new_frame = current_frame != self.last_frame.get();

        if is_new_frame {
            self.last_frame.set(current_frame);

            // Write frame uniforms once per frame
            let (view, proj) = camera.view_transform_pair(pass);

            // Split into the primary tier (this fixed uniform array, with shadows)
            // and the clustered overflow tier. The clustered lights are uploaded and
            // shaded separately by `crate::builtin::clustered`; here we only fill the
            // primary slots, in the exact order the shadow atlas also uses.
            let (primary, clustered) = lights.split_primary_clustered();
            let mut gpu_lights: [GpuLight; MAX_LIGHTS] = [GpuLight::default(); MAX_LIGHTS];
            for (slot, &li) in primary.iter().take(MAX_LIGHTS).enumerate() {
                gpu_lights[slot] = GpuLight::from_collected(&lights.lights[li]);
            }
            let num_primary = primary.len().min(MAX_LIGHTS) as u32;
            let num_clustered = clustered.len() as u32;

            let frame_uniforms = FrameUniforms {
                view: view.to_mat4().to_cols_array_2d(),
                proj: proj.to_cols_array_2d(),
                lights: gpu_lights,
                num_lights: num_primary,
                ambient_intensity: lights.ambient,
                _padding: [0.0; 2],
                ambient_color: [
                    lights.ambient_color.r,
                    lights.ambient_color.g,
                    lights.ambient_color.b,
                    1.0,
                ],
                fog_color: [
                    lights.fog.color.r,
                    lights.fog.color.g,
                    lights.fog.color.b,
                    lights.fog.color.a,
                ],
                fog_params: lights.fog.params(),
                camera_pos: {
                    let e = camera.eye();
                    // w = SSAO-enabled flag (gates the AO sample in the shader).
                    [e.x, e.y, e.z, if self.ssao_has.get() { 1.0 } else { 0.0 }]
                },
                ibl_params: [
                    if self.ibl_has.get() { 1.0 } else { 0.0 },
                    self.ibl_max_lod.get(),
                    self.ibl_intensity.get(),
                    self.ibl_rotation.get(),
                ],
                cluster_grid_dims: {
                    use crate::builtin::clustered::{GRID_X, GRID_Y, GRID_Z};
                    let n = if self.clustered && !self.capture_mode.get() {
                        num_clustered
                    } else {
                        0
                    };
                    [GRID_X as f32, GRID_Y as f32, GRID_Z as f32, n as f32]
                },
                cluster_depth: {
                    let (near, far) = camera.clip_planes();
                    [near, far, (far / near).ln(), 0.0]
                },
                cluster_tile: {
                    use crate::builtin::clustered::{GRID_X, GRID_Y};
                    [
                        viewport_width as f32 / GRID_X as f32,
                        viewport_height as f32 / GRID_Y as f32,
                        0.0,
                        0.0,
                    ]
                },
                // Probes are suppressed during capture so captured surfaces reflect
                // only the skybox/IBL, not the probe being captured (no feedback).
                probe_count: [
                    if self.capture_mode.get() {
                        0
                    } else {
                        self.probe_count.get()
                    },
                    0,
                    0,
                    0,
                ],
                clip_plane: self.clip_plane.get(),
                probes: self.probe_records.get(),
            };

            ctxt.write_buffer(
                &self.frame_uniform_buffer,
                0,
                bytemuck::bytes_of(&frame_uniforms),
            );

            // Write shared wireframe/points view uniforms once per frame
            // These contain view, proj, viewport which are same for all objects
            let wireframe_view_uniforms = WireframeViewUniforms {
                view: view.to_mat4().to_cols_array_2d(),
                proj: proj.to_cols_array_2d(),
                viewport: [0.0, 0.0, viewport_width as f32, viewport_height as f32],
            };

            ctxt.write_buffer(
                &self.wireframe_view_uniform_buffer,
                0,
                bytemuck::bytes_of(&wireframe_view_uniforms),
            );

            // Points use the same view uniform layout
            ctxt.write_buffer(
                &self.points_view_uniform_buffer,
                0,
                bytemuck::bytes_of(&wireframe_view_uniforms),
            );
        }

        // Create object uniforms
        let formatted_transform = transform.to_mat4();
        let ntransform = Mat3::from_quat(transform.rotation);
        let formatted_scale = Mat3::from_diagonal(scale);

        // Pad mat3x3 to mat3x4 for proper alignment
        let ntransform_cols = ntransform.to_cols_array_2d();
        let ntransform_padded: [[f32; 4]; 3] = [
            [
                ntransform_cols[0][0],
                ntransform_cols[0][1],
                ntransform_cols[0][2],
                0.0,
            ],
            [
                ntransform_cols[1][0],
                ntransform_cols[1][1],
                ntransform_cols[1][2],
                0.0,
            ],
            [
                ntransform_cols[2][0],
                ntransform_cols[2][1],
                ntransform_cols[2][2],
                0.0,
            ],
        ];
        let scale_cols = formatted_scale.to_cols_array_2d();
        let scale_padded: [[f32; 4]; 3] = [
            [scale_cols[0][0], scale_cols[0][1], scale_cols[0][2], 0.0],
            [scale_cols[1][0], scale_cols[1][1], scale_cols[1][2], 0.0],
            [scale_cols[2][0], scale_cols[2][1], scale_cols[2][2], 0.0],
        ];

        let color = data.color();
        let emissive = data.emissive();
        let object_uniforms = ObjectUniforms {
            transform: formatted_transform.to_cols_array_2d(),
            ntransform: ntransform_padded,
            scale: scale_padded,
            color: [color.r, color.g, color.b, color.a],
            metallic: data.metallic(),
            roughness: data.roughness(),
            _pad0: [0.0; 2],
            emissive: [emissive.r, emissive.g, emissive.b, emissive.a],
            has_normal_map: if data.normal_map().is_some() {
                1.0
            } else {
                0.0
            },
            has_metallic_roughness_map: if data.metallic_roughness_map().is_some() {
                1.0
            } else {
                0.0
            },
            has_ao_map: if data.ao_map().is_some() { 1.0 } else { 0.0 },
            has_emissive_map: if data.emissive_map().is_some() {
                1.0
            } else {
                0.0
            },
            reflectance: data.reflectance(),
            clearcoat: data.clearcoat(),
            clearcoat_roughness: data.clearcoat_roughness(),
            anisotropy: data.anisotropy(),
            anisotropy_rotation: data.anisotropy_rotation(),
            transmission: data.transmission(),
            alpha_mode: {
                let (code, _) = data.alpha_mode().shader_params();
                code as f32
            },
            alpha_cutoff: data.alpha_mode().shader_params().1,
            specular_tint: {
                let t = data.specular_tint();
                [t.r, t.g, t.b, t.a]
            },
            parallax: [
                if data.height_map().is_some() {
                    1.0
                } else {
                    0.0
                },
                data.parallax_scale(),
                data.parallax_layers(),
                data.parallax_method().code(),
            ],
            ssr: crate::renderer::SsrMaterial::pack(data.ssr()),
            reflector_view_proj: match data.reflector() {
                Some(r) => r.view_proj().to_cols_array_2d(),
                None => glamx::Mat4::IDENTITY.to_cols_array_2d(),
            },
            reflection_params: match data.reflector() {
                Some(r) => [r.intensity(), 1.0, r.normal_falloff(), 0.0],
                None => [0.0; 4],
            },
            reflector_normal: match data.reflector() {
                // World plane normal = object world rotation * the reflector's
                // object-space normal (the falloff compares the surface normal to it).
                Some(r) => {
                    let n = (transform.rotation * r.local_normal()).normalize();
                    [n.x, n.y, n.z, 0.0]
                }
                None => [0.0; 4],
            },
        };

        // Push to dynamic buffer and store offset in gpu_data
        let object_offset = self.object_uniform_buffer.push(&object_uniforms);
        gpu_data.object_uniform_offset = Some(object_offset);

        // Prepare wireframe model uniforms if needed (view uniforms are shared)
        let render_wireframe = data.lines_width() > 0.0;
        if render_wireframe {
            // Compute model uniforms (num_edges will be set in render when mesh is available)
            let wireframe_color = data.lines_color().unwrap_or(data.color());
            let cached_num_edges = gpu_data
                .wireframe_edges
                .as_ref()
                .map(|e| e.len())
                .unwrap_or(0) as u32;
            gpu_data.wireframe_model_uniforms = WireframeModelUniforms {
                transform: formatted_transform.to_cols_array_2d(),
                scale: scale.into(),
                num_edges: cached_num_edges,
                default_color: [
                    wireframe_color.r,
                    wireframe_color.g,
                    wireframe_color.b,
                    wireframe_color.a,
                ],
                default_width: data.lines_width(),
                use_perspective: if data.lines_use_perspective() { 1 } else { 0 },
                _padding: [0.0; 2],
            };

            // Write model uniforms to GPU (view uniforms are shared and written once per frame)
            ctxt.write_buffer(
                &gpu_data.wireframe_model_uniform_buffer,
                0,
                bytemuck::bytes_of(&gpu_data.wireframe_model_uniforms),
            );
        }

        // Prepare points model uniforms if needed (view uniforms are shared)
        let render_points = data.points_size() > 0.0;
        if render_points {
            // Compute model uniforms (num_vertices will be set in render when mesh is available)
            let points_color = data.points_color().unwrap_or(data.color());
            let cached_num_vertices = gpu_data
                .points_vertices
                .as_ref()
                .map(|v| v.len())
                .unwrap_or(0) as u32;
            gpu_data.points_model_uniforms = PointsModelUniforms {
                transform: formatted_transform.to_cols_array_2d(),
                scale: scale.into(),
                num_vertices: cached_num_vertices,
                default_color: [
                    points_color.r,
                    points_color.g,
                    points_color.b,
                    points_color.a,
                ],
                default_size: data.points_size(),
                use_perspective: if data.points_use_perspective() { 1 } else { 0 },
                _padding: [0.0; 2],
            };

            // Write model uniforms to GPU (view uniforms are shared and written once per frame)
            ctxt.write_buffer(
                &gpu_data.points_model_uniform_buffer,
                0,
                bytemuck::bytes_of(&gpu_data.points_model_uniforms),
            );
        }
    }

    fn renders_in_transparent_phase(&self) -> bool {
        // ObjectMaterial has dedicated OIT pipelines whose targets match the
        // transparent (weighted-blended) pass, so it draws in both phases.
        true
    }

    fn set_environment_lighting(&mut self, env: Option<crate::resource::EnvLight<'_>>) {
        match env {
            Some(e) => {
                self.ibl_has.set(true);
                self.ibl_max_lod.set((e.mip_count.max(1) - 1) as f32);
                self.ibl_intensity.set(e.intensity);
                self.ibl_rotation.set(e.rotation);
                let ptr = e.view as *const wgpu::TextureView as usize;
                if self.ibl_bound_ptr != ptr {
                    self.cur_ibl_view = e.view.clone();
                    self.cur_ibl_sampler = e.sampler.clone();
                    self.ibl_bound_ptr = ptr;
                    self.rebuild_frame_bind_group();
                }
            }
            None => {
                self.ibl_has.set(false);
                // Rebind the fallback if a real env was bound (it may be dropped).
                if self.ibl_bound_ptr != 0 {
                    self.cur_ibl_view = self.ibl_fallback_view.clone();
                    self.cur_ibl_sampler = self.ibl_fallback_sampler.clone();
                    self.ibl_bound_ptr = 0;
                    self.rebuild_frame_bind_group();
                }
            }
        }
    }

    fn set_reflection_probes(&mut self, probes: Option<crate::resource::ProbeLighting<'_>>) {
        match probes {
            Some(p) if !p.probes.is_empty() => {
                let mut records = [GpuProbe::default(); MAX_PROBES];
                let n = p.probes.len().min(MAX_PROBES);
                for (slot, probe) in p.probes.iter().take(MAX_PROBES).enumerate() {
                    let c = probe.center;
                    let h = probe.half_extents;
                    records[slot] = GpuProbe {
                        center_active: [c.x, c.y, c.z, 1.0],
                        box_min_layer: [c.x - h.x, c.y - h.y, c.z - h.z, probe.layer as f32],
                        box_max_intensity: [c.x + h.x, c.y + h.y, c.z + h.z, probe.intensity],
                        params: [probe.rotation, probe.falloff.max(1e-4), p.max_lod, 0.0],
                    };
                }
                self.probe_records.set(records);
                self.probe_count.set(n as u32);
                let ptr = p.array_view as *const wgpu::TextureView as usize;
                if self.probe_bound_ptr != ptr {
                    self.cur_probe_view = p.array_view.clone();
                    self.probe_bound_ptr = ptr;
                    self.rebuild_frame_bind_group();
                }
            }
            _ => {
                self.probe_count.set(0);
                if self.probe_bound_ptr != 0 {
                    self.cur_probe_view = self.probe_fallback_view.clone();
                    self.probe_bound_ptr = 0;
                    self.rebuild_frame_bind_group();
                }
            }
        }
    }

    fn set_capture_mode(&mut self, on: bool) {
        self.capture_mode.set(on);
    }

    fn set_clip_plane(&mut self, plane: Option<[f32; 4]>) {
        self.clip_plane.set(plane.unwrap_or([0.0; 4]));
    }

    fn set_ssao(&mut self, ao: Option<&wgpu::TextureView>) {
        match ao {
            Some(view) => {
                self.ssao_has.set(true);
                let ptr = view as *const wgpu::TextureView as usize;
                if self.ao_bound_ptr != ptr {
                    self.cur_ao_view = view.clone();
                    self.ao_bound_ptr = ptr;
                    self.rebuild_frame_bind_group();
                }
            }
            None => {
                self.ssao_has.set(false);
                if self.ao_bound_ptr != 0 {
                    self.cur_ao_view = self.ao_fallback_view.clone();
                    self.ao_bound_ptr = 0;
                    self.rebuild_frame_bind_group();
                }
            }
        }
    }

    fn set_clustered_buffers(
        &mut self,
        lights: &wgpu::Buffer,
        grid: &wgpu::Buffer,
        index: &wgpu::Buffer,
        force_rebind: bool,
    ) {
        if !self.clustered {
            return;
        }
        // Rebind on the first frame (placeholders -> real buffers) and whenever the
        // light buffer was reallocated (its handle changed). The grid/index buffers
        // are fixed-size, so only `force_rebind` (from a light-buffer grow) matters
        // after the initial bind.
        if force_rebind || !self.clustered_bound {
            self.clustered_lights_buf = lights.clone();
            self.cluster_grid_buf = grid.clone();
            self.cluster_index_buf = index.clone();
            self.clustered_bound = true;
            self.rebuild_frame_bind_group();
        }
    }

    fn flush(&mut self) {
        let ctxt = Context::get();

        self.object_uniform_buffer.flush();

        // Recreate bind group if buffer grew
        if self.object_uniform_buffer.capacity() != self.object_bind_group_capacity {
            self.object_bind_group = Some(ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dynamic_object_bind_group"),
                layout: &self.object_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: self.object_uniform_buffer.buffer(),
                        offset: 0,
                        size: std::num::NonZeroU64::new(self.object_uniform_buffer.aligned_size()),
                    }),
                }],
            }));
            self.object_bind_group_capacity = self.object_uniform_buffer.capacity();
        }
    }

    fn render(
        &mut self,
        _pass: usize,
        _transform: Pose3,
        _scale: Vec3,
        _camera: &mut dyn Camera3d,
        _lights: &LightCollection,
        data: &ObjectData3d,
        mesh: &mut GpuMesh3d,
        instances: &mut InstancesBuffer3d,
        gpu_data: &mut dyn GpuData,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext,
    ) {
        let ctxt = Context::get();

        // A reflector surface is excluded from every reflection capture: it would
        // otherwise sample its own (currently-written) target — an illegal usage —
        // and a mirror doesn't render other mirrors (v1). Its own surface is thus
        // absent from its reflection too (the floor doesn't reflect itself).
        if self.capture_mode.get() && data.reflector().is_some() {
            return;
        }

        // Order-independent transparency phase split: opaque surfaces (and all
        // wireframe/point overlays) draw in the opaque phase; surfaces whose color
        // is translucent draw in the OIT transparent phase. Transparency is keyed
        // off the object color's alpha (per-instance alpha uses this classification
        // too).
        let transparent = data.alpha_mode().is_transparent(data.color().a);
        let render_surface = data.surface_rendering_active()
            && match context.phase {
                // The prepass rasterizes opaque surfaces only (for SSAO geometry).
                crate::resource::RenderPhase::Prepass => !transparent,
                crate::resource::RenderPhase::Opaque => !transparent,
                crate::resource::RenderPhase::Transparent => transparent,
            };
        let in_opaque_phase = context.phase == crate::resource::RenderPhase::Opaque;
        let render_wireframe = in_opaque_phase && data.lines_width() > 0.0;
        let render_points = in_opaque_phase && data.points_size() > 0.0;

        // Nothing to render
        if !render_surface && !render_wireframe && !render_points {
            return;
        }

        // Downcast gpu_data to our specific type
        let gpu_data = gpu_data
            .as_any_mut()
            .downcast_mut::<ObjectMaterialGpuData>()
            .expect("ObjectMaterial requires ObjectMaterialGpuData");

        // Get the pre-computed object uniform offset from prepare() phase
        let object_offset = gpu_data
            .object_uniform_offset
            .expect("prepare() must be called before render()");

        // Load instance data directly to GPU without conversion
        let num_instances = instances.len();
        instances.positions.load_to_gpu();
        instances.colors.load_to_gpu();
        instances.deformations.load_to_gpu();

        // Ensure mesh buffers are on GPU
        mesh.coords().write().unwrap().load_to_gpu();
        mesh.uvs().write().unwrap().load_to_gpu();
        mesh.normals().write().unwrap().load_to_gpu();
        mesh.faces().write().unwrap().load_to_gpu();

        let coords_buffer = mesh.coords().read().unwrap();
        let uvs_buffer = mesh.uvs().read().unwrap();
        let normals_buffer = mesh.normals().read().unwrap();
        let faces_buffer = mesh.faces().read().unwrap();

        let coords_buf = match coords_buffer.buffer() {
            Some(b) => b,
            None => return,
        };
        let uvs_buf = match uvs_buffer.buffer() {
            Some(b) => b,
            None => return,
        };
        let normals_buf = match normals_buffer.buffer() {
            Some(b) => b,
            None => return,
        };
        let faces_buf = match faces_buffer.buffer() {
            Some(b) => b,
            None => return,
        };

        // Get instance buffers
        let inst_positions_buf = match instances.positions.buffer() {
            Some(b) => b,
            None => return,
        };
        let inst_colors_buf = match instances.colors.buffer() {
            Some(b) => b,
            None => return,
        };
        let inst_deformations_buf = match instances.deformations.buffer() {
            Some(b) => b,
            None => return,
        };

        // Cache the combined material-texture bind group (albedo + PBR maps),
        // rebuilding it whenever any of the source textures change.
        let texture_ptr = std::sync::Arc::as_ptr(data.texture()) as usize;
        let normal_map = data.normal_map().unwrap_or(&self.default_normal_map);
        let metallic_roughness_map = data
            .metallic_roughness_map()
            .unwrap_or(&self.default_metallic_roughness_map);
        let ao_map = data.ao_map().unwrap_or(&self.default_ao_map);
        let emissive_map = data.emissive_map().unwrap_or(&self.default_emissive_map);
        let height_map = data.height_map().unwrap_or(&self.default_height_map);

        let normal_ptr = std::sync::Arc::as_ptr(normal_map) as usize;
        let mr_ptr = std::sync::Arc::as_ptr(metallic_roughness_map) as usize;
        let ao_ptr = std::sync::Arc::as_ptr(ao_map) as usize;
        let emissive_ptr = std::sync::Arc::as_ptr(emissive_map) as usize;
        let height_ptr = std::sync::Arc::as_ptr(height_map) as usize;

        // Per-object planar reflection (binding 12). During capture, bind the 1x1
        // fallback (reflections aren't sampled then, and binding a reflector's own
        // target while it's the render target would be an illegal usage) — note
        // reflector objects are skipped entirely during capture (see below), so this
        // fallback only applies to non-reflector objects. In the main pass, a
        // reflector object binds its own target.
        let (reflection_view, reflection_gen) = if self.capture_mode.get() {
            (&self.default_emissive_map.view, 0)
        } else {
            match data.reflector() {
                Some(r) => (r.color_view(), r.generation()),
                None => (&self.default_emissive_map.view, 0),
            }
        };
        let reflection_ptr = reflection_view as *const wgpu::TextureView as usize;

        let textures_changed = gpu_data.texture_bind_group.is_none()
            || gpu_data.cached_texture_ptr != texture_ptr
            || gpu_data.cached_normal_map_ptr != normal_ptr
            || gpu_data.cached_metallic_roughness_map_ptr != mr_ptr
            || gpu_data.cached_ao_map_ptr != ao_ptr
            || gpu_data.cached_emissive_map_ptr != emissive_ptr
            || gpu_data.cached_height_map_ptr != height_ptr
            || gpu_data.cached_reflection_ptr != reflection_ptr
            || gpu_data.cached_reflection_gen != reflection_gen;

        if textures_changed {
            gpu_data.texture_bind_group = Some(self.create_texture_bind_group(
                data.texture(),
                normal_map,
                metallic_roughness_map,
                ao_map,
                emissive_map,
                height_map,
                reflection_view,
            ));
            gpu_data.cached_texture_ptr = texture_ptr;
            gpu_data.cached_normal_map_ptr = normal_ptr;
            gpu_data.cached_metallic_roughness_map_ptr = mr_ptr;
            gpu_data.cached_ao_map_ptr = ao_ptr;
            gpu_data.cached_emissive_map_ptr = emissive_ptr;
            gpu_data.cached_height_map_ptr = height_ptr;
            gpu_data.cached_reflection_ptr = reflection_ptr;
            gpu_data.cached_reflection_gen = reflection_gen;
        }

        // Render surface (filled triangles)
        if render_surface {
            let cull = data.backface_culling_enabled() && !context.force_no_cull;

            // A skinned/morphed object uses the deform pipeline only when the deform
            // pipelines exist (native) and the object's deform bind group was built
            // this frame (in `SceneNode3d::update_deformations`); otherwise it renders
            // through the plain path (base rest shape).
            let use_deform = self.deform.is_some() && data.deform_bind_group().is_some();

            let texture_bind_group = gpu_data.texture_bind_group.as_ref().unwrap();
            let object_bind_group = self.object_bind_group.as_ref().unwrap();

            // Select pipeline: deformed vs. plain, OIT (transparent phase) vs. opaque,
            // and cull vs. no-cull per the object's backface-culling setting.
            let pipeline = if use_deform {
                let dr = self.deform.as_ref().unwrap();
                match (context.phase, cull) {
                    (crate::resource::RenderPhase::Prepass, _) => &dr.prepass_pipeline,
                    (crate::resource::RenderPhase::Transparent, true) => &dr.oit_pipeline_cull,
                    (crate::resource::RenderPhase::Transparent, false) => &dr.oit_pipeline_no_cull,
                    (crate::resource::RenderPhase::Opaque, true) => &dr.pipeline_cull,
                    (crate::resource::RenderPhase::Opaque, false) => &dr.pipeline_no_cull,
                }
                .get(context.sample_count)
            } else {
                match (context.phase, cull) {
                    (crate::resource::RenderPhase::Prepass, _) => &self.prepass_pipeline,
                    (crate::resource::RenderPhase::Transparent, true) => &self.oit_pipeline_cull,
                    (crate::resource::RenderPhase::Transparent, false) => {
                        &self.oit_pipeline_no_cull
                    }
                    (crate::resource::RenderPhase::Opaque, true) => &self.pipeline_cull,
                    (crate::resource::RenderPhase::Opaque, false) => &self.pipeline_no_cull,
                }
                .get(context.sample_count)
            };
            render_pass.set_pipeline(&pipeline);
            render_pass.set_bind_group(0, &self.frame_bind_group, &[]);
            // Use dynamic offset for object uniforms!
            render_pass.set_bind_group(1, object_bind_group, &[object_offset]);
            // Group 2: combined material textures (albedo + PBR maps).
            render_pass.set_bind_group(2, texture_bind_group, &[]);
            // Group 3: shadow atlas + comparison sampler + shadow uniforms. Use the
            // window's shadow mapper bind group when present, else the neutral one.
            let shadow_bind_group = context
                .shadow_bind_group
                .as_ref()
                .unwrap_or(&self.default_shadow_bind_group);
            render_pass.set_bind_group(3, shadow_bind_group, &[]);
            // Group 4: per-object deform data (joint palette + skin streams + morph
            // deltas + control uniform), deform pipeline only.
            if use_deform {
                render_pass.set_bind_group(4, data.deform_bind_group().unwrap(), &[]);
            }

            // Set vertex buffers for mesh data. The deform pipeline uses the same
            // layout — deform data is read from group-4 storage buffers by index.
            render_pass.set_vertex_buffer(0, coords_buf.slice(..));
            render_pass.set_vertex_buffer(1, uvs_buf.slice(..));
            render_pass.set_vertex_buffer(2, normals_buf.slice(..));

            // Set instance buffers directly (no per-frame conversion needed)
            render_pass.set_vertex_buffer(3, inst_positions_buf.slice(..));
            render_pass.set_vertex_buffer(4, inst_colors_buf.slice(..));
            render_pass.set_vertex_buffer(5, inst_deformations_buf.slice(..));

            render_pass.set_index_buffer(faces_buf.slice(..), VERTEX_INDEX_FORMAT);

            render_pass.draw_indexed(0..mesh.num_indices(), 0, 0..num_instances as u32);
        }

        // Render wireframe (thick lines using polyline technique)
        if render_wireframe {
            // Build wireframe edges from mesh if needed
            // Use a simple hash of the faces buffer length as a cache key
            let faces_len = mesh.faces().read().unwrap().len();
            let faces_hash = faces_len as u64;

            if gpu_data.wireframe_edges.is_none()
                || gpu_data.wireframe_edges_mesh_hash != faces_hash
            {
                let coords_guard = mesh.coords().read().unwrap();
                let faces_guard = mesh.faces().read().unwrap();

                if let (Some(coords), Some(faces)) = (coords_guard.data(), faces_guard.data()) {
                    let mut edges = Vec::with_capacity(faces.len() * 3);
                    for face in faces.iter() {
                        let idx_a = face[0] as usize;
                        let idx_b = face[1] as usize;
                        let idx_c = face[2] as usize;

                        if idx_a < coords.len() && idx_b < coords.len() && idx_c < coords.len() {
                            edges.push((coords[idx_a], coords[idx_b]));
                            edges.push((coords[idx_b], coords[idx_c]));
                            edges.push((coords[idx_c], coords[idx_a]));
                        }
                    }
                    gpu_data.wireframe_edges = Some(edges);
                    gpu_data.wireframe_edges_mesh_hash = faces_hash;
                    // Invalidate model bind group since edges changed
                    gpu_data.wireframe_model_bind_group = None;
                }
            }

            // Get edges info and convert to GPU format first
            let (num_edges, gpu_edges) = {
                let edges = match &gpu_data.wireframe_edges {
                    Some(e) => e,
                    None => return,
                };
                let num = edges.len();
                if num == 0 {
                    return;
                }
                let gpu_e: Vec<GpuEdge> = edges
                    .iter()
                    .map(|(a, b)| GpuEdge {
                        point_a: (*a).into(),
                        _pad_a: 0.0,
                        point_b: (*b).into(),
                        _pad_b: 0.0,
                    })
                    .collect();
                (num, gpu_e)
            };

            // Now we can safely mutate gpu_data since edges borrow is done
            {
                // Load wireframe instance buffers to GPU
                instances.lines_colors.load_to_gpu();
                instances.lines_widths.load_to_gpu();

                let inst_lines_colors_buf = match instances.lines_colors.buffer() {
                    Some(b) => b,
                    None => return,
                };
                let inst_lines_widths_buf = match instances.lines_widths.buffer() {
                    Some(b) => b,
                    None => return,
                };

                // Ensure edge buffer capacity and upload edges (geometry data)
                gpu_data.ensure_edge_buffer_capacity(num_edges);

                ctxt.write_buffer(
                    &gpu_data.wireframe_edge_buffer,
                    0,
                    bytemuck::cast_slice(&gpu_edges),
                );

                // Update num_edges in model uniforms if it changed from prepare()
                if gpu_data.wireframe_model_uniforms.num_edges != num_edges as u32 {
                    gpu_data.wireframe_model_uniforms.num_edges = num_edges as u32;
                    ctxt.write_buffer(
                        &gpu_data.wireframe_model_uniform_buffer,
                        0,
                        bytemuck::bytes_of(&gpu_data.wireframe_model_uniforms),
                    );
                }

                // Get or create wireframe model bind group (view bind group is shared)
                if gpu_data.wireframe_model_bind_group.is_none() {
                    let edge_size = (num_edges * std::mem::size_of::<GpuEdge>()) as u64;
                    gpu_data.wireframe_model_bind_group =
                        Some(self.create_wireframe_model_bind_group(
                            &gpu_data.wireframe_model_uniform_buffer,
                            &gpu_data.wireframe_edge_buffer,
                            edge_size,
                        ));
                }

                let wireframe_model_bind_group =
                    gpu_data.wireframe_model_bind_group.as_ref().unwrap();

                let wireframe_pipeline = self.wireframe_pipeline.get(context.sample_count);
                render_pass.set_pipeline(&wireframe_pipeline);
                // Use shared view bind group (written once per frame)
                render_pass.set_bind_group(0, &self.wireframe_view_bind_group, &[]);
                render_pass.set_bind_group(1, wireframe_model_bind_group, &[]);

                // Set instance vertex buffers (5 total: positions, colors, deformations, lines_colors, lines_widths)
                render_pass.set_vertex_buffer(0, inst_positions_buf.slice(..));
                render_pass.set_vertex_buffer(1, inst_colors_buf.slice(..));
                render_pass.set_vertex_buffer(2, inst_deformations_buf.slice(..)); // Contains all 3 columns
                render_pass.set_vertex_buffer(3, inst_lines_colors_buf.slice(..));
                render_pass.set_vertex_buffer(4, inst_lines_widths_buf.slice(..));

                // Draw: 6 vertices per edge (computed from vertex_index), num_instances instances
                let num_vertices = (num_edges * 6) as u32;
                render_pass.draw(0..num_vertices, 0..num_instances as u32);
            }
        }

        // Render points
        if render_points {
            // Build vertex cache if needed (using mesh coords hash)
            let coords_hash = {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                let coords = mesh.coords().read().unwrap();
                coords.len().hash(&mut hasher);
                // Simple hash based on length - vertices rarely change
                hasher.finish()
            };

            if gpu_data.points_vertices.is_none()
                || gpu_data.points_vertices_mesh_hash != coords_hash
            {
                // Rebuild vertex cache from mesh coords
                let coords_guard = mesh.coords().read().unwrap();
                if let Some(coords) = coords_guard.data() {
                    gpu_data.points_vertices = Some(coords.to_vec());
                    gpu_data.points_vertices_mesh_hash = coords_hash;
                    // Invalidate model bind group since vertices changed
                    gpu_data.points_model_bind_group = None;
                }
            }

            // Get vertices info and convert to GPU format
            let (num_vertices, gpu_vertices) = {
                let vertices = match &gpu_data.points_vertices {
                    Some(v) => v,
                    None => return,
                };
                let num = vertices.len();
                if num == 0 {
                    return;
                }
                let gpu_v: Vec<GpuVertex> = vertices
                    .iter()
                    .map(|p| GpuVertex {
                        position: (*p).into(),
                        _pad: 0.0,
                    })
                    .collect();
                (num, gpu_v)
            };

            // Now we can safely mutate gpu_data since vertices borrow is done
            {
                // Load point instance buffers to GPU
                instances.points_colors.load_to_gpu();
                instances.points_sizes.load_to_gpu();

                let inst_points_colors_buf = match instances.points_colors.buffer() {
                    Some(b) => b,
                    None => return,
                };
                let inst_points_sizes_buf = match instances.points_sizes.buffer() {
                    Some(b) => b,
                    None => return,
                };

                // Ensure vertex buffer capacity and upload vertices (geometry data)
                gpu_data.ensure_vertex_buffer_capacity(num_vertices);

                ctxt.write_buffer(
                    &gpu_data.points_vertex_buffer,
                    0,
                    bytemuck::cast_slice(&gpu_vertices),
                );

                // Update num_vertices in model uniforms if it changed from prepare()
                if gpu_data.points_model_uniforms.num_vertices != num_vertices as u32 {
                    gpu_data.points_model_uniforms.num_vertices = num_vertices as u32;
                    ctxt.write_buffer(
                        &gpu_data.points_model_uniform_buffer,
                        0,
                        bytemuck::bytes_of(&gpu_data.points_model_uniforms),
                    );
                }

                // Get or create points model bind group (view bind group is shared)
                if gpu_data.points_model_bind_group.is_none() {
                    let vertex_size = (num_vertices * std::mem::size_of::<GpuVertex>()) as u64;
                    gpu_data.points_model_bind_group = Some(self.create_points_model_bind_group(
                        &gpu_data.points_model_uniform_buffer,
                        &gpu_data.points_vertex_buffer,
                        vertex_size,
                    ));
                }

                let points_model_bind_group = gpu_data.points_model_bind_group.as_ref().unwrap();

                let points_pipeline = self.points_pipeline.get(context.sample_count);
                render_pass.set_pipeline(&points_pipeline);
                // Use shared view bind group (written once per frame)
                render_pass.set_bind_group(0, &self.points_view_bind_group, &[]);
                render_pass.set_bind_group(1, points_model_bind_group, &[]);

                // Set instance vertex buffers (5 total: positions, colors, deformations, points_colors, points_sizes)
                render_pass.set_vertex_buffer(0, inst_positions_buf.slice(..));
                render_pass.set_vertex_buffer(1, inst_colors_buf.slice(..));
                render_pass.set_vertex_buffer(2, inst_deformations_buf.slice(..)); // Contains all 3 columns
                render_pass.set_vertex_buffer(3, inst_points_colors_buf.slice(..));
                render_pass.set_vertex_buffer(4, inst_points_sizes_buf.slice(..));

                // Draw: 6 vertices per point (computed from vertex_index), num_instances instances
                let num_draw_vertices = (num_vertices * 6) as u32;
                render_pass.draw(0..num_draw_vertices, 0..num_instances as u32);
            }
        }
    }
}

/// Vertex shader of the default object material.
pub static OBJECT_VERTEX_SRC: &str = include_str!("default.wgsl");
/// Fragment shader of the default object material.
pub static OBJECT_FRAGMENT_SRC: &str = include_str!("default.wgsl");
