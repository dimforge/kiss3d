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
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

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
    /// Light-layer bitmask (lighting channels). A fragment skips this light when
    /// `(object.light_layers & layers) == 0`. `u32::MAX` affects every object.
    layers: u32,
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
            layers: u32::MAX,
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
            layers: light.layers,
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
    // Light-layer bitmask (lighting channels): a light affects this object only
    // when `(light_layers & light.layers) != 0`.
    light_layers: u32,
    // Index of refraction for refractive transmission (glass). 1.5 = window glass.
    ior: f32,
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
    // Refractive transmission (glass) volume attenuation color (rgb); a unused.
    attenuation_color: [f32; 4],
    // Refractive transmission volume params: (thickness, attenuation_distance,
    // unused, unused). attenuation_distance < 0 means "infinite" (no tint).
    volume: [f32; 4],
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
    // === Specialized shader variants (WESL conditional compilation) ===
    /// Pipeline layout shared by every plain (non-deform) surface/prepass variant.
    pipeline_layout: wgpu::PipelineLayout,
    /// Pipeline layout for the deform variants (adds group 4); `None` when the
    /// backend has no spare bind group (web / WebGL2) — deform then falls back to the
    /// plain path.
    deform_pipeline_layout: Option<wgpu::PipelineLayout>,
    /// Opaque-surface pipeline builder: `(layout, module, _skinned, cull, label, samples)`.
    build_opaque: SurfacePipelineBuilder,
    /// Weighted-blended OIT pipeline builder (same signature as `build_opaque`).
    build_oit: SurfacePipelineBuilder,
    /// Depth + view-position prepass pipeline builder: `(layout, module, _skinned, samples)`.
    build_prepass: PrepassPipelineBuilder,
    /// WESL-compiled shader modules, keyed by feature mask (lazily compiled, cached).
    shader_modules: RefCell<HashMap<ShaderFeatures, Rc<wgpu::ShaderModule>>>,
    /// Specialized surface/prepass pipelines, keyed by `(features, sample_count, kind)`.
    surface_pipelines:
        RefCell<HashMap<(ShaderFeatures, u32, PipelineKind), Rc<wgpu::RenderPipeline>>>,
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
    /// Whether distance fog is active this frame (fog mode != off). Drives the `fog`
    /// shader feature; set in `prepare()` from the scene's fog params.
    fog_has: Cell<bool>,
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
    // === Transmission background (group 0 binding 8/9). ===
    /// Trilinear+clamp sampler for the refraction background mip chain.
    bg_sampler: wgpu::Sampler,
    /// Currently-bound background view (clone; defaults to the IBL 1x1 fallback).
    cur_bg_view: wgpu::TextureView,
    /// Identity of the bound background view (`0` = fallback) to avoid rebuilds.
    bg_bound_ptr: usize,
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
}

/// Builds an opaque-surface or OIT pipeline from a compiled module:
/// `(pipeline_layout, shader_module, _skinned, cull_mode, label, sample_count)`.
/// Captures nothing; the deform variant differs only in the module + layout passed.
type SurfacePipelineBuilder = Rc<
    dyn Fn(
        &wgpu::PipelineLayout,
        &wgpu::ShaderModule,
        bool,
        Option<wgpu::Face>,
        &'static str,
        u32,
    ) -> wgpu::RenderPipeline,
>;

/// Builds the depth + view-position prepass pipeline:
/// `(pipeline_layout, shader_module, _skinned, sample_count)`.
type PrepassPipelineBuilder =
    Rc<dyn Fn(&wgpu::PipelineLayout, &wgpu::ShaderModule, bool, u32) -> wgpu::RenderPipeline>;

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

/// Conditional-compilation features for the object shader. Each bit maps to a WESL
/// `@if(...)` feature flag in `default.wgsl`; a per-object/per-frame mask selects a
/// specialized shader variant so the features an object/frame doesn't use — and the
/// registers and bindings they need — are stripped out entirely. The vertex/binding
/// layout is identical across variants (unused bindings simply strip away), so all
/// variants share the same pipeline layout and bind groups. See `compile_object_wgsl`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub(crate) struct ShaderFeatures(u32);

impl ShaderFeatures {
    // Structural / capability axes (resolved against backend support).
    const DEFORM: u32 = 1 << 0;
    const CLUSTERED: u32 = 1 << 1;
    // Global / per-frame.
    const SHADOWS: u32 = 1 << 2;
    const IBL: u32 = 1 << 3;
    const PROBES: u32 = 1 << 4;
    const FOG: u32 = 1 << 5;
    const SSAO: u32 = 1 << 6;
    // Per-object material.
    const NORMAL_MAP: u32 = 1 << 7;
    const MR_MAP: u32 = 1 << 8;
    const AO_MAP: u32 = 1 << 9;
    const EMISSIVE_MAP: u32 = 1 << 10;
    const PARALLAX: u32 = 1 << 11;
    const CLEARCOAT: u32 = 1 << 12;
    const ANISOTROPY: u32 = 1 << 13;
    const TRANSMISSION: u32 = 1 << 14;
    const REFLECTOR: u32 = 1 << 15;

    /// `(WESL feature name, bit)` — names MUST match the `@if(...)` flags in
    /// `default.wgsl`.
    const TABLE: [(&'static str, u32); 16] = [
        ("deform", Self::DEFORM),
        ("clustered", Self::CLUSTERED),
        ("shadows", Self::SHADOWS),
        ("ibl", Self::IBL),
        ("probes", Self::PROBES),
        ("fog", Self::FOG),
        ("ssao", Self::SSAO),
        ("normal_map", Self::NORMAL_MAP),
        ("mr_map", Self::MR_MAP),
        ("ao_map", Self::AO_MAP),
        ("emissive_map", Self::EMISSIVE_MAP),
        ("parallax", Self::PARALLAX),
        ("clearcoat", Self::CLEARCOAT),
        ("anisotropy", Self::ANISOTROPY),
        ("transmission", Self::TRANSMISSION),
        ("reflector", Self::REFLECTOR),
    ];

    #[inline]
    fn has(self, bit: u32) -> bool {
        self.0 & bit != 0
    }

    #[inline]
    fn with(mut self, bit: u32, on: bool) -> Self {
        if on {
            self.0 |= bit;
        } else {
            self.0 &= !bit;
        }
        self
    }

    /// The feature subset that affects the prepass: only the vertex stage (`deform`)
    /// matters — `fs_prepass` ignores every shading feature — so collapsing to this
    /// keeps the prepass to a single module per deform-ness.
    #[inline]
    fn prepass_key(self) -> Self {
        ShaderFeatures(self.0 & Self::DEFORM)
    }
}

/// Compiles `default.wgsl` to specialized WGSL for `features`, via WESL conditional
/// translation. Dead-code elimination (WESL "strip") then removes the now-unreachable
/// helpers and their bindings, so the output is both leaner (fewer live registers →
/// higher GPU occupancy) and, on the non-`clustered`/non-`deform` variants, free of
/// storage bindings (WebGL2-safe).
fn compile_object_wgsl(features: ShaderFeatures) -> String {
    let feats: Vec<(&str, bool)> = ShaderFeatures::TABLE
        .iter()
        .map(|(name, bit)| (*name, features.has(*bit)))
        .collect();
    crate::builtin::compile_wesl(
        &[
            ("package::default", include_str!("default.wgsl")),
            ("package::pbr_env", crate::builtin::PBR_ENV_WESL),
        ],
        "package::default",
        &feats,
    )
}

/// Which surface pipeline to build from a compiled module (the opaque/OIT cull and
/// no-cull variants, plus the shared depth/G-buffer prepass).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum PipelineKind {
    OpaqueCull,
    OpaqueNoCull,
    OitCull,
    OitNoCull,
    Prepass,
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
        // Transmission background (binding 8) + its sampler (9): the resolved opaque
        // scene color with a blurred mip chain, sampled by refractive (glass) objects
        // to refract the scene behind them. Always present; only the refraction shader
        // variant reads it (a 1x1 fallback is bound when no glass is in the scene).
        frame_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        frame_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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

        // Sampler for the transmission background (binding 9): clamp + trilinear so
        // the refraction sample can read the blurred mip chain by roughness LOD.
        let bg_sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("object_material_transmission_bg_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
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

        // Shared opaque-surface pipeline builder, parameterized by the pipeline
        // layout and the (WESL-specialized) shader module. The `skinned` flag is
        // vestigial — deform data is read from group-4 storage by index, so the
        // vertex layout is identical; the deform variant differs only in the module +
        // layout passed. Stored on the material and invoked lazily per
        // `(features, sample_count)` by `surface_pipeline`.
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
        // Deform pipeline layout (native only). The deform bind group is a 5th bind
        // group, which exceeds WebGPU/WebGL2's 4-group cap, so on web (or any adapter
        // without a free group) skinned/morphed objects fall back to the plain path
        // and render in their base rest shape. The actual deform pipelines are built
        // lazily per `(features, sample_count)` by `surface_pipeline`, using this
        // layout and the `deform`-featured module. The deform bind-group layout is the
        // shared one from `builtin::deform`, so the per-object bind group also works
        // in the shadow pipelines.
        #[cfg(not(target_arch = "wasm32"))]
        let deform_pipeline_layout = {
            let deform_bind_group_layout = crate::builtin::deform::deform_bind_group_layout();
            Some(
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
                }),
            )
        };
        #[cfg(target_arch = "wasm32")]
        let deform_pipeline_layout: Option<wgpu::PipelineLayout> = None;

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
        // Transmission background: 1x1 black fallback (reusing the IBL fallback) until
        // a glass-bearing frame hands over the snapshot via `set_transmission_background`.
        frame_group_entries.push(wgpu::BindGroupEntry {
            binding: 8,
            resource: wgpu::BindingResource::TextureView(&ibl_fallback_view),
        });
        frame_group_entries.push(wgpu::BindGroupEntry {
            binding: 9,
            resource: wgpu::BindingResource::Sampler(&bg_sampler),
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
            pipeline_layout,
            deform_pipeline_layout,
            build_opaque,
            build_oit,
            build_prepass,
            shader_modules: RefCell::new(HashMap::new()),
            surface_pipelines: RefCell::new(HashMap::new()),
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
            cur_bg_view: ibl_fallback_view.clone(),
            bg_sampler,
            bg_bound_ptr: 0,
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
            fog_has: Cell::new(false),
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
        }
    }

    /// Returns the WESL-compiled, specialized shader module for `features`, compiling
    /// and caching it on first use. The vertex/binding layout is identical across
    /// variants — unused bindings simply strip away — so every module is compatible
    /// with the shared pipeline layouts.
    fn shader_module(&self, features: ShaderFeatures) -> Rc<wgpu::ShaderModule> {
        if let Some(m) = self.shader_modules.borrow().get(&features) {
            return m.clone();
        }
        let wgsl = compile_object_wgsl(features);
        let module =
            Rc::new(Context::get().create_shader_module(Some("object_material_shader"), &wgsl));
        self.shader_modules
            .borrow_mut()
            .insert(features, module.clone());
        module
    }

    /// Returns the surface/prepass pipeline for `(features, sample_count, kind)`,
    /// building (and compiling its module) on first use and caching it. Picks the
    /// deform pipeline layout when `features` has `DEFORM`, else the plain one.
    fn surface_pipeline(
        &self,
        features: ShaderFeatures,
        sample_count: u32,
        kind: PipelineKind,
    ) -> Rc<wgpu::RenderPipeline> {
        let sample_count = sample_count.max(1);
        let key = (features, sample_count, kind);
        if let Some(p) = self.surface_pipelines.borrow().get(&key) {
            return p.clone();
        }
        let module = self.shader_module(features);
        let layout = if features.has(ShaderFeatures::DEFORM) {
            self.deform_pipeline_layout
                .as_ref()
                .expect("deform pipeline layout (native only) must exist for a deform variant")
        } else {
            &self.pipeline_layout
        };
        let pipeline = match kind {
            PipelineKind::OpaqueCull => (self.build_opaque)(
                layout,
                &module,
                false,
                Some(wgpu::Face::Back),
                "object_material_pipeline_cull",
                sample_count,
            ),
            PipelineKind::OpaqueNoCull => (self.build_opaque)(
                layout,
                &module,
                false,
                None,
                "object_material_pipeline_no_cull",
                sample_count,
            ),
            PipelineKind::OitCull => (self.build_oit)(
                layout,
                &module,
                false,
                Some(wgpu::Face::Back),
                "object_material_oit_pipeline_cull",
                sample_count,
            ),
            PipelineKind::OitNoCull => (self.build_oit)(
                layout,
                &module,
                false,
                None,
                "object_material_oit_pipeline_no_cull",
                sample_count,
            ),
            PipelineKind::Prepass => (self.build_prepass)(layout, &module, false, sample_count),
        };
        let pipeline = Rc::new(pipeline);
        self.surface_pipelines
            .borrow_mut()
            .insert(key, pipeline.clone());
        pipeline
    }

    /// Number of `@if` feature flags the object shader has — so the shader-validity
    /// test can enumerate variants (`bits` in `0..2^FEATURE_COUNT`).
    #[cfg(test)]
    pub(crate) const FEATURE_COUNT: u32 = ShaderFeatures::TABLE.len() as u32;

    /// Test-only: instantiates (compiles the WESL module + creates the real wgpu
    /// shader module and render pipeline for) the object-shader variant whose feature
    /// mask is `bits`, so the shader-validity test can confirm it compiles on the
    /// device. Returns `false` (without building) when the variant needs a
    /// `CLUSTERED`/`DEFORM` binding this material's pipeline layout doesn't carry
    /// (otherwise the layout wouldn't match the shader). Building the opaque pipeline
    /// creates the module, which also validates `fs_oit`/`fs_prepass` (same module).
    /// Any invalid variant surfaces as a wgpu validation error in the caller's error
    /// scope.
    #[cfg(test)]
    pub(crate) fn try_instantiate_variant_for_test(&self, bits: u32, sample_count: u32) -> bool {
        let f = ShaderFeatures(bits);
        if (f.has(ShaderFeatures::CLUSTERED) && !self.clustered)
            || (f.has(ShaderFeatures::DEFORM) && self.deform_pipeline_layout.is_none())
        {
            return false;
        }
        let _ = self.surface_pipeline(f, sample_count, PipelineKind::OpaqueCull);
        true
    }

    /// Computes the specialized shader feature mask for `data` in the current frame.
    /// Per-object features come from the material; global ones (shadows/ibl/probes/
    /// fog/ssao) from the per-frame state; structural ones (deform/clustered) are
    /// resolved against backend capability. Each flag is a *superset* of the shader's
    /// runtime test, so a stripped feature can never skip work the object needs.
    ///
    /// `use_deform` and `shadows_active` are passed from `render()` (they depend on the
    /// object's deform bind group and the bound shadow group, respectively).
    fn object_features(
        &self,
        data: &ObjectData3d,
        use_deform: bool,
        shadows_active: bool,
    ) -> ShaderFeatures {
        let f = ShaderFeatures::default()
            // Structural / capability.
            .with(ShaderFeatures::DEFORM, use_deform)
            .with(ShaderFeatures::CLUSTERED, self.clustered);

        let probes = self.probe_count.get() > 0 && !self.capture_mode.get();
        f.with(ShaderFeatures::SHADOWS, shadows_active)
            .with(ShaderFeatures::IBL, self.ibl_has.get())
            .with(ShaderFeatures::PROBES, probes)
            .with(ShaderFeatures::FOG, self.fog_has.get())
            .with(ShaderFeatures::SSAO, self.ssao_has.get())
            .with(ShaderFeatures::NORMAL_MAP, data.normal_map().is_some())
            .with(
                ShaderFeatures::MR_MAP,
                data.metallic_roughness_map().is_some(),
            )
            .with(ShaderFeatures::AO_MAP, data.ao_map().is_some())
            .with(ShaderFeatures::EMISSIVE_MAP, data.emissive_map().is_some())
            .with(ShaderFeatures::PARALLAX, data.height_map().is_some())
            .with(ShaderFeatures::CLEARCOAT, data.clearcoat() > 0.0)
            .with(ShaderFeatures::ANISOTROPY, data.anisotropy() != 0.0)
            .with(ShaderFeatures::TRANSMISSION, data.transmission() > 0.0)
            .with(ShaderFeatures::REFLECTOR, data.reflector().is_some())
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
        entries.push(wgpu::BindGroupEntry {
            binding: 8,
            resource: wgpu::BindingResource::TextureView(&self.cur_bg_view),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 9,
            resource: wgpu::BindingResource::Sampler(&self.bg_sampler),
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

            // Record whether distance fog is active this frame (mode != off, params.x)
            // so `object_features` can gate the `fog` shader feature.
            self.fog_has.set(lights.fog.params()[0] != 0.0);

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
            light_layers: data.light_layers(),
            ior: data.ior(),
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
            attenuation_color: {
                let a = data.attenuation_color();
                [a.r, a.g, a.b, 1.0]
            },
            volume: {
                // Encode an infinite attenuation distance as a negative sentinel so
                // the shader can branch cheaply (no tint).
                let dist = data.attenuation_distance();
                let encoded = if dist.is_finite() { dist } else { -1.0 };
                [data.thickness(), encoded, 0.0, 0.0]
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
                // Always rebind: the env map is recreated when the skybox is (re)loaded,
                // and a fresh `TextureView` can reuse a just-freed address, so pointer
                // identity can't reliably detect the change (stale-view-after-realloc
                // trap — same as SSAO / probes / the transmission background).
                self.cur_ibl_view = e.view.clone();
                self.cur_ibl_sampler = e.sampler.clone();
                self.ibl_bound_ptr = 1;
                self.rebuild_frame_bind_group();
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
                // Always rebind: the probe array view can be recreated (re-registration
                // / runtime capture) and reuse a just-freed address, so pointer identity
                // can't reliably detect the change (stale-view-after-realloc trap).
                self.cur_probe_view = p.array_view.clone();
                self.probe_bound_ptr = 1;
                self.rebuild_frame_bind_group();
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
            // Always rebind: the SSAO texture is recreated on every window resize, and
            // a fresh `TextureView` can reuse a just-freed address, so pointer identity
            // can't reliably detect the change (would leave a stale/freed view bound —
            // the same "frozen after resize" trap as the transmission background).
            Some(view) => {
                self.ssao_has.set(true);
                self.cur_ao_view = view.clone();
                self.ao_bound_ptr = 1;
                self.rebuild_frame_bind_group();
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

    /// Binds the transmission background (the resolved opaque scene color + its
    /// blurred mip chain) sampled by refractive (glass) objects, or `None` to fall
    /// back to the 1x1 placeholder. Mirrors [`set_ssao`].
    fn set_transmission_background(&mut self, bg: Option<&wgpu::TextureView>) {
        match bg {
            // Always rebind when a background is supplied: the snapshot texture is
            // rebuilt on every resize, and a fresh `TextureView` can reuse a just-freed
            // pointer, so pointer identity can't reliably detect the change. Skipping
            // the rebuild would leave the bind group referencing a stale/freed view
            // (the "frozen glass after resizing" bug). The cost is one bind-group
            // rebuild per frame, only while refractive surfaces are present.
            Some(view) => {
                self.cur_bg_view = view.clone();
                self.bg_bound_ptr = 1;
                self.rebuild_frame_bind_group();
            }
            None => {
                if self.bg_bound_ptr != 0 {
                    self.cur_bg_view = self.ibl_fallback_view.clone();
                    self.bg_bound_ptr = 0;
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
        // Refractive glass draws in its own post-resolve pass (so it can sample the
        // scene behind it), not the opaque/prepass passes — otherwise it would be
        // drawn opaque and double-rendered.
        let glass = data.transmission() > 0.0;
        let render_surface = data.surface_rendering_active()
            && match context.phase {
                // The prepass rasterizes opaque surfaces only (for SSAO geometry +
                // the depth the glass pass tests against — glass stays out of it).
                crate::resource::RenderPhase::Prepass => !transparent && !glass,
                crate::resource::RenderPhase::Opaque => !transparent && !glass,
                crate::resource::RenderPhase::Transparent => transparent,
                crate::resource::RenderPhase::Transmission => glass,
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
            // path exists (native) and the object's deform bind group was built this
            // frame (in `SceneNode3d::update_deformations`); otherwise it renders
            // through the plain path (base rest shape).
            let use_deform =
                self.deform_pipeline_layout.is_some() && data.deform_bind_group().is_some();

            let texture_bind_group = gpu_data.texture_bind_group.as_ref().unwrap();
            let object_bind_group = self.object_bind_group.as_ref().unwrap();

            // Select the specialized pipeline: which entry/targets (opaque vs. OIT vs.
            // prepass, cull vs. no-cull) and which WESL feature variant. Shadows are
            // active this frame iff the window bound a real shadow group (else the
            // neutral one is used and the `shadows` feature strips out).
            let shadows_active = context.shadow_bind_group.is_some();
            let kind = match (context.phase, cull) {
                (crate::resource::RenderPhase::Prepass, _) => PipelineKind::Prepass,
                (crate::resource::RenderPhase::Transparent, true) => PipelineKind::OitCull,
                (crate::resource::RenderPhase::Transparent, false) => PipelineKind::OitNoCull,
                // Glass reuses the opaque pipeline (it writes opaque color/depth);
                // the refraction is computed in-shader by sampling the background.
                (crate::resource::RenderPhase::Transmission, true)
                | (crate::resource::RenderPhase::Opaque, true) => PipelineKind::OpaqueCull,
                (crate::resource::RenderPhase::Transmission, false)
                | (crate::resource::RenderPhase::Opaque, false) => PipelineKind::OpaqueNoCull,
            };
            let mut features = self.object_features(data, use_deform, shadows_active);
            // The prepass ignores all shading features; collapse to the structural key
            // so it stays a single module per deform-ness.
            if kind == PipelineKind::Prepass {
                features = features.prepass_key();
            }
            let pipeline = self.surface_pipeline(features, context.sample_count, kind);
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
