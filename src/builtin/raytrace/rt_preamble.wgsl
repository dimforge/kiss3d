// Shared declarations for the path tracer: data structures and bind group 0/1
// bindings common to both the compute (BVH) and hardware (ray-query) backends.
//
// The full compute module is assembled at runtime as:
//     rt_preamble.wgsl  +  rt_intersect_*.wgsl  +  rt_kernel.wgsl
// so every symbol declared here is visible to the snippets that follow.

struct RtVertex {
    position: vec3<f32>,
    // `uv` is packed into the w components of position/normal to keep the
    // std430 stride at 32 bytes (two vec4s): position.w = u, normal.w = v.
    u: f32,
    normal: vec3<f32>,
    v: f32,
};

struct RtTriangle {
    v0: u32,
    v1: u32,
    v2: u32,
    material_id: u32,
};

// BSDF/material model tags matching the Rust `RtBsdf` constants.
const BSDF_OPAQUE: u32 = 0u;      // metallic-roughness PBR (default)
const BSDF_GLASS: u32 = 1u;       // smooth/rough dielectric (refraction)
const BSDF_METAL: u32 = 2u;       // pure conductor (reflection only)
const BSDF_EMISSIVE: u32 = 3u;    // emitter (treated as opaque shading-wise)

// Unified Disney-style material, std430 96-byte layout (6 x vec4).
struct RtMaterial {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    // metallic, roughness, index of refraction, transmission factor.
    metallic: f32,
    roughness: f32,
    ior: f32,
    transmission: f32,
    // specular tint (rgb) + bsdf type packed in w (as a bitcast u32).
    specular_tint: vec3<f32>,
    bsdf_type: u32,
    // subsurface factor, subsurface radius, then two reserved scalars.
    subsurface: f32,
    subsurface_radius: f32,
    pad0: f32,
    pad1: f32,
    // Texture-array layer indices (-1 = none): albedo, normal, MR, emissive.
    albedo_tex: i32,
    normal_tex: i32,
    mr_tex: i32,
    emissive_tex: i32,
};

struct RtLight {
    position: vec3<f32>,
    light_type: u32,
    direction: vec3<f32>,
    intensity: f32,
    color: vec3<f32>,
    attenuation_radius: f32,
    inner_cone_cos: f32,
    outer_cone_cos: f32,
    // Sphere radius for soft shadows (0 = delta point/spot light).
    radius: f32,
    pad: f32,
};

struct FrameUniforms {
    inv_view_proj: mat4x4<f32>,
    // Environment rotation about the Y axis (cos, sin) and its luminance scale.
    env_rotation: vec4<f32>,
    cam_eye: vec3<f32>,
    width: u32,
    height: u32,
    sample_index: u32,
    num_triangles: u32,
    num_lights: u32,
    ambient: f32,
    max_bounces: u32,
    seed: u32,
    samples_per_frame: u32,
    num_emitters: u32,
    // Thin-lens camera: lens radius (0 = pinhole) and focus distance.
    lens_radius: f32,
    focus_distance: f32,
    // 1 if an environment map is bound, else 0 (use the background color).
    has_env: u32,
    // Background color shown where a directly-seen ray escapes the scene when no
    // environment map is bound. Cosmetic only — it does NOT light the scene (unlike
    // an HDRI environment), so it never tints objects via indirect bounces.
    background: vec4<f32>,
};

// Result of a closest-hit query. `valid == false` means the ray escaped.
// All fields are in WORLD space and filled identically by both intersection
// backends, so the kernel is backend-agnostic.
struct Hit {
    valid: bool,
    t: f32,
    normal: vec3<f32>,
    // Geometric (face) normal, used to decide front/back face for refraction.
    geom_normal: vec3<f32>,
    material_id: u32,
    // World-space area of the hit triangle (for emitter-area MIS).
    tri_area: f32,
    // Interpolated texture coordinates at the hit point.
    uv: vec2<f32>,
};

const PI: f32 = 3.14159265359;
const EPS: f32 = 1.0e-4;
const T_MAX: f32 = 1.0e30;

// One emissive triangle baked into WORLD space (positions + emission radiance),
// so emitter sampling is independent of how each backend lays out its geometry
// (the compute backend stores mesh-local vertices; the hardware backend stores
// world-space vertices).
struct RtEmitter {
    p0: vec3<f32>,
    pad0: f32,
    p1: vec3<f32>,
    pad1: f32,
    p2: vec3<f32>,
    pad2: f32,
    emission: vec3<f32>,
    pad3: f32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
// One buffer holding three contiguous `width*height` regions: region 0 = radiance
// running mean, region 1 = first-hit albedo guide, region 2 = first-hit normal
// guide. Region `k` of pixel `p` is at `pixels[k * frame.width * frame.height + p]`.
// Packed into one binding to stay within WebGPU's 8 storage buffers per stage.
@group(0) @binding(1) var<storage, read_write> pixels: array<vec4<f32>>;

@group(1) @binding(0) var<storage, read> vertices: array<RtVertex>;
@group(1) @binding(1) var<storage, read> triangles: array<RtTriangle>;
@group(1) @binding(2) var<storage, read> materials: array<RtMaterial>;
@group(1) @binding(3) var<storage, read> lights: array<RtLight>;
// Intersection backend uses binding 4 (BVH buffer or acceleration structure).
@group(1) @binding(5) var<storage, read> emitters: array<RtEmitter>;
// Object textures packed into a 2D-array (per-material layer index); a 1x1
// fallback layer is always present so the binding is valid even with no maps.
@group(1) @binding(6) var tex_array: texture_2d_array<f32>;
@group(1) @binding(7) var tex_sampler: sampler;
// Equirectangular HDR environment map + its importance-sampling marginal CDF.
@group(1) @binding(8) var env_tex: texture_2d<f32>;
@group(1) @binding(9) var env_sampler: sampler;
