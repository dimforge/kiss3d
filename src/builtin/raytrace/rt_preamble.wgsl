// Shared declarations for the path tracer: data structures and bind group 0/1
// bindings common to both the compute (BVH) and hardware (ray-query) backends.
//
// The full compute module is assembled at runtime as:
//     rt_preamble.wgsl  +  rt_intersect_*.wgsl  +  rt_kernel.wgsl
// so every symbol declared here is visible to the snippets that follow.

struct RtVertex {
    position: vec3<f32>,
    normal: vec3<f32>,
};

struct RtTriangle {
    v0: u32,
    v1: u32,
    v2: u32,
    material_id: u32,
};

struct RtMaterial {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    metallic: f32,
    roughness: f32,
    pad: vec2<f32>,
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
    pad: vec2<f32>,
};

struct FrameUniforms {
    inv_view_proj: mat4x4<f32>,
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
};

// Result of a closest-hit query. `valid == false` means the ray escaped.
struct Hit {
    valid: bool,
    t: f32,
    normal: vec3<f32>,
    material_id: u32,
};

const PI: f32 = 3.14159265359;
const EPS: f32 = 1.0e-4;
const T_MAX: f32 = 1.0e30;

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<storage, read_write> accum: array<vec4<f32>>;

@group(1) @binding(0) var<storage, read> vertices: array<RtVertex>;
@group(1) @binding(1) var<storage, read> triangles: array<RtTriangle>;
@group(1) @binding(2) var<storage, read> materials: array<RtMaterial>;
@group(1) @binding(3) var<storage, read> lights: array<RtLight>;
