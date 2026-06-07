// Colored-transmittance pass for translucent shadow casters.
//
// Runs after the opaque depth pre-pass, into the matching layer of the colored
// transmittance atlas. Transparent occluders are rasterized with multiplicative
// blending (the atlas is cleared to white = 1) and depth-tested (read-only)
// against the opaque depth map, so only occluders *between* the light and the
// nearest opaque surface tint the light. Multiplicative transmittance commutes,
// so overlapping translucent occluders compose order-independently.
//
// The vertex transform mirrors `shadow_depth.wgsl` (and `default.wgsl`) so the
// transmittance geometry matches the lit/occluding geometry exactly.

// Group 0: per-view light-space matrix (dynamic offset, one slot per atlas view).
struct ViewUniforms {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> view: ViewUniforms;

// Group 1: per-object model transform + base color (dynamic offset per object).
// Layout matches `ShadowModelUniforms` in builtin/shadow.rs.
struct ModelUniforms {
    transform: mat4x4<f32>,
    scale: mat3x3<f32>,
    color: vec4<f32>,
}

@group(1) @binding(0)
var<uniform> model: ModelUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct InstanceInput {
    @location(1) inst_tra: vec3<f32>,
    @location(2) inst_def_0: vec3<f32>,
    @location(3) inst_def_1: vec3<f32>,
    @location(4) inst_def_2: vec3<f32>,
}

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> @builtin(position) vec4<f32> {
    let deformation = mat3x3<f32>(
        instance.inst_def_0,
        instance.inst_def_1,
        instance.inst_def_2,
    );

    let scaled_pos = model.scale * vertex.position;
    let deformed_pos = deformation * scaled_pos;
    let model_pos = model.transform * vec4<f32>(deformed_pos, 1.0);
    let world_pos = vec4<f32>(instance.inst_tra, 0.0) + model_pos;

    return view.view_proj * vec4<f32>(world_pos.xyz, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // Colored transmittance of a translucent occluder: `T = 1 - a*(1 - rgb)`.
    // Clear glass (a = 0) leaves light untouched (T = 1); as opacity rises the
    // occluder both dims and tints the light by its color. The blend multiplies
    // this into the accumulated transmittance.
    let a = model.color.a;
    let t = vec3<f32>(1.0) - a * (vec3<f32>(1.0) - model.color.rgb);
    return vec4<f32>(t, 1.0);
}
