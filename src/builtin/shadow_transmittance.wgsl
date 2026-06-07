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

// Group 2: the occluder's albedo (base-color) texture. The shadow tint follows the
// surface color, so a translucent object whose color comes from its texture
// (a white base color × an orange texture, say) casts a correspondingly colored
// shadow — not a clear one.
@group(2) @binding(0)
var t_albedo: texture_2d<f32>;
@group(2) @binding(1)
var s_albedo: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(7) uv: vec2<f32>,
}

struct InstanceInput {
    @location(1) inst_tra: vec3<f32>,
    @location(2) inst_def_0: vec3<f32>,
    @location(3) inst_def_1: vec3<f32>,
    @location(4) inst_def_2: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    let deformation = mat3x3<f32>(
        instance.inst_def_0,
        instance.inst_def_1,
        instance.inst_def_2,
    );

    let scaled_pos = model.scale * vertex.position;
    let deformed_pos = deformation * scaled_pos;
    let model_pos = model.transform * vec4<f32>(deformed_pos, 1.0);
    let world_pos = vec4<f32>(instance.inst_tra, 0.0) + model_pos;

    var out: VertexOutput;
    out.clip_position = view.view_proj * vec4<f32>(world_pos.xyz, 1.0);
    out.uv = vertex.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Surface color = base color × albedo texture. Colored transmittance of a
    // translucent occluder: `T = 1 - a*(1 - rgb)`. Clear/white surfaces (a = 0, or
    // rgb = 1) leave light untouched (T = 1); as opacity rises the occluder dims
    // and tints the light by its color. The blend multiplies this into the
    // accumulated transmittance.
    let albedo = model.color.rgb * textureSample(t_albedo, s_albedo, in.uv).rgb;
    let a = model.color.a;
    let t = vec3<f32>(1.0) - a * (vec3<f32>(1.0) - albedo);
    return vec4<f32>(t, 1.0);
}
