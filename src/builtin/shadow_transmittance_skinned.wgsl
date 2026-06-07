// Deformed (skinning + morph) variant of the colored-transmittance shadow pass.
//
// Deforms a translucent caster by morph targets and/or its joint-matrix palette so
// its tinted shadow tracks the animated/morphed pose, then writes the same colored
// transmittance as the non-deformed pass. Deform data is read from the group-2
// storage buffers by vertex index. A morph-only caster uses its model transform
// (group 1); a skinned caster ignores it (per the glTF spec).

// Group 0: per-view light-space matrix (dynamic offset, one slot per atlas view).
struct ViewUniforms {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> view: ViewUniforms;

// Group 1: per-object model transform + base color. Layout matches
// `ShadowModelUniforms` in builtin/shadow.rs.
struct ModelUniforms {
    transform: mat4x4<f32>,
    scale: mat3x3<f32>,
    color: vec4<f32>,
}

@group(1) @binding(0)
var<uniform> model: ModelUniforms;

// Group 2: the shared deform group (see builtin/deform.rs).
struct DeformControl {
    num_targets: u32,
    num_vertices: u32,
    has_skin: u32,
    has_morph_normals: u32,
    weights: array<vec4<f32>, 16>,
}
@group(2) @binding(0) var<storage, read> joint_palette: array<mat4x4<f32>>;
@group(2) @binding(1) var<storage, read> skin_joints: array<vec4<u32>>;
@group(2) @binding(2) var<storage, read> skin_weights: array<vec4<f32>>;
@group(2) @binding(3) var<storage, read> morph_pos: array<vec4<f32>>;
@group(2) @binding(4) var<storage, read> morph_nrm: array<vec4<f32>>;
@group(2) @binding(5) var<uniform> deform: DeformControl;

// Group 3: the occluder's albedo (base-color) texture (tints the shadow).
@group(3) @binding(0)
var t_albedo: texture_2d<f32>;
@group(3) @binding(1)
var s_albedo: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(5) uv: vec2<f32>,
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
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
    @builtin(vertex_index) vid: u32,
) -> VertexOutput {
    var pos = vertex.position;
    if (deform.num_targets > 0u) {
        for (var t = 0u; t < deform.num_targets; t = t + 1u) {
            let wgt = deform.weights[t >> 2u][t & 3u];
            if (wgt != 0.0) {
                pos = pos + wgt * morph_pos[t * deform.num_vertices + vid].xyz;
            }
        }
    }

    var world_pos: vec3<f32>;
    if (deform.has_skin != 0u) {
        var w = skin_weights[vid];
        let j = skin_joints[vid];
        let wsum = w.x + w.y + w.z + w.w;
        if (wsum > 0.0) { w = w / wsum; }
        let skin =
            w.x * joint_palette[j.x] +
            w.y * joint_palette[j.y] +
            w.z * joint_palette[j.z] +
            w.w * joint_palette[j.w];
        world_pos = (skin * vec4<f32>(pos, 1.0)).xyz;
    } else {
        let deformation = mat3x3<f32>(
            instance.inst_def_0,
            instance.inst_def_1,
            instance.inst_def_2,
        );
        let scaled_pos = model.scale * pos;
        let deformed_pos = deformation * scaled_pos;
        let model_pos = model.transform * vec4<f32>(deformed_pos, 1.0);
        world_pos = (vec4<f32>(instance.inst_tra, 0.0) + model_pos).xyz;
    }

    var out: VertexOutput;
    out.clip_position = view.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = vertex.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Surface color = base color × albedo texture; `T = 1 - a*(1 - rgb)`.
    let albedo = model.color.rgb * textureSample(t_albedo, s_albedo, in.uv).rgb;
    let a = model.color.a;
    let t = vec3<f32>(1.0) - a * (vec3<f32>(1.0) - albedo);
    return vec4<f32>(t, 1.0);
}
