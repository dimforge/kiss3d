// Depth-only shader for the shadow-map pre-pass.
//
// Transforms instanced mesh geometry by the object's world transform and then by
// the light-space view-projection matrix. No fragment stage is bound: only depth
// is written into the shadow atlas layer. The world-space transform mirrors the
// position pipeline of `default.wgsl` so the shadow geometry matches the lit one.
//
// The `@if(skinned)` deform variant (selected by the `skinned` feature) deforms the
// mesh by morph targets and/or the joint-matrix palette so an animated/morphed
// caster casts a correctly-posed shadow — reading skin joints/weights and morph
// deltas from the group-2 storage buffers by vertex index, exactly like the
// deformed color pass. A morph-only (un-skinned) caster still uses its model
// transform (group 1); a skinned caster ignores it (the palette maps the bind-pose
// vertex straight to world space, per the glTF spec).

// Group 0: per-view light-space matrix (dynamic offset, one slot per atlas view).
struct ViewUniforms {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> view: ViewUniforms;

// Group 1: per-object model transform (dynamic offset, one slot per object).
struct ModelUniforms {
    transform: mat4x4<f32>,
    scale: mat3x3<f32>,
}

@group(1) @binding(0)
var<uniform> model: ModelUniforms;

// Group 2: the shared deform group (see builtin/deform.rs), only in the skinned variant.
@if(skinned) struct DeformControl {
    num_targets: u32,
    num_vertices: u32,
    has_skin: u32,
    has_morph_normals: u32,
    weights: array<vec4<f32>, 16>,
}
@if(skinned) @group(2) @binding(0) var<storage, read> joint_palette: array<mat4x4<f32>>;
@if(skinned) @group(2) @binding(1) var<storage, read> skin_joints: array<vec4<u32>>;
@if(skinned) @group(2) @binding(2) var<storage, read> skin_weights: array<vec4<f32>>;
@if(skinned) @group(2) @binding(3) var<storage, read> morph_pos: array<vec4<f32>>;
@if(skinned) @group(2) @binding(4) var<storage, read> morph_nrm: array<vec4<f32>>;
@if(skinned) @group(2) @binding(5) var<uniform> deform: DeformControl;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct InstanceInput {
    @location(1) inst_tra: vec3<f32>,
    @location(2) inst_def_0: vec3<f32>,
    @location(3) inst_def_1: vec3<f32>,
    @location(4) inst_def_2: vec3<f32>,
}

@if(!skinned)
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

@if(skinned)
@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
    @builtin(vertex_index) vid: u32,
) -> @builtin(position) vec4<f32> {
    // Morph: accumulate weighted position deltas.
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

    return view.view_proj * vec4<f32>(world_pos, 1.0);
}
