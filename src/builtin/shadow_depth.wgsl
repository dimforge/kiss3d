// Depth-only shader for the shadow-map pre-pass.
//
// Transforms instanced mesh geometry by the object's world transform and then by
// the light-space view-projection matrix. No fragment stage is bound: only depth
// is written into the shadow atlas layer. The world-space transform mirrors the
// position pipeline of `default.wgsl` so the shadow geometry matches the lit one.

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
