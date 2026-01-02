// Wireframe shader for kiss3d
// Simple shader for rendering mesh edges as 1-pixel lines

// Bind group 0: Frame uniforms (view, projection)
struct FrameUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    light_position: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

// Bind group 1: Object uniforms (transform, scale, color)
// This matches the ObjectUniforms struct from object_material.rs
struct ObjectUniforms {
    transform: mat4x4<f32>,
    ntransform: mat3x3<f32>,  // mat3x3 padded to mat3x4 (3 vec4s)
    scale: mat3x3<f32>,        // mat3x3 padded to mat3x4 (3 vec4s)
    color: vec3<f32>,
    _padding: f32,
}

@group(1) @binding(0)
var<uniform> object: ObjectUniforms;

// Vertex input
struct VertexInput {
    @location(0) position: vec3<f32>,
}

// Instance input
struct InstanceInput {
    @location(3) inst_tra: vec3<f32>,
    @location(4) inst_color: vec4<f32>,
    @location(5) inst_def_0: vec3<f32>,
    @location(6) inst_def_1: vec3<f32>,
    @location(7) inst_def_2: vec3<f32>,
}

// Vertex output / Fragment input
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) vert_color: vec4<f32>,
}

@vertex
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
    out.vert_color = instance.inst_color * vec4<f32>(object.color, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.vert_color.rgb, in.vert_color.a);
}
