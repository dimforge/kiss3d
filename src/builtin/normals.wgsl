// Normals visualization shader for kiss3d
// Colors each vertex based on its normal direction

// Bind group 0: Frame uniforms
struct FrameUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

// Bind group 1: Object uniforms
struct ObjectUniforms {
    transform: mat4x4<f32>,
    scale: mat3x3<f32>,
}

@group(1) @binding(0)
var<uniform> object: ObjectUniforms;

// Vertex input
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

// Vertex output / Fragment input
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ls_normal: vec3<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let scaled_pos = object.scale * vertex.position;
    out.clip_position = frame.proj * frame.view * object.transform * vec4<f32>(scaled_pos, 1.0);
    out.ls_normal = vertex.normal;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Map normal from [-1, 1] to [0, 1] for visualization
    let color = (in.ls_normal + vec3<f32>(1.0)) / 2.0;
    return vec4<f32>(color, 1.0);
}
