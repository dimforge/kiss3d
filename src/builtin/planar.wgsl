// Planar (2D) object shader for kiss3d
// Used for rendering 2D overlay objects

// Bind group 0: Frame uniforms
// Note: mat3x3 is stored as array<vec4<f32>, 3> for proper alignment
struct FrameUniforms {
    view_0: vec4<f32>,
    view_1: vec4<f32>,
    view_2: vec4<f32>,
    proj_0: vec4<f32>,
    proj_1: vec4<f32>,
    proj_2: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

// Bind group 1: Object uniforms
struct ObjectUniforms {
    model_0: vec4<f32>,
    model_1: vec4<f32>,
    model_2: vec4<f32>,
    scale_0: vec4<f32>,
    scale_1: vec4<f32>,
    color: vec3<f32>,
    _padding: f32,
}

@group(1) @binding(0)
var<uniform> object: ObjectUniforms;

// Bind group 2: Texture and sampler
@group(2) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(2) @binding(1)
var s_diffuse: sampler;

// Vertex input
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coord: vec2<f32>,
}

// Instance input (separate buffers for better performance)
struct InstanceInput {
    @location(2) inst_tra: vec2<f32>,
    @location(3) inst_color: vec4<f32>,
    @location(4) inst_def_0: vec2<f32>, // deformation matrix column 0
    @location(5) inst_def_1: vec2<f32>, // deformation matrix column 1
}

// Vertex output / Fragment input
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) vert_color: vec4<f32>,
}

// Reconstruct mat3x3 from padded vec4 columns
fn unpack_mat3(col0: vec4<f32>, col1: vec4<f32>, col2: vec4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        col0.xyz,
        col1.xyz,
        col2.xyz
    );
}

// Reconstruct mat2x2 from padded vec4 columns
fn unpack_mat2(col0: vec4<f32>, col1: vec4<f32>) -> mat2x2<f32> {
    return mat2x2<f32>(
        col0.xy,
        col1.xy
    );
}

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    // Reconstruct matrices from uniform data
    let view = unpack_mat3(frame.view_0, frame.view_1, frame.view_2);
    let proj = unpack_mat3(frame.proj_0, frame.proj_1, frame.proj_2);
    let model = unpack_mat3(object.model_0, object.model_1, object.model_2);
    let scale = unpack_mat2(object.scale_0, object.scale_1);

    // Build deformation matrix from separate column vectors
    let def = mat2x2<f32>(
        instance.inst_def_0,
        instance.inst_def_1
    );

    // Transform position
    let scaled_pos = scale * vertex.position;
    let deformed_pos = def * scaled_pos;
    let model_pos = model * vec3<f32>(deformed_pos, 1.0);
    let view_pos = vec3<f32>(instance.inst_tra, 0.0) + model_pos;
    var projected_pos = proj * view * view_pos;
    projected_pos.z = 0.0;

    out.clip_position = vec4<f32>(projected_pos, 1.0);
    out.tex_coord = vertex.tex_coord;
    out.vert_color = instance.inst_color;

    return out;
}

// Convert linear RGB to sRGB for display.
fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = linear < vec3<f32>(0.0031308);
    let lower = linear * 12.92;
    let higher = pow(linear, vec3<f32>(1.0 / 2.4)) * 1.055 - vec3<f32>(0.055);
    return select(higher, lower, cutoff);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(t_diffuse, s_diffuse, in.tex_coord);
    let final_color = tex_color * (vec4<f32>(object.color, 1.0) * in.vert_color);
    return vec4<f32>(linear_to_srgb(final_color.rgb), final_color.a);
}
