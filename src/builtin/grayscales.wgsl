// Grayscale post-processing effect shader

// Bind group 0: Texture and sampler
@group(0) @binding(0)
var t_fbo: texture_2d<f32>;
@group(0) @binding(1)
var s_fbo: sampler;

// Vertex input
struct VertexInput {
    @location(0) position: vec2<f32>,
}

// Vertex output / Fragment input
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(vertex.position, 0.0, 1.0);
    out.tex_coord = (vertex.position + vec2<f32>(1.0, 1.0)) / 2.0;
    // Flip Y coordinate for wgpu coordinate system
    out.tex_coord.y = 1.0 - out.tex_coord.y;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_fbo, s_fbo, in.tex_coord);
    // Use standard luminance weights
    let gray = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
    let result = vec3<f32>(gray, gray, gray);
    return vec4<f32>(result, color.a);
}
