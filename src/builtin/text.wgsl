// Text rendering shader for kiss3d
// Used for rendering text with a glyph cache texture

// Bind group 0: Uniforms
struct TextUniforms {
    inv_size: vec2<f32>,
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: TextUniforms;

// Bind group 1: Texture and sampler
@group(1) @binding(0)
var t_glyph: texture_2d<f32>;
@group(1) @binding(1)
var s_glyph: sampler;

// Vertex input - interleaved position, UV, and color
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coord: vec2<f32>,
    @location(2) color: vec4<f32>,
}

// Vertex output / Fragment input
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(
        vertex.position.x * uniforms.inv_size.x - 1.0,
        vertex.position.y * uniforms.inv_size.y + 1.0,
        0.0,  // z=0 is valid in wgpu's [0,1] depth range
        1.0
    );
    out.tex_coord = vertex.tex_coord;
    out.color = vertex.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let glyph_alpha = textureSample(t_glyph, s_glyph, in.tex_coord).r;
    return vec4<f32>(in.color.rgb, in.color.a * glyph_alpha);
}
