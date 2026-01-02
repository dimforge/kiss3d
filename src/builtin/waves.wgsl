// Waves post-processing effect shader

// Bind group 0: Texture and sampler
@group(0) @binding(0)
var t_fbo: texture_2d<f32>;
@group(0) @binding(1)
var s_fbo: sampler;

// Bind group 1: Uniforms
struct WavesUniforms {
    offset: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(1) @binding(0)
var<uniform> uniforms: WavesUniforms;

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
    var texcoord = in.tex_coord;
    texcoord.x += sin(texcoord.y * 4.0 * 2.0 * 3.14159 + uniforms.offset) / 100.0;
    let color = textureSample(t_fbo, s_fbo, texcoord);
    return vec4<f32>(color.rgb, color.a);
}
