// Oculus Rift stereo post-processing effect shader

// Bind group 0: Texture and sampler
@group(0) @binding(0)
var t_fbo: texture_2d<f32>;
@group(0) @binding(1)
var s_fbo: sampler;

// Bind group 1: Uniforms
struct OculusUniforms {
    kappa_0: f32,
    kappa_1: f32,
    kappa_2: f32,
    kappa_3: f32,
    scale: vec2<f32>,
    scale_in: vec2<f32>,
}

@group(1) @binding(0)
var<uniform> uniforms: OculusUniforms;

// Constants
const LensCenterLeft: vec2<f32> = vec2<f32>(0.25, 0.5);
const LensCenterRight: vec2<f32> = vec2<f32>(0.75, 0.5);

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
    var theta: vec2<f32>;
    var rSq: f32;
    var rvector: vec2<f32>;
    var tc: vec2<f32>;
    var left_eye: bool;

    if (in.tex_coord.x < 0.5) {
        left_eye = true;
    } else {
        left_eye = false;
    }

    if (left_eye) {
        theta = (in.tex_coord - LensCenterLeft) * uniforms.scale_in;
    } else {
        theta = (in.tex_coord - LensCenterRight) * uniforms.scale_in;
    }

    rSq = theta.x * theta.x + theta.y * theta.y;
    rvector = theta * (uniforms.kappa_0 + uniforms.kappa_1 * rSq + uniforms.kappa_2 * rSq * rSq + uniforms.kappa_3 * rSq * rSq * rSq);

    if (left_eye) {
        tc = LensCenterLeft + uniforms.scale * rvector;
    } else {
        tc = LensCenterRight + uniforms.scale * rvector;
    }

    // Keep within bounds of texture
    if ((left_eye && (tc.x < 0.0 || tc.x > 0.5)) ||
        (!left_eye && (tc.x < 0.5 || tc.x > 1.0)) ||
        tc.y < 0.0 || tc.y > 1.0) {
        discard;
    }

    let color = textureSample(t_fbo, s_fbo, tc);
    return vec4<f32>(color.rgb, color.a);
}
