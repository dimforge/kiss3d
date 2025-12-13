// Sobel edge highlight post-processing effect shader

// Bind group 0: Color texture and sampler
@group(0) @binding(0)
var t_color: texture_2d<f32>;
@group(0) @binding(1)
var s_color: sampler;

// Bind group 1: Depth texture and sampler
@group(1) @binding(0)
var t_depth: texture_depth_2d;
@group(1) @binding(1)
var s_depth: sampler;

// Bind group 2: Uniforms
struct SobelUniforms {
    nx: f32,          // 2.0 / width (pixel step in x)
    ny: f32,          // 2.0 / height (pixel step in y)
    znear: f32,
    zfar: f32,
    threshold: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(2) @binding(0)
var<uniform> uniforms: SobelUniforms;

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

// Convert non-linear depth to linear depth
fn lin_depth(uv: vec2<f32>) -> f32 {
    // textureSample on texture_depth_2d returns a scalar f32
    let nlin_depth = textureSample(t_depth, s_depth, uv);
    return uniforms.znear * uniforms.zfar / ((nlin_depth * (uniforms.zfar - uniforms.znear)) - uniforms.zfar);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texcoord = in.tex_coord;

    // Sobel kernel for X direction
    let KX = array<f32, 9>(
        1.0, 0.0, -1.0,
        2.0, 0.0, -2.0,
        1.0, 0.0, -1.0
    );

    var gx: f32 = 0.0;
    for (var i: i32 = -1; i < 2; i++) {
        for (var j: i32 = -1; j < 2; j++) {
            let off = (i + 1) * 3 + j + 1;
            let sample_pos = vec2<f32>(
                texcoord.x + f32(i) * uniforms.nx,
                texcoord.y + f32(j) * uniforms.ny
            );
            gx += KX[off] * lin_depth(sample_pos);
        }
    }

    // Sobel kernel for Y direction
    let KY = array<f32, 9>(
        1.0,  2.0,  1.0,
        0.0,  0.0,  0.0,
        -1.0, -2.0, -1.0
    );

    var gy: f32 = 0.0;
    for (var i: i32 = -1; i < 2; i++) {
        for (var j: i32 = -1; j < 2; j++) {
            let off = (i + 1) * 3 + j + 1;
            let sample_pos = vec2<f32>(
                texcoord.x + f32(i) * uniforms.nx,
                texcoord.y + f32(j) * uniforms.ny
            );
            gy += KY[off] * lin_depth(sample_pos);
        }
    }

    let gradient = sqrt(gx * gx + gy * gy);

    var edge: f32;
    if (gradient > uniforms.threshold) {
        edge = 0.0;
    } else {
        edge = 1.0 - gradient / uniforms.threshold;
    }

    let color = textureSample(t_color, s_color, texcoord);
    return vec4<f32>(linear_to_srgb(edge * color.xyz), 1.0);
}

// Convert linear RGB to sRGB for display.
fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = linear < vec3<f32>(0.0031308);
    let lower = linear * 12.92;
    let higher = pow(linear, vec3<f32>(1.0 / 2.4)) * 1.055 - vec3<f32>(0.055);
    return select(higher, lower, cutoff);
}
