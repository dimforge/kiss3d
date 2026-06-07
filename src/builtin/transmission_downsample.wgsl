import package::common::{fullscreen_triangle_xy, fullscreen_uv_from_clip};
// Downsample filters for the refraction (transmission) background blur chain, one
// per quality preset. A plain box downsample keeps hard edges, so coarse mips look
// blocky when magnified across rough glass; wider near-Gaussian kernels stay smooth
// at the cost of more taps.
//   fs_low    — 1 bilinear tap (2x2 box). Cheapest, can look blocky.
//   fs_medium — 4 bilinear taps (≈4x4 tent).
//   fs_high   — 13-tap near-Gaussian (Jimenez / Call of Duty). Smoothest.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let xy = fullscreen_triangle_xy(vid);
    var out: VsOut;
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    out.uv = fullscreen_uv_from_clip(xy);
    return out;
}

@fragment
fn fs_low(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSampleLevel(src, samp, in.uv, 0.0).rgb, 1.0);
}

@fragment
fn fs_medium(in: VsOut) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src));
    let uv = in.uv;
    var c = textureSampleLevel(src, samp, uv + t * vec2<f32>(-1.0, -1.0), 0.0).rgb;
    c += textureSampleLevel(src, samp, uv + t * vec2<f32>( 1.0, -1.0), 0.0).rgb;
    c += textureSampleLevel(src, samp, uv + t * vec2<f32>(-1.0,  1.0), 0.0).rgb;
    c += textureSampleLevel(src, samp, uv + t * vec2<f32>( 1.0,  1.0), 0.0).rgb;
    return vec4<f32>(c * 0.25, 1.0);
}

@fragment
fn fs_high(in: VsOut) -> @location(0) vec4<f32> {
    // Texel size of the *source* mip (offsets are in source texels).
    let t = 1.0 / vec2<f32>(textureDimensions(src));
    let uv = in.uv;

    // Outer ring (2-texel reach), inner ring (1-texel reach), and center.
    let a = textureSampleLevel(src, samp, uv + t * vec2<f32>(-2.0, -2.0), 0.0).rgb;
    let b = textureSampleLevel(src, samp, uv + t * vec2<f32>( 0.0, -2.0), 0.0).rgb;
    let c = textureSampleLevel(src, samp, uv + t * vec2<f32>( 2.0, -2.0), 0.0).rgb;
    let d = textureSampleLevel(src, samp, uv + t * vec2<f32>(-2.0,  0.0), 0.0).rgb;
    let e = textureSampleLevel(src, samp, uv,                              0.0).rgb;
    let f = textureSampleLevel(src, samp, uv + t * vec2<f32>( 2.0,  0.0), 0.0).rgb;
    let g = textureSampleLevel(src, samp, uv + t * vec2<f32>(-2.0,  2.0), 0.0).rgb;
    let h = textureSampleLevel(src, samp, uv + t * vec2<f32>( 0.0,  2.0), 0.0).rgb;
    let i = textureSampleLevel(src, samp, uv + t * vec2<f32>( 2.0,  2.0), 0.0).rgb;
    let j = textureSampleLevel(src, samp, uv + t * vec2<f32>(-1.0, -1.0), 0.0).rgb;
    let k = textureSampleLevel(src, samp, uv + t * vec2<f32>( 1.0, -1.0), 0.0).rgb;
    let l = textureSampleLevel(src, samp, uv + t * vec2<f32>(-1.0,  1.0), 0.0).rgb;
    let m = textureSampleLevel(src, samp, uv + t * vec2<f32>( 1.0,  1.0), 0.0).rgb;

    // Weights: center group + four overlapping 2x2 boxes (Jimenez 13-tap).
    var color = e * 0.125;
    color += (a + c + g + i) * 0.03125;
    color += (b + d + f + h) * 0.0625;
    color += (j + k + l + m) * 0.125;
    return vec4<f32>(color, 1.0);
}
