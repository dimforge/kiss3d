// HDR bloom passes for the rasterization pipeline.
//
// This shader bundles the three building blocks of a dual-filter (Kawase) bloom:
//   * `fs_prefilter` extracts pixels brighter than a threshold from the HDR scene
//     texture into the first (half-resolution) bloom mip.
//   * `fs_downsample` halves resolution with a 13-tap filter (used while walking
//     down the mip chain).
//   * `fs_upsample` performs a 9-tap tent filter while walking back up the chain,
//     additively blending the blurred result into the larger mip.
//
// All passes sample an `Rgba16Float` HDR texture and write `Rgba16Float`, keeping
// energy in linear space until the final tonemap composite.

struct BloomUniforms {
    // Texel size (1/width, 1/height) of the *source* texture being sampled.
    src_texel: vec2<f32>,
    // Bloom brightness threshold (only used by the prefilter pass).
    threshold: f32,
    // Soft-knee width around the threshold for a smooth roll-off.
    knee: f32,
};

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;
@group(0) @binding(2) var<uniform> u: BloomUniforms;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VsOut {
    var out: VsOut;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    out.uv = (position + vec2<f32>(1.0, 1.0)) * 0.5;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

// Soft-knee threshold curve (as used by Unreal/Call-of-Duty style bloom).
fn prefilter(color: vec3<f32>) -> vec3<f32> {
    let brightness = max(color.r, max(color.g, color.b));
    let knee = max(u.knee, 1e-4);
    // Smooth quadratic roll-off in [threshold - knee, threshold + knee].
    var soft = brightness - u.threshold + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-4);
    let contribution = max(soft, brightness - u.threshold) / max(brightness, 1e-4);
    return color * contribution;
}

@fragment
fn fs_prefilter(in: VsOut) -> @location(0) vec4<f32> {
    let color = textureSample(t_src, s_src, in.uv).rgb;
    return vec4<f32>(prefilter(color), 1.0);
}

// 13-tap downsample filter (Sledgehammer / "Next Generation Post Processing").
@fragment
fn fs_downsample(in: VsOut) -> @location(0) vec4<f32> {
    let t = u.src_texel;
    let uv = in.uv;

    let a = textureSample(t_src, s_src, uv + vec2<f32>(-2.0 * t.x, 2.0 * t.y)).rgb;
    let b = textureSample(t_src, s_src, uv + vec2<f32>(0.0, 2.0 * t.y)).rgb;
    let c = textureSample(t_src, s_src, uv + vec2<f32>(2.0 * t.x, 2.0 * t.y)).rgb;

    let d = textureSample(t_src, s_src, uv + vec2<f32>(-2.0 * t.x, 0.0)).rgb;
    let e = textureSample(t_src, s_src, uv).rgb;
    let f = textureSample(t_src, s_src, uv + vec2<f32>(2.0 * t.x, 0.0)).rgb;

    let g = textureSample(t_src, s_src, uv + vec2<f32>(-2.0 * t.x, -2.0 * t.y)).rgb;
    let h = textureSample(t_src, s_src, uv + vec2<f32>(0.0, -2.0 * t.y)).rgb;
    let i = textureSample(t_src, s_src, uv + vec2<f32>(2.0 * t.x, -2.0 * t.y)).rgb;

    let j = textureSample(t_src, s_src, uv + vec2<f32>(-t.x, t.y)).rgb;
    let k = textureSample(t_src, s_src, uv + vec2<f32>(t.x, t.y)).rgb;
    let l = textureSample(t_src, s_src, uv + vec2<f32>(-t.x, -t.y)).rgb;
    let m = textureSample(t_src, s_src, uv + vec2<f32>(t.x, -t.y)).rgb;

    var color = e * 0.125;
    color = color + (a + c + g + i) * 0.03125;
    color = color + (b + d + f + h) * 0.0625;
    color = color + (j + k + l + m) * 0.125;
    return vec4<f32>(color, 1.0);
}

// 9-tap tent upsample filter.
@fragment
fn fs_upsample(in: VsOut) -> @location(0) vec4<f32> {
    let t = u.src_texel;
    let uv = in.uv;

    var color = textureSample(t_src, s_src, uv + vec2<f32>(-t.x, t.y)).rgb;
    color = color + textureSample(t_src, s_src, uv + vec2<f32>(0.0, t.y)).rgb * 2.0;
    color = color + textureSample(t_src, s_src, uv + vec2<f32>(t.x, t.y)).rgb;

    color = color + textureSample(t_src, s_src, uv + vec2<f32>(-t.x, 0.0)).rgb * 2.0;
    color = color + textureSample(t_src, s_src, uv).rgb * 4.0;
    color = color + textureSample(t_src, s_src, uv + vec2<f32>(t.x, 0.0)).rgb * 2.0;

    color = color + textureSample(t_src, s_src, uv + vec2<f32>(-t.x, -t.y)).rgb;
    color = color + textureSample(t_src, s_src, uv + vec2<f32>(0.0, -t.y)).rgb * 2.0;
    color = color + textureSample(t_src, s_src, uv + vec2<f32>(t.x, -t.y)).rgb;

    return vec4<f32>(color * (1.0 / 16.0), 1.0);
}
