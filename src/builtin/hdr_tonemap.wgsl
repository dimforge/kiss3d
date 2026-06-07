// Final HDR resolve pass for the rasterization pipeline.
//
// Reads the linear HDR scene texture (`Rgba16Float`), additively composites the
// blurred bloom texture, applies an exposure multiplier, then the selected
// tonemap operator + gamma via the shared `apply_tonemap` (see `tonemap_ops.wgsl`,
// concatenated as a prefix), and writes the LDR result to the output view.
//
// `tonemap_ops.wgsl` declares the Tony McMapface LUT at group(0) bindings 6 & 7.

struct TonemapUniforms {
    exposure: f32,
    // Operator code, matching `post_processing::Tonemap::as_u32`.
    tonemap_op: u32,
    // Additive bloom intensity (0 disables the bloom contribution).
    bloom_intensity: f32,
    _pad: f32,
    // Color grading: white-balance gain (rgb) + unused.
    white_balance: vec4<f32>,
    // (saturation, contrast, gamma, hue).
    grading: vec4<f32>,
};

// Rec. 709 luminance.
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Artistic color grading in linear HDR space: white balance, hue rotation,
// saturation, contrast (around mid-gray 0.18) and gamma. A neutral uniform
// (gains 1, hue 0) returns the input unchanged.
fn color_grade(c_in: vec3<f32>) -> vec3<f32> {
    var c = c_in * u.white_balance.rgb;

    // Hue rotation about the (1,1,1) grayscale axis.
    let hue = u.grading.w;
    if abs(hue) > 1e-4 {
        let cosA = cos(hue);
        let sinA = sin(hue);
        let k = 0.57735026; // 1/sqrt(3)
        // Rodrigues rotation matrix about the normalized (1,1,1) axis.
        let m = mat3x3<f32>(
            vec3<f32>(cosA + (1.0 - cosA) / 3.0,
                      (1.0 - cosA) / 3.0 - k * sinA,
                      (1.0 - cosA) / 3.0 + k * sinA),
            vec3<f32>((1.0 - cosA) / 3.0 + k * sinA,
                      cosA + (1.0 - cosA) / 3.0,
                      (1.0 - cosA) / 3.0 - k * sinA),
            vec3<f32>((1.0 - cosA) / 3.0 - k * sinA,
                      (1.0 - cosA) / 3.0 + k * sinA,
                      cosA + (1.0 - cosA) / 3.0),
        );
        c = m * c;
    }

    // Saturation around luminance.
    let l = luminance(c);
    c = mix(vec3<f32>(l), c, u.grading.x);

    // Contrast around mid-gray.
    c = (c - vec3<f32>(0.18)) * u.grading.y + vec3<f32>(0.18);

    // Gamma in linear space.
    c = pow(max(c, vec3<f32>(0.0)), vec3<f32>(u.grading.z));

    return max(c, vec3<f32>(0.0));
}

@group(0) @binding(0) var t_scene: texture_2d<f32>;
@group(0) @binding(1) var s_scene: sampler;
@group(0) @binding(2) var t_bloom: texture_2d<f32>;
@group(0) @binding(3) var s_bloom: sampler;
@group(0) @binding(4) var<uniform> u: TonemapUniforms;

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let scene = textureSample(t_scene, s_scene, in.uv);
    let bloom = textureSample(t_bloom, s_bloom, in.uv).rgb;

    // Composite bloom additively in linear HDR space, then expose.
    let exposed = (scene.rgb + bloom * u.bloom_intensity) * u.exposure;

    // Artistic color grading, then the tonemap operator.
    let hdr = color_grade(exposed);

    return vec4<f32>(apply_tonemap(hdr, u.tonemap_op), scene.a);
}
