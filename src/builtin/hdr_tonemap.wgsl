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
};

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
    let hdr = (scene.rgb + bloom * u.bloom_intensity) * u.exposure;

    return vec4<f32>(apply_tonemap(hdr, u.tonemap_op), scene.a);
}
