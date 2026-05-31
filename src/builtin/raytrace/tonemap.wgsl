// Fullscreen tonemap pass: reads the HDR accumulation buffer and applies the
// selected tonemap operator + gamma via the shared `apply_tonemap` (see
// `tonemap_ops.wgsl`, concatenated as a prefix — it also declares the Tony
// McMapface LUT at group(0) bindings 6 & 7). Upscales from the traced resolution
// to the framebuffer if they differ.

struct TonemapUniforms {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    exposure: f32,
    // Operator code, matching `post_processing::Tonemap::as_u32`.
    tonemap_op: u32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0) var<storage, read> accum: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> u: TonemapUniforms;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VsOut {
    var out: VsOut;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let dx = u32(frag.x);
    let dy = u32(frag.y);
    if (dx >= u.dst_width || dy >= u.dst_height) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // Map the destination pixel back to a source (traced) pixel.
    let sx = min((dx * u.src_width) / u.dst_width, u.src_width - 1u);
    let sy = min((dy * u.src_height) / u.dst_height, u.src_height - 1u);
    let idx = sy * u.src_width + sx;
    let hdr = accum[idx].rgb * u.exposure;
    return vec4<f32>(apply_tonemap(hdr, u.tonemap_op), 1.0);
}
