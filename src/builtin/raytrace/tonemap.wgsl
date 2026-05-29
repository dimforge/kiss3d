// Fullscreen tonemap pass: reads the HDR accumulation buffer, applies ACES
// filmic tonemapping and manual gamma (the surface format is non-sRGB), and
// writes the LDR result to the output view.

struct TonemapUniforms {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    exposure: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
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

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
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
    var color = aces(hdr);
    color = pow(color, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(color, 1.0);
}
