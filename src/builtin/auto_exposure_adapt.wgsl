import package::common::fullscreen_triangle_xy;
// Auto-exposure adaptation: smoothly move the current exposure toward the target
// implied by the metered average luminance (eye adaptation).
//
// Reads the 1x1 metered luminance and the previous frame's 1x1 exposure, and
// writes the new 1x1 exposure. The target exposure is `key / avg_luminance`,
// clamped to a [min, max] multiplier, approached exponentially over `dt`.

struct AdaptUniforms {
    dt: f32,
    speed: f32,
    min_exposure: f32,
    max_exposure: f32,
    key: f32,
    // Scalar padding (a vec3 here would force 16-byte alignment → 48-byte struct,
    // mismatching the tightly-packed 32-byte Rust `AdaptUniforms`).
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var t_meter: texture_2d<f32>;
@group(0) @binding(1) var t_prev: texture_2d<f32>;
@group(0) @binding(2) var s_point: sampler;
@group(0) @binding(3) var<uniform> u: AdaptUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    out.pos = vec4<f32>(fullscreen_triangle_xy(vid), 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(_in: VsOut) -> @location(0) vec4<f32> {
    let c = vec2<f32>(0.5, 0.5);
    let avg = textureSampleLevel(t_meter, s_point, c, 0.0).r;
    let prev = textureSampleLevel(t_prev, s_point, c, 0.0).r;

    var tgt = u.key / max(avg, 1e-4);
    tgt = clamp(tgt, u.min_exposure, u.max_exposure);

    // Exponential smoothing, frame-rate independent via dt.
    let t = clamp(1.0 - exp(-u.dt * u.speed), 0.0, 1.0);
    // First frame (prev == 0) snaps straight to the target.
    let blended = select(mix(prev, tgt, t), tgt, prev <= 0.0);
    return vec4<f32>(blended, 0.0, 0.0, 1.0);
}
