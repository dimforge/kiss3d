import package::common::{fullscreen_triangle_xy, luminance};
// Auto-exposure metering: reduce the HDR scene to a single average luminance.
//
// Renders into a 1x1 R16Float target. The (single) fragment samples a coarse
// grid over the scene and returns the geometric-mean luminance (the exponential
// of the average log-luminance), which is robust to a few very bright pixels.

@group(0) @binding(0) var t_scene: texture_2d<f32>;
@group(0) @binding(1) var s_scene: sampler;

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
    let n = 32;
    var sum = 0.0;
    for (var y = 0; y < n; y = y + 1) {
        for (var x = 0; x < n; x = x + 1) {
            let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) / f32(n);
            let l = luminance(textureSampleLevel(t_scene, s_scene, uv, 0.0).rgb);
            sum += log(max(l, 1e-4));
        }
    }
    let avg = exp(sum / f32(n * n));
    return vec4<f32>(avg, 0.0, 0.0, 1.0);
}
