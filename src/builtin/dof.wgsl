import package::common::{fullscreen_triangle_xy, fullscreen_uv_from_clip};
// Depth of field — two fragment entry points sharing one full-screen vertex stage.
//
// `fs_coc` reads the resolved HDR scene + the prepass view-position G-buffer and
// writes `vec4(color.rgb, signed_coc)` into the DoF chain (mip 0). The signed
// circle-of-confusion diameter is in pixels: negative in front of the focal plane
// (near field), positive behind it (far field). A box mip chain is then built from
// that texture (so each coarser mip averages both color and CoC), giving cheap
// pre-blurred reads at large blur radii.
//
// `fs_gather` reconstructs each output pixel by gathering a golden-angle spiral of
// taps over a disk of radius = max CoC. A tap contributes only where its own CoC
// reaches the destination pixel (scatter-as-gather), which keeps blurry foreground
// bleeding over sharp focus while preventing sharp background from bleeding onto
// it. The mip LOD of each tap grows with its distance so a fixed tap budget still
// covers large blur radii smoothly. Two kernels: a uniform disk (Bokeh) and a
// gaussian falloff (Gaussian). The composite is written back over the HDR scene.

const PI: f32 = 3.14159265359;
const GOLDEN_ANGLE: f32 = 2.39996323;

struct DofUniforms {
    proj: mat4x4<f32>,
    // (inv_w, inv_h, viewport_height, max_lod)
    params0: vec4<f32>,
    // (focal_distance, aperture_f_stops, sensor_height, max_coc_diameter)
    params1: vec4<f32>,
    // (max_depth, background_depth, mode, num_taps)
    params2: vec4<f32>,
}

@group(0) @binding(0) var t_a: texture_2d<f32>;
// `t_b` is the view-position G-buffer in the CoC pass and unused in the gather
// pass (the same two-texture layout serves both pipelines).
@group(0) @binding(1) var t_b: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> u: DofUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let xy = fullscreen_triangle_xy(vid);
    var o: VsOut;
    o.pos = vec4<f32>(xy, 0.0, 1.0);
    o.uv = fullscreen_uv_from_clip(xy);
    return o;
}

// Signed CoC diameter (pixels) for a given linear (positive) view-space depth.
fn circle_of_confusion(depth_in: f32) -> f32 {
    let depth = max(min(depth_in, u.params2.x), 1e-4); // clamp to max_depth
    let focal_distance = u.params1.x;
    let f_stops = max(u.params1.y, 1e-3);
    let sensor_h = max(u.params1.z, 1e-6);
    // proj[1][1] = cot(fov_y / 2); focal_length and aperture share sensor units.
    let focal_length = 0.5 * sensor_h * u.proj[1][1];
    let aperture_d = focal_length / f_stops;
    let denom = focal_distance - focal_length;
    var coc_frac = 0.0;
    if abs(denom) > 1e-6 {
        // Signed CoC as a fraction of the sensor / viewport height.
        coc_frac = aperture_d * focal_length / denom * (1.0 - focal_distance / depth) / sensor_h;
    }
    let coc_px = coc_frac * u.params0.z; // -> pixels (× viewport height)
    let max_coc = u.params1.w;
    return clamp(coc_px, -max_coc, max_coc);
}

@fragment
fn fs_coc(in: VsOut) -> @location(0) vec4<f32> {
    let color = textureSampleLevel(t_a, samp, in.uv, 0.0).rgb;
    let vp = textureSampleLevel(t_b, samp, in.uv, 0.0);
    // Background (no opaque surface) is treated as the far clip plane so the sky
    // gets the same far-field blur a distant surface would.
    var depth = u.params2.y;
    if vp.a >= 0.5 {
        depth = max(-vp.z, 1e-4); // view space looks down -Z
    }
    return vec4<f32>(color, circle_of_confusion(depth));
}

@fragment
fn fs_gather(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let texel = u.params0.xy;
    let max_lod = u.params0.w;
    let bokeh = u.params2.z < 0.5;
    let num_taps = i32(u.params2.w);

    let center = textureSampleLevel(t_a, samp, uv, 0.0);
    let center_coc = center.a;

    // Gather over the maximum possible blur disk so foreground pixels can scatter
    // onto in-focus neighbours; sparse far taps are smoothed by the mip chain.
    let gather_r = max(u.params1.w * 0.5, 1.0);
    if gather_r < 1.0 {
        return vec4<f32>(center.rgb, 1.0);
    }

    var color = center.rgb;
    var weight = 1.0;
    for (var i = 0; i < num_taps; i = i + 1) {
        let t = (f32(i) + 0.5) / f32(num_taps);
        let r = sqrt(t) * gather_r;            // uniform areal distribution
        let ang = f32(i) * GOLDEN_ANGLE;
        let off = vec2<f32>(cos(ang), sin(ang)) * r;
        let lod = clamp(log2(max(r, 1.0)) - 1.0, 0.0, max_lod);
        let s = textureSampleLevel(t_a, samp, uv + off * texel, lod);
        let s_r = abs(s.a);

        // A tap reaches this pixel only if its CoC radius covers the distance `r`.
        var w = smoothstep(r - 1.0, r + 1.0, s_r);
        // Keep sharp background from leaking onto a focused pixel: a far tap
        // (larger, positive CoC than the centre) is limited to the centre's reach.
        if s.a > center_coc {
            w = w * smoothstep(abs(center_coc) + 1.0, abs(center_coc) - 1.0, r);
        }
        if !bokeh {
            // Gaussian kernel: weight by distance within the tap's own radius.
            let sigma = max(s_r, 1.0) * 0.5;
            w = w * exp(-(r * r) / (2.0 * sigma * sigma));
        }
        color += s.rgb * w;
        weight += w;
    }
    return vec4<f32>(color / weight, 1.0);
}
