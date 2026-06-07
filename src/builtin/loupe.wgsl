// Magnifier loupe post-processing effect.
//
// Blits the input frame to the output, then draws a nearest-neighbour magnified
// crop of a focus region into a corner inset so individual pixels read as crisp
// blocks. `@builtin(position)` gives framebuffer pixel coords (y down).

struct Uniforms {
    // Framebuffer size, in pixels.
    resolution: vec2<f32>,
    // Center of the magnified source region, in pixels.
    focus_px: vec2<f32>,
    // Inset rectangle (the on-screen loupe), in pixels.
    inset_min: vec2<f32>,
    inset_max: vec2<f32>,
    // Half the side of the (square) magnified source region, in pixels.
    region_half_px: f32,
    // RGB color of the region outline and inset frame (alpha unused).
    border_color: vec4<f32>,
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;
@group(0) @binding(2) var<uniform> u: Uniforms;

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos, 0.0, 1.0);
}

// True when `p` lies within `t` pixels of the rectangle [mn, mx] outline.
fn on_border(p: vec2<f32>, mn: vec2<f32>, mx: vec2<f32>, t: f32) -> bool {
    let outside = any(p < mn - vec2<f32>(t)) || any(p > mx + vec2<f32>(t));
    let inside = all(p > mn + vec2<f32>(t)) && all(p < mx - vec2<f32>(t));
    return !outside && !inside;
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = frag.xy;

    // Passthrough of the full frame.
    var col = textureSampleLevel(src_tex, src_smp, p / u.resolution, 0.0).rgb;

    // Outline of the magnified source region on the main image.
    let region_min = u.focus_px - vec2<f32>(u.region_half_px);
    let region_max = u.focus_px + vec2<f32>(u.region_half_px);
    let in_inset = all(p >= u.inset_min) && all(p <= u.inset_max);
    if (!in_inset && on_border(p, region_min, region_max, 1.0)) {
        col = u.border_color.rgb;
    }

    // Magnified inset, nearest-neighbour so source pixels read as crisp blocks.
    if (in_inset) {
        let local = (p - u.inset_min) / (u.inset_max - u.inset_min);
        let src_px = u.focus_px + (local - vec2<f32>(0.5)) * (2.0 * u.region_half_px);
        let snapped = floor(src_px) + vec2<f32>(0.5);
        col = textureSampleLevel(src_tex, src_smp, snapped / u.resolution, 0.0).rgb;
    }
    // Inset frame.
    if (on_border(p, u.inset_min, u.inset_max, 2.0)) {
        col = u.border_color.rgb;
    }

    return vec4<f32>(col, 1.0);
}
