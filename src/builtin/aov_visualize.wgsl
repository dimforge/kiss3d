// Visualizes raw AOV buffers (linear depth, encoded normals, segmentation ids)
// as display-ready colors, with a fullscreen triangle.
//
// Two fragment entry points share the vertex stage: `fs_float` reads the float
// AOV texture (depth or normals, selected by `params.x`), `fs_seg` reads the
// integer segmentation texture. Each pipeline's bind group layout only covers
// the bindings its entry point uses.
//
// params:
//   x: float mode (0 = depth, 1 = normals)
//   y: depth range in world units (depth mode only)
//   z: 1.0 when the target is an sRGB format. The visualization is computed in
//      display space (matching the CPU `snap_*` images); linearizing it first
//      makes the hardware sRGB encode round-trip to the intended value.

struct VisUniforms {
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uni: VisUniforms;
@group(0) @binding(1) var t_float: texture_2d<f32>;
@group(0) @binding(2) var t_seg: texture_2d<u32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Fullscreen triangle.
    var out: VsOut;
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

fn to_target(color: vec3<f32>) -> vec4<f32> {
    if uni.params.z > 0.5 {
        return vec4<f32>(pow(color, vec3<f32>(2.2)), 1.0);
    }
    return vec4<f32>(color, 1.0);
}

@fragment
fn fs_float(in: VsOut) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(in.pos.xy);
    let texel = textureLoad(t_float, coord, 0);
    if uni.params.x < 0.5 {
        // Depth: nearer = brighter over [0, range]; background (0) = black.
        let d = texel.r;
        var v = 0.0;
        if d > 0.0 {
            v = 1.0 - clamp(d / max(uni.params.y, 1.0e-6), 0.0, 1.0);
        }
        return to_target(vec3<f32>(v));
    }
    // Normals: already encoded to [0, 1].
    return to_target(texel.rgb);
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let i = floor(h * 6.0);
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    switch i32(i) % 6 {
        case 0: { return vec3<f32>(v, t, p); }
        case 1: { return vec3<f32>(q, v, p); }
        case 2: { return vec3<f32>(p, v, t); }
        case 3: { return vec3<f32>(p, q, v); }
        case 4: { return vec3<f32>(t, p, v); }
        default: { return vec3<f32>(v, p, q); }
    }
}

@fragment
fn fs_seg(in: VsOut) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(in.pos.xy);
    let id = textureLoad(t_seg, coord, 0).r;
    if id == 0u {
        return to_target(vec3<f32>(0.0));
    }
    // Golden-ratio hue stepping, matching the CPU `snap_segmentation_colored`.
    let hue = fract(f32(id) * 0.618034);
    return to_target(hsv_to_rgb(hue, 0.65, 0.95));
}
