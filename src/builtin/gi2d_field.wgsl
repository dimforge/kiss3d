import package::common::{unpack_mat3, fullscreen_uv_from_clip};
// 2D global-illumination irradiance field (low-resolution, temporally accumulated).
//
// For each (low-res) pixel we ray-march a jittered fan against the analytic emitters
// and either the analytic occluder discs or, when `use_sdf` is set, a jump-flooded
// occluder distance field (one texture fetch per step, so the cost is independent of
// occluder count). The fan is rotated by a per-frame golden-angle offset and the
// result blended with the previous frame reprojected through the camera, so a few
// rays per frame converge to a smooth result.

const MAX_EMITTERS: u32 = 32u;
const MAX_OCCLUDERS: u32 = 64u;
const TAU: f32 = 6.2831853;
const GOLDEN: f32 = 2.39996323;

struct Emitter {
    pos_radius: vec4<f32>,
    color: vec4<f32>,
}

struct Occluder {
    pos_radius: vec4<f32>,
}

struct FieldUniforms {
    inv_vp_0: vec4<f32>,
    inv_vp_1: vec4<f32>,
    inv_vp_2: vec4<f32>,
    prev_vp_0: vec4<f32>,
    prev_vp_1: vec4<f32>,
    prev_vp_2: vec4<f32>,
    cur_vp_0: vec4<f32>,
    cur_vp_1: vec4<f32>,
    cur_vp_2: vec4<f32>,
    // num_rays, frame_index, temporal_blend, history_valid
    params: vec4<f32>,
    // use_sdf, _, _, _
    flags: vec4<f32>,
    // num_emitters, num_occluders, max_dist, max_steps
    counts: vec4<f32>,
    emitters: array<Emitter, MAX_EMITTERS>,
    occluders: array<Occluder, MAX_OCCLUDERS>,
}

@group(0) @binding(0)
var t_history: texture_2d<f32>;
@group(0) @binding(1)
var s_history: sampler;

@group(1) @binding(0)
var<uniform> u: FieldUniforms;

@group(2) @binding(0)
var t_sdf: texture_2d<f32>;
@group(2) @binding(1)
var s_sdf: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(vertex.position, 0.0, 1.0);
    out.tex_coord = fullscreen_uv_from_clip(vertex.position);
    return out;
}

fn unproject(uv: vec2<f32>) -> vec2<f32> {
    let clip = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let inv = mat3x3<f32>(u.inv_vp_0.xyz, u.inv_vp_1.xyz, u.inv_vp_2.xyz);
    let h = inv * vec3<f32>(clip, 1.0);
    return h.xy / h.z;
}

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Distance to the nearest occluder at world point `p`: a jump-flooded SDF texture
// fetch when `use_sdf`, otherwise the analytic minimum over the occluder discs.
fn occluder_distance(p: vec2<f32>) -> f32 {
    if (u.flags.x > 0.5) {
        let clip = mat3x3<f32>(u.cur_vp_0.xyz, u.cur_vp_1.xyz, u.cur_vp_2.xyz) * vec3<f32>(p, 1.0);
        let uv = vec2<f32>((clip.x + 1.0) * 0.5, (1.0 - clip.y) * 0.5);
        // Off-screen has no captured occluder data → treat as open.
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            return 1e9;
        }
        // Bias the unsigned field so it crosses zero just outside the surface.
        return textureSampleLevel(t_sdf, s_sdf, uv, 0.0).r - u.flags.y;
    }

    let num_occluders = u32(u.counts.y);
    var d = 1e9;
    for (var i = 0u; i < num_occluders; i = i + 1u) {
        let o = u.occluders[i];
        d = min(d, length(p - o.pos_radius.xy) - o.pos_radius.z);
    }
    return d;
}

fn trace(origin: vec2<f32>, dir: vec2<f32>) -> vec3<f32> {
    let num_emitters = u32(u.counts.x);
    let max_dist = u.counts.z;
    let max_steps = u32(u.counts.w);

    var t = 1.0;
    for (var step = 0u; step < max_steps; step = step + 1u) {
        let p = origin + dir * t;

        var d_emit = 1e9;
        var emit_idx = 0u;
        for (var i = 0u; i < num_emitters; i = i + 1u) {
            let e = u.emitters[i];
            let d = length(p - e.pos_radius.xy) - e.pos_radius.z;
            if (d < d_emit) {
                d_emit = d;
                emit_idx = i;
            }
        }

        let d_occ = occluder_distance(p);

        if (d_emit <= 0.0) {
            let e = u.emitters[emit_idx];
            return e.color.rgb * e.pos_radius.w;
        }
        if (d_occ <= 0.0) {
            return vec3<f32>(0.0);
        }

        t += max(min(d_emit, d_occ), 0.5);
        if (t > max_dist) {
            break;
        }
    }
    return vec3<f32>(0.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let world = unproject(in.tex_coord);
    let num_rays = max(u32(u.params.x), 1u);
    let frame = u.params.y;
    let blend = u.params.z;
    let history_valid = u.params.w;

    let jitter = hash12(in.tex_coord * 4096.0) + frame * GOLDEN;

    var irradiance = vec3<f32>(0.0);
    for (var r = 0u; r < num_rays; r = r + 1u) {
        let a = (f32(r) + jitter) / f32(num_rays) * TAU;
        irradiance += trace(world, vec2<f32>(cos(a), sin(a)));
    }
    irradiance /= f32(num_rays);

    if (history_valid > 0.5 && blend > 0.0) {
        let prev_vp = mat3x3<f32>(u.prev_vp_0.xyz, u.prev_vp_1.xyz, u.prev_vp_2.xyz);
        let clip = prev_vp * vec3<f32>(world, 1.0);
        let prev_uv = vec2<f32>((clip.x + 1.0) * 0.5, (1.0 - clip.y) * 0.5);
        if (prev_uv.x >= 0.0 && prev_uv.x <= 1.0 && prev_uv.y >= 0.0 && prev_uv.y <= 1.0) {
            let hist = textureSampleLevel(t_history, s_history, prev_uv, 0.0).rgb;
            irradiance = mix(irradiance, hist, blend);
        }
    }

    return vec4<f32>(irradiance, 1.0);
}
