import package::common::unpack_mat3;
// One level of 2D Radiance Cascades (decoupled layout).
//
// Each cascade is a texture packing a grid of probes; each probe owns an `e x e` tile
// of texels, one per ray direction. Probe spacing `s` and direction-tile edge `e`
// scale independently per level (s doubles, e doubles → direction count quadruples),
// so the cascade texture stays the same size while spacing and angular resolution are
// chosen independently — letting us keep a fine probe grid AND many directions. A
// texel marches its probe's ray over a radial interval against the emitter/occluder
// discs, then merges the higher cascade's continuation (the 4 finer-angle directions,
// bilinearly interpolated across the 4 nearest upper probes at this probe's position).

const MAX_EMITTERS: u32 = 32u;
const MAX_OCCLUDERS: u32 = 64u;
const TAU: f32 = 6.2831853;

struct Emitter {
    pos_radius: vec4<f32>,
    color: vec4<f32>,
}

struct Occluder {
    pos_radius: vec4<f32>,
}

// Reused from gi2d_field.wgsl (the same field uniform buffer is bound).
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
    params: vec4<f32>,
    flags: vec4<f32>,
    counts: vec4<f32>,
    emitters: array<Emitter, MAX_EMITTERS>,
    occluders: array<Occluder, MAX_OCCLUDERS>,
}

struct CascadeParams {
    // dir-tile edge e_c, probe spacing s_c (field px), fieldW, fieldH
    v0: vec4<f32>,
    // upper dir-tile edge e_up, upper spacing s_up, up_probesX, up_probesY
    v1: vec4<f32>,
    // start_c, end_c, max_steps, is_top
    v2: vec4<f32>,
}

@group(0) @binding(0)
var t_upper: texture_2d<f32>;

@group(1) @binding(0)
var<uniform> p: CascadeParams;

@group(2) @binding(0)
var<uniform> field: FieldUniforms;

@group(3) @binding(0)
var t_sdf: texture_2d<f32>;
@group(3) @binding(1)
var s_sdf: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(vertex.position, 0.0, 1.0);
    return out;
}

fn unproject(uv: vec2<f32>) -> vec2<f32> {
    let clip = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let inv = mat3x3<f32>(field.inv_vp_0.xyz, field.inv_vp_1.xyz, field.inv_vp_2.xyz);
    let h = inv * vec3<f32>(clip, 1.0);
    return h.xy / h.z;
}

// Distance to the nearest occluder at world point `q`: a jump-flooded SDF texture
// fetch when `use_sdf`, otherwise the analytic minimum over the occluder discs.
fn occluder_distance(q: vec2<f32>) -> f32 {
    if (field.flags.x > 0.5) {
        let clip = mat3x3<f32>(field.cur_vp_0.xyz, field.cur_vp_1.xyz, field.cur_vp_2.xyz)
            * vec3<f32>(q, 1.0);
        let uv = vec2<f32>((clip.x + 1.0) * 0.5, (1.0 - clip.y) * 0.5);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            return 1e9;
        }
        // Bias the unsigned field so it crosses zero just outside the surface.
        return textureSampleLevel(t_sdf, s_sdf, uv, 0.0).r - field.flags.y;
    }
    let num_o = u32(field.counts.y);
    var d = 1e9;
    for (var i = 0u; i < num_o; i = i + 1u) {
        let o = field.occluders[i];
        d = min(d, length(q - o.pos_radius.xy) - o.pos_radius.z);
    }
    return d;
}

// March a ray over [start, end): returns (radiance.rgb, transmittance).
fn march_interval(origin: vec2<f32>, dir: vec2<f32>, start: f32, end: f32, max_steps: u32) -> vec4<f32> {
    let num_e = u32(field.counts.x);

    var t = start;
    for (var step = 0u; step < max_steps; step = step + 1u) {
        let q = origin + dir * t;

        var d_emit = 1e9;
        var ei = 0u;
        for (var i = 0u; i < num_e; i = i + 1u) {
            let e = field.emitters[i];
            let d = length(q - e.pos_radius.xy) - e.pos_radius.z;
            if (d < d_emit) {
                d_emit = d;
                ei = i;
            }
        }
        let d_occ = occluder_distance(q);

        if (d_emit <= 0.0) {
            let e = field.emitters[ei];
            return vec4<f32>(e.color.rgb * e.pos_radius.w, 0.0);
        }
        if (d_occ <= 0.0) {
            return vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }

        t += max(min(d_emit, d_occ), 0.5);
        if (t >= end) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

fn read_upper(probe: vec2<i32>, dir: i32, e: i32, probes: vec2<i32>) -> vec4<f32> {
    let pc = clamp(probe, vec2<i32>(0), probes - vec2<i32>(1));
    let coord = pc * e + vec2<i32>(dir % e, dir / e);
    return textureLoad(t_upper, coord, 0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag.xy);
    let e_c = i32(p.v0.x);
    let s_c = p.v0.y;
    let field_size = p.v0.zw;

    let probe = coord / e_c;
    let within = coord - probe * e_c;
    let dir = within.y * e_c + within.x;
    let dir_count = e_c * e_c;

    let probe_px = (vec2<f32>(probe) + 0.5) * s_c;
    let origin = unproject(probe_px / field_size);
    let angle = TAU * (f32(dir) + 0.5) / f32(dir_count);
    let rdir = vec2<f32>(cos(angle), sin(angle));

    let near = march_interval(origin, rdir, p.v2.x, p.v2.y, u32(p.v2.z));

    // Top cascade: nothing above to merge.
    if (p.v2.w > 0.5 || near.a <= 0.0) {
        return near;
    }

    let e_up = i32(p.v1.x);
    let s_up = p.v1.y;
    let up_probes = vec2<i32>(i32(p.v1.z), i32(p.v1.w));
    // This probe's position in the upper probe grid (both in field pixels).
    let pf = probe_px / s_up - vec2<f32>(0.5);
    let p0 = vec2<i32>(floor(pf));
    let fr = pf - floor(pf);
    let w00 = (1.0 - fr.x) * (1.0 - fr.y);
    let w10 = fr.x * (1.0 - fr.y);
    let w01 = (1.0 - fr.x) * fr.y;
    let w11 = fr.x * fr.y;

    var cont = vec4<f32>(0.0);
    for (var k = 0; k < 4; k = k + 1) {
        let ud = dir * 4 + k;
        cont += w00 * read_upper(p0 + vec2<i32>(0, 0), ud, e_up, up_probes);
        cont += w10 * read_upper(p0 + vec2<i32>(1, 0), ud, e_up, up_probes);
        cont += w01 * read_upper(p0 + vec2<i32>(0, 1), ud, e_up, up_probes);
        cont += w11 * read_upper(p0 + vec2<i32>(1, 1), ud, e_up, up_probes);
    }
    cont *= 0.25;

    return vec4<f32>(near.rgb + near.a * cont.rgb, near.a * cont.a);
}
