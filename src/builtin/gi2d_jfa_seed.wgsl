import package::common::{unpack_mat3, fullscreen_uv_from_clip};
// Jump-flood seed pass for the 2D occluder distance field. Each low-res texel is
// classified against the analytic occluder discs: texels inside any occluder become
// seeds storing their own world position (xy) with a valid flag (z = 1); empty
// texels store z = 0. The jump-flood passes then propagate the nearest seed.

const MAX_EMITTERS: u32 = 32u;
const MAX_OCCLUDERS: u32 = 64u;

struct Emitter {
    pos_radius: vec4<f32>,
    color: vec4<f32>,
}

struct Occluder {
    pos_radius: vec4<f32>,
}

// Same layout as FieldUniforms in gi2d_field.wgsl (the field uniform buffer is reused).
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

@group(0) @binding(0)
var<uniform> u: FieldUniforms;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let world = unproject(in.tex_coord);
    let num_occluders = u32(u.counts.y);

    var inside = false;
    for (var i = 0u; i < num_occluders; i = i + 1u) {
        let o = u.occluders[i];
        if (length(world - o.pos_radius.xy) <= o.pos_radius.z) {
            inside = true;
            break;
        }
    }

    if (inside) {
        return vec4<f32>(world, 1.0, 0.0);
    }
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
