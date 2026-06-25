import package::common::unpack_mat3;
// Jump-flood step and resolve passes for the 2D occluder distance field.
//
// `fs_step` runs once per halving step size: each texel adopts, from itself and its
// eight neighbors at the current step offset, the seed whose stored world position
// is nearest — so after log2(size) passes every texel holds the nearest occluder
// seed. `fs_resolve` turns those seeds into a scalar world-space distance field that
// the GI march samples.

struct JfaUniforms {
    inv_vp_0: vec4<f32>,
    inv_vp_1: vec4<f32>,
    inv_vp_2: vec4<f32>,
    // step, size_x, size_y, _
    aux: vec4<f32>,
}

@group(0) @binding(0)
var t_seed: texture_2d<f32>;

@group(1) @binding(0)
var<uniform> u: JfaUniforms;

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

fn texel_world(coord: vec2<i32>, size: vec2<f32>) -> vec2<f32> {
    let uv = (vec2<f32>(coord) + 0.5) / size;
    let clip = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let inv = mat3x3<f32>(u.inv_vp_0.xyz, u.inv_vp_1.xyz, u.inv_vp_2.xyz);
    let h = inv * vec3<f32>(clip, 1.0);
    return h.xy / h.z;
}

@fragment
fn fs_step(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag.xy);
    let size = vec2<f32>(u.aux.y, u.aux.z);
    let isize = vec2<i32>(i32(u.aux.y), i32(u.aux.z));
    let step = i32(u.aux.x);
    let world_self = texel_world(coord, size);

    var best = textureLoad(t_seed, coord, 0);
    var best_d = select(1e18, distance(world_self, best.xy), best.z > 0.5);

    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let c = coord + vec2<i32>(dx, dy) * step;
            if (c.x < 0 || c.y < 0 || c.x >= isize.x || c.y >= isize.y) {
                continue;
            }
            let s = textureLoad(t_seed, c, 0);
            if (s.z > 0.5) {
                let d = distance(world_self, s.xy);
                if (d < best_d) {
                    best_d = d;
                    best = s;
                }
            }
        }
    }
    return best;
}

@fragment
fn fs_resolve(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag.xy);
    let size = vec2<f32>(u.aux.y, u.aux.z);
    let world_self = texel_world(coord, size);

    let s = textureLoad(t_seed, coord, 0);
    var dist = 1e9;
    if (s.z > 0.5) {
        dist = distance(world_self, s.xy);
    }
    return vec4<f32>(dist, 0.0, 0.0, 0.0);
}
