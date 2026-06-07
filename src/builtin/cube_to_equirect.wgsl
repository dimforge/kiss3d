// Reprojects six captured cube faces (stored as layers of a 2D array) into one
// equirectangular map, used by runtime reflection-probe capture.
//
// The face layout is defined here AND in `renderer::reflection_probe`'s capture
// camera, so the two stay consistent by construction (no reliance on hardware
// cube-map face conventions): face `i` was rendered by a 90°-FOV right-handed
// camera looking along `FORWARD[i]` with up hint `UP[i]`. For an output direction
// `d` we find the face it falls in and reproject using the same basis the camera
// used (matching glam's `look_at_rh`).

@group(0) @binding(0) var faces: texture_2d_array<f32>;
@group(0) @binding(1) var samp: sampler;

const PI: f32 = 3.14159265359;

fn face_forward(i: u32) -> vec3<f32> {
    switch i {
        case 0u: { return vec3<f32>(1.0, 0.0, 0.0); }
        case 1u: { return vec3<f32>(-1.0, 0.0, 0.0); }
        case 2u: { return vec3<f32>(0.0, 1.0, 0.0); }
        case 3u: { return vec3<f32>(0.0, -1.0, 0.0); }
        case 4u: { return vec3<f32>(0.0, 0.0, 1.0); }
        default: { return vec3<f32>(0.0, 0.0, -1.0); }
    }
}

fn face_up(i: u32) -> vec3<f32> {
    switch i {
        case 2u: { return vec3<f32>(0.0, 0.0, -1.0); }
        case 3u: { return vec3<f32>(0.0, 0.0, 1.0); }
        default: { return vec3<f32>(0.0, 1.0, 0.0); }
    }
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let xy = corners[vid];
    var out: VsOut;
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    out.uv = vec2<f32>((xy.x + 1.0) * 0.5, (1.0 - xy.y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Equirect UV -> world direction (inverse of `ibl_dir_to_uv`).
    let theta = (in.uv.x - 0.5) * 2.0 * PI;
    let phi = in.uv.y * PI;
    let sp = sin(phi);
    let d = vec3<f32>(cos(theta) * sp, cos(phi), sin(theta) * sp);

    // Pick the face whose forward axis is most aligned with `d` and reproject.
    var best = 0u;
    var best_dot = -2.0;
    for (var i = 0u; i < 6u; i = i + 1u) {
        let dd = dot(d, face_forward(i));
        if dd > best_dot {
            best_dot = dd;
            best = i;
        }
    }
    let fwd = face_forward(best);
    let s = normalize(cross(fwd, face_up(best))); // camera right
    let u = cross(s, fwd); // camera up (recomputed, matches look_at_rh)
    let denom = max(dot(d, fwd), 1e-4);
    let ndc = vec2<f32>(dot(d, s) / denom, dot(d, u) / denom);
    let tex_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return textureSampleLevel(faces, samp, tex_uv, i32(best), 0.0);
}
