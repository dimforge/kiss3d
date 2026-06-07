// Screen-space reflections — single additive pass.
//
// For each glossy pixel it reconstructs the view-space reflection ray, marches it
// in screen space (DDA with perspective-correct depth + binary-search refinement),
// and on a screen hit samples the (roughness-blurred) scene color. To avoid
// double-counting the environment specular the forward pass already added, it
// writes the *delta* `(ssr - env) * BRDF * confidence` and the pipeline blends it
// additively (One, One) into the scene, with a COLOR write mask so alpha is
// untouched. Where the ray misses, the delta is zero and the forward pass's
// environment/probe specular remains. See `ssr.rs`.

// Shared equirectangular mapping + analytic env-BRDF (same as the default material).
import package::pbr_env::{equirect_dir_to_uv, env_brdf_approx};
import package::common::{fullscreen_triangle_xy, fullscreen_uv_from_clip};

struct SsrUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    // (inv_res.x, inv_res.y, max_steps, thickness)
    params0: vec4<f32>,
    // (infinite_thick, max_distance, roughness_cutoff, edge_fade)
    params1: vec4<f32>,
    // (ibl_has, ibl_max_lod, ibl_intensity, ibl_rotation)
    ibl: vec4<f32>,
    // (refl_max_lod, user_intensity, distance_attenuation, fresnel)
    misc: vec4<f32>,
}

@group(0) @binding(0) var t_viewpos: texture_2d<f32>;
@group(0) @binding(1) var t_normal: texture_2d<f32>;
@group(0) @binding(2) var t_material: texture_2d<f32>;
@group(0) @binding(3) var t_refl: texture_2d<f32>;
@group(0) @binding(4) var t_env: texture_2d<f32>;
// Per-object SSR params from the prepass: (intensity, infinite_thick,
// distance_attenuation, fresnel). intensity 0 = object receives no SSR.
@group(0) @binding(5) var t_ssr: texture_2d<f32>;
@group(0) @binding(6) var samp: sampler;
@group(0) @binding(7) var<uniform> u: SsrUniforms;

fn project_uv(vp: vec3<f32>) -> vec2<f32> {
    let clip = u.proj * vec4<f32>(vp, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn ibl_rotate(rd: vec3<f32>, rot: f32) -> vec3<f32> {
    let c = cos(rot);
    let s = sin(rot);
    return vec3<f32>(c * rd.x + s * rd.z, rd.y, -s * rd.x + c * rd.z);
}

fn env_sample(dir: vec3<f32>, lod: f32) -> vec3<f32> {
    return textureSampleLevel(t_env, samp, equirect_dir_to_uv(ibl_rotate(dir, u.ibl.w)), lod).rgb;
}

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let vp = textureSampleLevel(t_viewpos, samp, uv, 0.0);
    if vp.a < 0.5 {
        return vec4<f32>(0.0); // background: no surface, add nothing
    }
    let nr = textureSampleLevel(t_normal, samp, uv, 0.0);
    let roughness = nr.a;
    let cutoff = u.params1.z;
    if roughness > cutoff {
        return vec4<f32>(0.0); // too rough for plausible SSR
    }
    let f0 = textureSampleLevel(t_material, samp, uv, 0.0).rgb;

    // Per-object SSR params (gate + flags). intensity 0 => this object opted out.
    let obj = textureSampleLevel(t_ssr, samp, uv, 0.0);
    let obj_intensity = obj.x;
    if obj_intensity <= 0.0 {
        return vec4<f32>(0.0);
    }
    let infinite_thick = obj.y > 0.5;
    let dist_atten = obj.z > 0.5;
    let fresnel_on = obj.w > 0.5;

    let p = vp.xyz; // view-space position
    let n_world = normalize(nr.xyz);
    let view_rot = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    let n = normalize(view_rot * n_world);
    let v = normalize(-p);
    let nov = max(dot(n, v), 1e-4);
    let r = reflect(-v, n); // view-space reflection ray

    // Screen-space DDA march. Project the reflection ray's start and (near-plane-
    // clipped) end into screen space, then step in uniform pixel increments,
    // reconstructing a perspective-correct view-space depth at each step (1/z is
    // linear in screen space). Uniform screen-space sampling avoids the banding a
    // fixed view-space stride causes; a binary search refines the crossing.
    let max_steps = i32(u.params0.z);
    let thickness = u.params0.w;
    let max_dist = u.params1.y;

    var hit_uv = vec2<f32>(0.0);
    var hit = false;

    // Clip the ray just in front of the near plane; rays pointing back toward the
    // camera (r.z > 0) shrink t_max toward 0 and fade out (SSR can't trace them).
    var t_max = max_dist;
    if r.z > 0.0 {
        t_max = min(t_max, (-1e-2 - p.z) / r.z);
    }
    if t_max > 1e-3 {
        let res = vec2<f32>(textureDimensions(t_viewpos));
        let uv0 = uv; // == project_uv(p)
        let p1 = p + r * t_max;
        let uv1 = project_uv(p1);

        // Clip the screen segment to the viewport.
        let dir = uv1 - uv0;
        var s_exit = 1.0;
        if abs(dir.x) > 1e-6 {
            s_exit = min(s_exit, select((0.0 - uv0.x) / dir.x, (1.0 - uv0.x) / dir.x, dir.x > 0.0));
        }
        if abs(dir.y) > 1e-6 {
            s_exit = min(s_exit, select((0.0 - uv0.y) / dir.y, (1.0 - uv0.y) / dir.y, dir.y > 0.0));
        }
        s_exit = clamp(s_exit, 0.0, 1.0);

        let pix_len = distance(uv0 * res, mix(uv0, uv1, s_exit) * res);
        let steps = clamp(pix_len, 1.0, f32(max_steps));
        let num = i32(steps);
        let invz0 = 1.0 / p.z;
        let invz1 = 1.0 / p1.z;
        // Stable per-pixel jitter dithers residual aliasing.
        let jitter = hash12(uv0 * res);

        var prev_s = 0.0;
        var prev_in_front = false;
        for (var i = 1; i <= num; i = i + 1) {
            let s = clamp(s_exit * (f32(i) - jitter) / steps, 0.0, s_exit);
            let suv = mix(uv0, uv1, s);
            let ray_z = 1.0 / mix(invz0, invz1, s); // perspective-correct depth
            let svp = textureSampleLevel(t_viewpos, samp, suv, 0.0);
            var in_front = false;
            if svp.a >= 0.5 {
                let d = ray_z - svp.z; // > 0 in front of the surface, < 0 behind it
                if d >= 0.0 {
                    in_front = true;
                } else if prev_in_front && (infinite_thick || d > -thickness) {
                    var lo = prev_s;
                    var hi = s;
                    for (var k = 0; k < 8; k = k + 1) {
                        let mid = 0.5 * (lo + hi);
                        let mz = 1.0 / mix(invz0, invz1, mid);
                        let md = mz - textureSampleLevel(t_viewpos, samp, mix(uv0, uv1, mid), 0.0).z;
                        if md < 0.0 {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    hit_uv = mix(uv0, uv1, hi);
                    hit = true;
                    break;
                }
            }
            prev_in_front = in_front;
            prev_s = s;
        }
    }

    // Environment fallback in the world-space reflection direction.
    let r_world = transpose(view_rot) * r;
    var env_col = vec3<f32>(0.0);
    if u.ibl.x > 0.5 {
        env_col = env_sample(r_world, roughness * u.ibl.y) * u.ibl.z;
    }

    var refl_col = env_col;
    var conf = 0.0;
    if hit {
        refl_col = textureSampleLevel(t_refl, samp, hit_uv, roughness * u.misc.x).rgb;
        let edge = max(u.params1.w, 1e-3);
        let fx = smoothstep(0.0, edge, hit_uv.x) * smoothstep(0.0, edge, 1.0 - hit_uv.x);
        let fy = smoothstep(0.0, edge, hit_uv.y) * smoothstep(0.0, edge, 1.0 - hit_uv.y);
        let rough_fade = 1.0 - smoothstep(cutoff * 0.7, cutoff, roughness);
        // Optional (per-object) distance² falloff (fades far, less-reliable hits).
        var dfade = 1.0;
        if dist_atten {
            let hit_pos = textureSampleLevel(t_viewpos, samp, hit_uv, 0.0).xyz;
            let d = clamp(1.0 - distance(hit_pos, p) / max_dist, 0.0, 1.0);
            dfade = d * d;
        }
        conf = fx * fy * rough_fade * dfade;
    }

    // Optional (per-object) grazing Fresnel boost, on top of the BRDF.
    var fresnel = 1.0;
    if fresnel_on {
        fresnel = (dot(-v, r) + 1.0) * 0.5;
    }

    // Additive delta: replace the environment specular with the SSR hit where
    // confident (no double counting), scaled by the global + per-object intensity.
    let brdf = env_brdf_approx(f0, roughness, nov);
    let delta = (refl_col - env_col) * brdf * conf * fresnel * u.misc.y * obj_intensity;
    return vec4<f32>(delta, 0.0);
}
