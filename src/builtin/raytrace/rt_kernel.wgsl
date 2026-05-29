// Backend-independent path-tracing kernel. Appended after the preamble and an
// intersection snippet, so `trace_closest` / `trace_any` and all bindings are in
// scope. One sample per pixel per frame is accumulated as a running mean.

// ---- Random numbers (PCG hash) ------------------------------------------------

fn pcg_hash(x: u32) -> u32 {
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn init_rng(pixel_index: u32, seed: u32) -> u32 {
    return pcg_hash(pixel_index + pcg_hash(seed + 1u));
}

fn rand(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    return f32(*state) * (1.0 / 4294967296.0);
}

// ---- Sampling helpers ---------------------------------------------------------

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Duff et al. "Building an Orthonormal Basis, Revisited".
fn onb(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bt, n);
}

fn cosine_sample(n: vec3<f32>, r1: f32, r2: f32) -> vec3<f32> {
    let phi = 2.0 * PI * r1;
    let cos_t = sqrt(1.0 - r2);
    let sin_t = sqrt(r2);
    let local = vec3<f32>(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
    return normalize(onb(n) * local);
}

// Samples a GGX half-vector around `n` (NDF importance sampling).
fn ggx_sample_h(n: vec3<f32>, a: f32, r1: f32, r2: f32) -> vec3<f32> {
    let phi = 2.0 * PI * r1;
    let cos_t = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let local = vec3<f32>(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
    return normalize(onb(n) * local);
}

fn ggx_d(n_dot_h: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-7);
}

fn smith_g1(n_dot_x: f32, a: f32) -> f32 {
    let a2 = a * a;
    return 2.0 * n_dot_x / (n_dot_x + sqrt(a2 + (1.0 - a2) * n_dot_x * n_dot_x) + 1e-7);
}

fn smith_g(n_dot_v: f32, n_dot_l: f32, a: f32) -> f32 {
    return smith_g1(n_dot_v, a) * smith_g1(n_dot_l, a);
}

fn fresnel(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
}

// Full Cook-Torrance BRDF (already multiplied by N·L), for next-event estimation.
fn brdf_eval(
    n: vec3<f32>, wo: vec3<f32>, wi: vec3<f32>,
    albedo: vec3<f32>, metallic: f32, roughness: f32,
) -> vec3<f32> {
    let n_dot_l = dot(n, wi);
    let n_dot_v = dot(n, wo);
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = normalize(wo + wi);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(wo, h), 0.0);
    let a = max(roughness * roughness, 1e-3);

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f = fresnel(f0, v_dot_h);
    let d = ggx_d(n_dot_h, a);
    let g = smith_g(n_dot_v, n_dot_l, a);
    let spec = (d * g * f) / (4.0 * n_dot_v * n_dot_l + 1e-5);
    let diffuse = (vec3<f32>(1.0) - f) * (1.0 - metallic) * albedo / PI;
    return (diffuse + spec) * n_dot_l;
}

// ---- Environment --------------------------------------------------------------

fn sky(rd: vec3<f32>) -> vec3<f32> {
    let t = 0.5 * (rd.y + 1.0);
    let horizon = vec3<f32>(1.0, 1.0, 1.0);
    let zenith = vec3<f32>(0.5, 0.7, 1.0);
    return frame.ambient * mix(horizon, zenith, t);
}

// ---- Next-event estimation ----------------------------------------------------

fn sample_light(
    p: vec3<f32>, n: vec3<f32>, wo: vec3<f32>,
    albedo: vec3<f32>, metallic: f32, roughness: f32,
    rng: ptr<function, u32>,
) -> vec3<f32> {
    if (frame.num_lights == 0u) {
        return vec3<f32>(0.0);
    }
    let li = min(u32(rand(rng) * f32(frame.num_lights)), frame.num_lights - 1u);
    let light = lights[li];

    var wi: vec3<f32>;
    var dist: f32;
    var radiance: vec3<f32>;

    if (light.light_type == 1u) {
        // Directional.
        wi = normalize(-light.direction);
        dist = T_MAX;
        radiance = light.color * light.intensity;
    } else {
        let d = light.position - p;
        let dist2 = dot(d, d);
        dist = sqrt(dist2);
        wi = d / dist;
        let atten = 1.0 / max(dist2, 1e-4);
        let win = pow(clamp(1.0 - pow(dist / max(light.attenuation_radius, 1e-3), 4.0), 0.0, 1.0), 2.0);
        radiance = light.color * light.intensity * atten * win;
        if (light.light_type == 2u) {
            // Spot: cosine between the spot axis and the light->point direction.
            let cd = dot(normalize(light.direction), -wi);
            let s = clamp((cd - light.outer_cone_cos) / max(light.inner_cone_cos - light.outer_cone_cos, 1e-3), 0.0, 1.0);
            radiance = radiance * s * s;
        }
    }

    if (dot(n, wi) <= 0.0) {
        return vec3<f32>(0.0);
    }
    if (trace_any(p + n * EPS, wi, dist - 2.0 * EPS)) {
        return vec3<f32>(0.0);
    }

    let f = brdf_eval(n, wo, wi, albedo, metallic, roughness);
    // Compensate for picking 1 of N lights uniformly.
    return f * radiance * f32(frame.num_lights);
}

// ---- Main ---------------------------------------------------------------------

// Traces a single full path for pixel `gid` and returns its radiance.
fn sample_pixel(gid: vec3<u32>, rng: ptr<function, u32>) -> vec3<f32> {
    // Jittered camera ray, mirroring `Camera3d::unproject`.
    let jx = rand(rng);
    let jy = rand(rng);
    let ndc = vec2<f32>(
        2.0 * (f32(gid.x) + jx) / f32(frame.width) - 1.0,
        1.0 - 2.0 * (f32(gid.y) + jy) / f32(frame.height),
    );
    let near_h = frame.inv_view_proj * vec4<f32>(ndc, -1.0, 1.0);
    let far_h = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let near_p = near_h.xyz / near_h.w;
    let far_p = far_h.xyz / far_h.w;

    var ro = near_p;
    var rd = normalize(far_p - near_p);
    var throughput = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);

    for (var bounce = 0u; bounce < frame.max_bounces; bounce = bounce + 1u) {
        let hit = trace_closest(ro, rd, T_MAX);
        if (!hit.valid) {
            radiance = radiance + throughput * sky(rd);
            break;
        }

        let mat = materials[hit.material_id];
        radiance = radiance + throughput * mat.emissive.rgb;

        let p = ro + rd * hit.t;
        var n = hit.normal;
        let wo = -rd;
        if (dot(n, wo) < 0.0) {
            n = -n;
        }

        let albedo = mat.base_color.rgb;
        radiance = radiance + throughput * sample_light(p, n, wo, albedo, mat.metallic, mat.roughness, rng);

        // Choose a diffuse or specular lobe and importance-sample it.
        let f0 = mix(vec3<f32>(0.04), albedo, mat.metallic);
        let diffuse_color = albedo * (1.0 - mat.metallic);
        let lum_s = luminance(f0);
        let lum_d = luminance(diffuse_color);
        let p_spec = clamp(lum_s / (lum_s + lum_d + 1e-4), 0.1, 0.9);
        let a = max(mat.roughness * mat.roughness, 1e-3);

        var wi: vec3<f32>;
        if (rand(rng) < p_spec) {
            let h = ggx_sample_h(n, a, rand(rng), rand(rng));
            wi = reflect(rd, h);
            let n_dot_l = dot(n, wi);
            if (n_dot_l <= 0.0) {
                break;
            }
            let n_dot_v = max(dot(n, wo), 1e-4);
            let n_dot_h = max(dot(n, h), 1e-4);
            let v_dot_h = max(dot(wo, h), 1e-4);
            let f = fresnel(f0, v_dot_h);
            let g = smith_g(n_dot_v, n_dot_l, a);
            // BRDF * cos / pdf for NDF-sampled GGX simplifies to F*G*VoH/(NoV*NoH).
            throughput = throughput * (f * g * v_dot_h / (n_dot_v * n_dot_h)) / p_spec;
        } else {
            wi = cosine_sample(n, rand(rng), rand(rng));
            throughput = throughput * diffuse_color / (1.0 - p_spec);
        }

        ro = p + n * EPS;
        rd = wi;

        // Russian roulette after a few bounces.
        if (bounce >= 3u) {
            let q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.99);
            if (rand(rng) > q) {
                break;
            }
            throughput = throughput / q;
        }
    }

    return radiance;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= frame.width || gid.y >= frame.height) {
        return;
    }
    let pidx = gid.y * frame.width + gid.x;

    // Trace `samples_per_frame` paths this dispatch and accumulate their sum.
    let spp = max(frame.samples_per_frame, 1u);
    var radiance_sum = vec3<f32>(0.0);
    for (var s = 0u; s < spp; s = s + 1u) {
        var rng = init_rng(pidx, frame.seed + s * 9277u);
        radiance_sum = radiance_sum + sample_pixel(gid, &rng);
    }

    // Running mean: blend `spp` new samples into the `sample_index` already stored.
    let spp_f = f32(spp);
    var out_color: vec3<f32>;
    if (frame.sample_index == 0u) {
        out_color = radiance_sum / spp_f;
    } else {
        let s = f32(frame.sample_index);
        let prev = accum[pidx].rgb;
        out_color = (prev * s + radiance_sum) / (s + spp_f);
    }
    // Reject NaNs/Infs so a single bad sample can't poison the running mean.
    if (any(out_color != out_color) || any(out_color > vec3<f32>(1e20))) {
        out_color = vec3<f32>(0.0);
    }
    accum[pidx] = vec4<f32>(out_color, 1.0);
}
