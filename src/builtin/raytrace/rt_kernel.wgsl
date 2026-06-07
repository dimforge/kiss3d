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

// Power heuristic (beta = 2) used for multiple-importance sampling.
fn power_heuristic(pf: f32, pg: f32) -> f32 {
    let f2 = pf * pf;
    let g2 = pg * pg;
    return f2 / (f2 + g2 + 1e-8);
}

// Samples a unit disk uniformly (for the thin-lens camera).
fn sample_disk(r1: f32, r2: f32) -> vec2<f32> {
    let r = sqrt(r1);
    let theta = 2.0 * PI * r2;
    return vec2<f32>(r * cos(theta), r * sin(theta));
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

fn fresnel_schlick(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
}

// Exact dielectric Fresnel reflectance for unpolarised light.
// `cos_i` is |cos(theta_i)|, `eta` is n_i / n_t (incident over transmitted).
fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    let sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
    if (sin_t2 >= 1.0) {
        return 1.0; // total internal reflection
    }
    let cos_t = sqrt(1.0 - sin_t2);
    let rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5 * (rs * rs + rp * rp);
}

// ---- Material resolution (textures) -------------------------------------------

// Resolved per-hit shading parameters after texture lookups.
struct Surface {
    albedo: vec3<f32>,
    emissive: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ior: f32,
    transmission: f32,
    spec_tint: vec3<f32>,
    bsdf_type: u32,
    subsurface: f32,
    subsurface_radius: f32,
};

fn sample_tex(layer: i32, uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(tex_array, tex_sampler, uv, layer, 0.0);
}

fn resolve_surface(mat: RtMaterial, uv: vec2<f32>) -> Surface {
    var s: Surface;
    s.albedo = mat.base_color.rgb;
    s.emissive = mat.emissive.rgb;
    s.metallic = mat.metallic;
    s.roughness = mat.roughness;
    s.ior = mat.ior;
    s.transmission = mat.transmission;
    s.spec_tint = mat.specular_tint;
    s.bsdf_type = mat.bsdf_type;
    s.subsurface = mat.subsurface;
    s.subsurface_radius = mat.subsurface_radius;

    if (mat.albedo_tex >= 0) {
        let t = sample_tex(mat.albedo_tex, uv);
        s.albedo = s.albedo * t.rgb;
    }
    if (mat.mr_tex >= 0) {
        // glTF convention: G = roughness, B = metallic.
        let t = sample_tex(mat.mr_tex, uv);
        s.roughness = s.roughness * t.g;
        s.metallic = s.metallic * t.b;
    }
    if (mat.emissive_tex >= 0) {
        let t = sample_tex(mat.emissive_tex, uv);
        s.emissive = s.emissive * t.rgb;
    }
    return s;
}

// Perturbs the shading normal `n` by the tangent-space normal map, building a
// crude tangent frame from the geometric ONB (no per-vertex tangents available).
fn apply_normal_map(mat: RtMaterial, n: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    if (mat.normal_tex < 0) {
        return n;
    }
    let t = sample_tex(mat.normal_tex, uv).xyz * 2.0 - vec3<f32>(1.0);
    let frame = onb(n);
    return normalize(frame * t);
}

// ---- BSDF evaluation (for NEE / MIS) ------------------------------------------

// Reflective Cook-Torrance lobe value (already multiplied by N·L), with its pdf
// returned in `pdf` (solid-angle, for the GGX-sampled half vector). Diffuse +
// specular for the opaque/metal lobes; transmission is sampled stochastically and
// is not included here.
fn brdf_eval(
    s: Surface, n: vec3<f32>, wo: vec3<f32>, wi: vec3<f32>, pdf: ptr<function, f32>,
) -> vec3<f32> {
    *pdf = 0.0;
    let n_dot_l = dot(n, wi);
    let n_dot_v = dot(n, wo);
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = normalize(wo + wi);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(wo, h), 0.0);
    let a = max(s.roughness * s.roughness, 1e-3);

    var f0: vec3<f32>;
    if (s.bsdf_type == BSDF_METAL) {
        f0 = mix(vec3<f32>(0.04), s.albedo, s.metallic) * s.spec_tint;
    } else {
        f0 = mix(vec3<f32>(0.04) * s.spec_tint, s.albedo, s.metallic);
    }
    let fr = fresnel_schlick(f0, v_dot_h);
    let d = ggx_d(n_dot_h, a);
    let g = smith_g(n_dot_v, n_dot_l, a);
    let spec = (d * g * fr) / (4.0 * n_dot_v * n_dot_l + 1e-5);

    var diffuse = vec3<f32>(0.0);
    if (s.bsdf_type != BSDF_METAL) {
        let kd = (vec3<f32>(1.0) - fr) * (1.0 - s.metallic) * (1.0 - s.transmission);
        // Cheap subsurface: lerp toward a wrapped/half-Lambert term.
        let wrap = clamp((n_dot_l + s.subsurface) / (1.0 + s.subsurface), 0.0, 1.0);
        let diff_shade = mix(n_dot_l, wrap, s.subsurface);
        diffuse = kd * s.albedo / PI * diff_shade;
        // Re-fold the cosine that the spec term carries explicitly below.
        diffuse = diffuse / max(n_dot_l, 1e-4);
    }

    // pdf: weighted average of the diffuse (cosine) and specular (GGX) pdfs.
    let lum_s = luminance(f0);
    let lum_d = luminance(s.albedo * (1.0 - s.metallic));
    let p_spec = clamp(lum_s / (lum_s + lum_d + 1e-4), 0.1, 0.9);
    let pdf_spec = d * n_dot_h / (4.0 * v_dot_h + 1e-5);
    let pdf_diff = n_dot_l / PI;
    *pdf = p_spec * pdf_spec + (1.0 - p_spec) * pdf_diff;

    return (diffuse + spec) * n_dot_l;
}

// ---- Environment --------------------------------------------------------------

// Rotates a direction around Y by the env_rotation (cos, sin).
fn env_rotate(rd: vec3<f32>) -> vec3<f32> {
    let c = frame.env_rotation.x;
    let sn = frame.env_rotation.y;
    return vec3<f32>(c * rd.x + sn * rd.z, rd.y, -sn * rd.x + c * rd.z);
}

fn dir_to_equirect(rd: vec3<f32>) -> vec2<f32> {
    let d = env_rotate(rd);
    let u = atan2(d.z, d.x) / (2.0 * PI) + 0.5;
    let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2<f32>(u, v);
}

// Radiance seen where a ray escapes the scene: the HDRI environment map if one is
// bound, otherwise the flat background color. The env map is a real light source
// (used by every ray); the flat background is cosmetic (the caller only adds it
// along directly-seen paths — see the miss handler in `sample_pixel`).
fn sky(rd: vec3<f32>) -> vec3<f32> {
    if (frame.has_env != 0u) {
        let uv = dir_to_equirect(rd);
        let c = textureSampleLevel(env_tex, env_sampler, uv, 0.0).rgb;
        return c * frame.env_rotation.z;
    }
    return frame.background.rgb;
}

// Solid-angle pdf of sampling the environment by direction (uniform-sphere
// fallback; the CDF importance sampling shares the same pdf basis for MIS).
fn env_pdf(rd: vec3<f32>) -> f32 {
    if (frame.has_env == 0u) {
        return 0.0;
    }
    return 1.0 / (4.0 * PI);
}

// ---- Next-event estimation: analytic lights -----------------------------------

struct LightSample {
    wi: vec3<f32>,
    dist: f32,
    radiance: vec3<f32>, // already includes 1/pdf-style weighting for analytic lights
    pdf: f32,            // solid-angle pdf (0 for delta lights)
    valid: bool,
};

fn sample_analytic_light(light: RtLight, p: vec3<f32>, rng: ptr<function, u32>) -> LightSample {
    var ls: LightSample;
    ls.valid = false;
    ls.pdf = 0.0;

    if (light.light_type == 1u) {
        // Directional (delta).
        ls.wi = normalize(-light.direction);
        ls.dist = T_MAX;
        ls.radiance = light.color * light.intensity;
        ls.valid = true;
        return ls;
    }

    // Point/spot, optionally with a finite radius → sphere (soft shadows).
    var lpos = light.position;
    if (light.radius > 0.0) {
        // Sample a point on the sphere surface (uniform, cheap).
        let z = 1.0 - 2.0 * rand(rng);
        let r = sqrt(max(0.0, 1.0 - z * z));
        let phi = 2.0 * PI * rand(rng);
        lpos = light.position + light.radius * vec3<f32>(r * cos(phi), r * sin(phi), z);
    }

    let d = lpos - p;
    let dist2 = dot(d, d);
    ls.dist = sqrt(dist2);
    ls.wi = d / ls.dist;
    // Match the rasterizer's `calculate_point_attenuation`: a smooth window that
    // reaches zero at `attenuation_radius`, with NO inverse-square term. kiss3d's
    // light `intensity` is defined by that artistic model (the rasterizer is the
    // reference), so using physical 1/d² falloff here would make the same point
    // light look ~orders of magnitude dimmer under the path tracer.
    let nd = ls.dist / max(light.attenuation_radius, 1e-3);
    let win = clamp(1.0 - nd * nd, 0.0, 1.0);
    let atten = win * win;
    ls.radiance = light.color * light.intensity * atten;
    if (light.light_type == 2u) {
        let cd = dot(normalize(light.direction), -ls.wi);
        let sp = clamp((cd - light.outer_cone_cos) / max(light.inner_cone_cos - light.outer_cone_cos, 1e-3), 0.0, 1.0);
        ls.radiance = ls.radiance * sp * sp;
    }
    ls.valid = true;
    return ls;
}

// ---- Next-event estimation: emissive triangles --------------------------------

fn tri_positions(e: RtEmitter) -> mat3x3<f32> {
    return mat3x3<f32>(e.p0, e.p1, e.p2);
}

// Direct lighting from one randomly chosen analytic light, returning radiance
// already weighted by the BSDF and MIS against the BSDF-sampling strategy.
fn sample_lights(
    s: Surface, p: vec3<f32>, n: vec3<f32>, wo: vec3<f32>, rng: ptr<function, u32>,
) -> vec3<f32> {
    var result = vec3<f32>(0.0);

    // --- Analytic lights ---
    if (frame.num_lights > 0u) {
        let li = min(u32(rand(rng) * f32(frame.num_lights)), frame.num_lights - 1u);
        let light = lights[li];
        let ls = sample_analytic_light(light, p, rng);
        if (ls.valid && dot(n, ls.wi) > 0.0 &&
            !trace_any(p + n * EPS, ls.wi, ls.dist - 2.0 * EPS)) {
            var bpdf: f32;
            let f = brdf_eval(s, n, wo, ls.wi, &bpdf);
            // Delta lights: no MIS (pdf is infinite). Multiply by N to undo the
            // 1-of-N uniform light pick.
            result += f * ls.radiance * f32(frame.num_lights);
        }
    }

    // --- Emissive triangles (area lights) with MIS ---
    if (frame.num_emitters > 0u) {
        let ei = min(u32(rand(rng) * f32(frame.num_emitters)), frame.num_emitters - 1u);
        let e = emitters[ei];
        let pos = tri_positions(e);
        // Sample a point on the triangle uniformly by area.
        var u = rand(rng);
        var v = rand(rng);
        if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
        let lp = pos[0] + (pos[1] - pos[0]) * u + (pos[2] - pos[0]) * v;
        let e1 = pos[1] - pos[0];
        let e2 = pos[2] - pos[0];
        let ng_raw = cross(e1, e2);
        let area = 0.5 * length(ng_raw);
        var ln = normalize(ng_raw);

        let d = lp - p;
        let dist2 = dot(d, d);
        let dist = sqrt(dist2);
        let wi = d / dist;

        // Two-sided emitter: flip the light normal toward the shading point.
        if (dot(ln, -wi) < 0.0) { ln = -ln; }
        let cos_l = dot(ln, -wi);

        if (dot(n, wi) > 0.0 && cos_l > 1e-4 && area > 1e-8 &&
            !trace_any(p + n * EPS, wi, dist - 2.0 * EPS)) {
            // Area pdf -> solid-angle pdf, averaged over the emitter count.
            let pdf_area = 1.0 / (area * f32(frame.num_emitters));
            let pdf_sa = pdf_area * dist2 / cos_l;
            var bpdf: f32;
            let f = brdf_eval(s, n, wo, wi, &bpdf);
            let le = e.emission;
            let w = power_heuristic(pdf_sa, bpdf);
            result += f * le * w / max(pdf_sa, 1e-6);
        }
    }

    return result;
}

// Returns the solid-angle pdf of having hit emitter triangle `tri_index` from
// point `p` along `wi` (for MIS weighting of BSDF-sampled emitter hits).
fn emitter_pdf(p: vec3<f32>, wi: vec3<f32>, hit_p: vec3<f32>, ln: vec3<f32>, area: f32) -> f32 {
    let dist2 = dot(hit_p - p, hit_p - p);
    let cos_l = abs(dot(ln, -wi));
    if (cos_l < 1e-4 || area < 1e-8) {
        return 0.0;
    }
    let pdf_area = 1.0 / (area * f32(max(frame.num_emitters, 1u)));
    return pdf_area * dist2 / cos_l;
}

// ---- BSDF sampling ------------------------------------------------------------

struct BsdfSample {
    wi: vec3<f32>,
    throughput: vec3<f32>, // f * cos / pdf
    pdf: f32,              // solid-angle pdf (0 = delta / specular)
    specular: bool,
    transmitted: bool,
    valid: bool,
};

// Samples the unified BSDF at the surface. `n` is the shading normal already
// oriented to the same side as `wo`; `entering` tells whether `rd` enters the
// medium (dot(geom_normal, rd) < 0).
fn sample_bsdf(
    s: Surface, n: vec3<f32>, wo: vec3<f32>, rd: vec3<f32>, entering: bool,
    rng: ptr<function, u32>,
) -> BsdfSample {
    var bs: BsdfSample;
    bs.valid = false;
    bs.specular = false;
    bs.transmitted = false;
    bs.pdf = 0.0;
    let a = max(s.roughness * s.roughness, 1e-3);

    // --- Glass / dielectric: reflect or refract via Fresnel ---
    if (s.bsdf_type == BSDF_GLASS || s.transmission > 0.0) {
        let eta_i = select(s.ior, 1.0, entering);
        let eta_t = select(1.0, s.ior, entering);
        let eta = eta_i / eta_t;

        // Microfacet normal: smooth glass uses the shading normal.
        var m = n;
        if (a > 1e-3) {
            m = ggx_sample_h(n, a, rand(rng), rand(rng));
        }
        let cos_i = clamp(dot(wo, m), 0.0, 1.0);
        let fr = fresnel_dielectric(cos_i, eta);

        if (rand(rng) < fr) {
            // Reflection.
            bs.wi = reflect(-wo, m);
            if (dot(bs.wi, n) <= 0.0) { return bs; }
            bs.transmitted = false;
        } else {
            // Refraction (Snell). `refract` expects the incident dir = -wo.
            bs.wi = refract(-wo, m, eta);
            if (dot(bs.wi, bs.wi) < 1e-6) {
                // Total internal reflection fallback.
                bs.wi = reflect(-wo, m);
                bs.transmitted = false;
            } else {
                bs.transmitted = true;
            }
        }
        bs.specular = (a <= 1e-3);
        // Smooth glass is treated as a delta lobe (pdf folded into throughput).
        var tint = vec3<f32>(1.0);
        if (bs.transmitted) {
            tint = s.albedo;
        }
        bs.throughput = tint;
        bs.valid = true;
        return bs;
    }

    // --- Opaque / metal: diffuse + GGX specular ---
    var f0: vec3<f32>;
    if (s.bsdf_type == BSDF_METAL) {
        f0 = mix(vec3<f32>(0.04), s.albedo, s.metallic) * s.spec_tint;
    } else {
        f0 = mix(vec3<f32>(0.04) * s.spec_tint, s.albedo, s.metallic);
    }
    let diffuse_color = s.albedo * (1.0 - s.metallic);
    let lum_s = luminance(f0);
    let lum_d = luminance(diffuse_color);
    var p_spec = clamp(lum_s / (lum_s + lum_d + 1e-4), 0.1, 0.9);
    if (s.bsdf_type == BSDF_METAL) {
        p_spec = 1.0;
    }

    if (rand(rng) < p_spec) {
        let h = ggx_sample_h(n, a, rand(rng), rand(rng));
        bs.wi = reflect(-wo, h);
        let n_dot_l = dot(n, bs.wi);
        if (n_dot_l <= 0.0) { return bs; }
        let n_dot_v = max(dot(n, wo), 1e-4);
        let n_dot_h = max(dot(n, h), 1e-4);
        let v_dot_h = max(dot(wo, h), 1e-4);
        let fr = fresnel_schlick(f0, v_dot_h);
        let g = smith_g(n_dot_v, n_dot_l, a);
        // f*cos/pdf for NDF-sampled GGX simplifies to F*G*VoH/(NoV*NoH).
        bs.throughput = (fr * g * v_dot_h / (n_dot_v * n_dot_h)) / p_spec;
        let d = ggx_d(n_dot_h, a);
        bs.pdf = p_spec * d * n_dot_h / (4.0 * v_dot_h);
    } else {
        bs.wi = cosine_sample(n, rand(rng), rand(rng));
        let n_dot_l = max(dot(n, bs.wi), 0.0);
        // Subsurface wrap shading folded into the diffuse throughput.
        let wrap = clamp((n_dot_l + s.subsurface) / (1.0 + s.subsurface), 0.0, 1.0);
        let shade = mix(1.0, wrap / max(n_dot_l, 1e-4), s.subsurface);
        bs.throughput = diffuse_color * (1.0 - s.transmission) * shade / (1.0 - p_spec);
        bs.pdf = (1.0 - p_spec) * n_dot_l / PI;
    }
    bs.valid = true;
    return bs;
}

// ---- Main ---------------------------------------------------------------------

// Output of a path trace, carrying first-hit guide data for the denoiser.
struct PathResult {
    radiance: vec3<f32>,
    first_albedo: vec3<f32>,
    first_normal: vec3<f32>,
    has_first: bool,
};

// Traces a single full path for pixel `gid` and returns its radiance + guides.
fn sample_pixel(gid: vec3<u32>, rng: ptr<function, u32>) -> PathResult {
    var res: PathResult;
    res.radiance = vec3<f32>(0.0);
    res.first_albedo = vec3<f32>(0.0);
    res.first_normal = vec3<f32>(0.0);
    res.has_first = false;

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

    // Thin-lens depth of field: jitter the origin over the lens and re-aim
    // through the focus plane.
    if (frame.lens_radius > 0.0) {
        let focus_p = ro + rd * frame.focus_distance;
        let lens = sample_disk(rand(rng), rand(rng)) * frame.lens_radius;
        let basis = onb(rd);
        let offset = basis[0] * lens.x + basis[1] * lens.y;
        ro = ro + offset;
        rd = normalize(focus_p - ro);
    }

    var throughput = vec3<f32>(1.0);
    var prev_pdf = 0.0;      // BSDF pdf used to reach the current vertex
    var prev_specular = true; // first hit / specular bounces take full emission

    for (var bounce = 0u; bounce < frame.max_bounces; bounce = bounce + 1u) {
        let hit = trace_closest(ro, rd, T_MAX);
        if (!hit.valid) {
            if (prev_specular) {
                // Directly-seen path (camera rays + perfect-specular reflections/
                // refractions): show the backdrop — the HDRI environment if bound,
                // else the cosmetic flat background color. No ambient is added here,
                // so the visible background isn't washed out by the fill light.
                res.radiance += throughput * sky(rd);
            } else {
                // Scattered (diffuse/glossy) miss: this is the environment as a
                // LIGHT. A uniform ambient term (a constant white dome of radiance
                // `frame.ambient`) fills surfaces like the rasterizer's ambient,
                // plus the HDRI environment (MIS-weighted) when bound. The flat
                // background is deliberately excluded so it never tints objects.
                var lit = vec3<f32>(frame.ambient);
                if (frame.has_env != 0u) {
                    lit += sky(rd) * power_heuristic(prev_pdf, env_pdf(rd));
                }
                res.radiance += throughput * lit;
            }
            break;
        }

        let mat = materials[hit.material_id];
        let p = ro + rd * hit.t;

        // Alpha (coverage) transparency: with probability `1 - base_color.a` the
        // ray misses this surface entirely and continues unchanged. Averaged over
        // samples this yields order-independent alpha blending, distinct from the
        // physical refraction of the glass BSDF. Pass-throughs cost a bounce (so a
        // deep stack of transparent surfaces needs a larger `max_bounces`).
        if (rand(rng) >= mat.base_color.a) {
            ro = p + rd * EPS;
            continue;
        }

        var n = hit.normal;
        n = apply_normal_map(mat, n, hit.uv);
        let gn = hit.geom_normal;
        let entering = dot(gn, rd) < 0.0;
        let wo = -rd;
        // Orient the shading normal to the incoming side for shading.
        if (dot(n, wo) < 0.0) {
            n = -n;
        }

        let s = resolve_surface(mat, hit.uv);

        // Emission from a directly hit surface. When the previous bounce was a
        // (non-specular) BSDF sample and emitters are explicitly sampled in
        // `sample_lights`, MIS-weight this contribution by the power heuristic
        // between the BSDF pdf that reached here and the emitter-sampling pdf,
        // so the two strategies combine without double counting.
        if (any(s.emissive > vec3<f32>(0.0))) {
            var w = 1.0;
            if (!prev_specular && frame.num_emitters > 0u) {
                // Area of the directly hit triangle for the emitter-sampling pdf.
                let pdf_l = emitter_pdf(ro, rd, p, hit.geom_normal, hit.tri_area);
                w = power_heuristic(prev_pdf, pdf_l);
            }
            res.radiance += throughput * s.emissive * w;
        }

        // Record first-hit guides (albedo + world normal) for the denoiser.
        if (!res.has_first) {
            res.first_albedo = s.albedo;
            res.first_normal = n;
            res.has_first = true;
        }

        // Next-event estimation (analytic + area lights).
        res.radiance += throughput * sample_lights(s, p, n, wo, rng);

        // Sample the BSDF for the next bounce.
        let bs = sample_bsdf(s, n, wo, rd, entering, rng);
        if (!bs.valid) {
            break;
        }
        throughput *= bs.throughput;
        prev_pdf = bs.pdf;
        prev_specular = bs.specular;

        // Offset along the correct side: transmitted rays cross the surface.
        let offset_n = select(gn, -gn, dot(gn, bs.wi) < 0.0);
        ro = p + offset_n * EPS;
        rd = bs.wi;

        // Russian roulette after a few bounces.
        if (bounce >= 3u) {
            let q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.99);
            if (rand(rng) > q) {
                break;
            }
            throughput = throughput / q;
        }
    }

    return res;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= frame.width || gid.y >= frame.height) {
        return;
    }
    let pidx = gid.y * frame.width + gid.x;
    // Region stride into `pixels`: radiance at `pidx`, albedo guide at `npix+pidx`,
    // normal guide at `2*npix+pidx` (see the layout note in rt_preamble.wgsl).
    let npix = frame.width * frame.height;

    // Trace `samples_per_frame` paths this dispatch and accumulate their sum.
    let spp = max(frame.samples_per_frame, 1u);
    var radiance_sum = vec3<f32>(0.0);
    var albedo_sum = vec3<f32>(0.0);
    var normal_sum = vec3<f32>(0.0);
    for (var s = 0u; s < spp; s = s + 1u) {
        var rng = init_rng(pidx, frame.seed + s * 9277u);
        let r = sample_pixel(gid, &rng);
        radiance_sum += r.radiance;
        albedo_sum += r.first_albedo;
        normal_sum += r.first_normal;
    }

    // Running mean: blend `spp` new samples into the `sample_index` already stored.
    let spp_f = f32(spp);
    var out_color: vec3<f32>;
    var out_albedo: vec3<f32>;
    var out_normal: vec3<f32>;
    if (frame.sample_index == 0u) {
        out_color = radiance_sum / spp_f;
        out_albedo = albedo_sum / spp_f;
        out_normal = normal_sum / spp_f;
    } else {
        let s = f32(frame.sample_index);
        out_color = (pixels[pidx].rgb * s + radiance_sum) / (s + spp_f);
        out_albedo = (pixels[npix + pidx].rgb * s + albedo_sum) / (s + spp_f);
        out_normal = (pixels[2u * npix + pidx].rgb * s + normal_sum) / (s + spp_f);
    }
    // Reject NaNs/Infs so a single bad sample can't poison the running mean.
    if (any(out_color != out_color) || any(out_color > vec3<f32>(1e20))) {
        out_color = vec3<f32>(0.0);
    }
    pixels[pidx] = vec4<f32>(out_color, 1.0);
    pixels[npix + pidx] = vec4<f32>(out_albedo, 1.0);
    pixels[2u * npix + pidx] = vec4<f32>(normalize(out_normal + vec3<f32>(0.0, 0.0, 1e-8)), 1.0);
}
