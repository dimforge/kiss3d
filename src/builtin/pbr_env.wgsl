// Shared environment + image-based-lighting helpers, imported (via WESL) by every
// pass that samples an equirectangular environment or approximates the environment
// BRDF: the default PBR material, screen-space reflections, the skybox, and the
// path tracer. Keeping them here is the single source of truth for the
// direction<->UV convention (previously copy-pasted, with "matches the path
// tracer / skybox" comments standing in for a shared definition).

const PBR_ENV_PI: f32 = 3.14159265359;

// Equirectangular mapping: a (normalized) world-space direction -> environment UV.
// `u` wraps around the horizon via atan2(z, x); `v` runs top(+Y)->bottom(-Y) via
// acos(y). Must match the equirectangular maps the renderer builds.
fn equirect_dir_to_uv(d: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(
        atan2(d.z, d.x) / (2.0 * PBR_ENV_PI) + 0.5,
        acos(clamp(d.y, -1.0, 1.0)) / PBR_ENV_PI,
    );
}

// Karis' analytic environment-BRDF approximation ("Mobile" split-sum, avoids a
// precomputed LUT): returns the (scale, bias) applied to F0 for the prefiltered
// environment specular, folded into a single vec3.
fn env_brdf_approx(f0: vec3<f32>, roughness: f32, nov: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nov)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + vec2<f32>(r.z, r.w);
    return f0 * ab.x + vec3<f32>(ab.y);
}
