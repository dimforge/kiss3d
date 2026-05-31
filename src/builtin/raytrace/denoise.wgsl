// Edge-aware à-trous wavelet denoiser for the path tracer.
//
// One iteration of an SVGF-style à-trous filter: a 5x5 B-spline kernel is
// applied at an exponentially growing tap spacing (`step`), with edge-stopping
// weights derived from the guide normal and from the luminance of the radiance
// itself. The driver runs this entry point several times with increasing
// `step` values, ping-ponging between two scratch buffers.
//
// Albedo demodulation: before the very first iteration the driver divides the
// radiance by (albedo + eps) so only the incident lighting is filtered (the
// `demodulate` flag computes this on the fly when reading from the raw
// accumulation buffer); after the last iteration the result is re-multiplied by
// the albedo in the tonemap pass... no — to keep the tonemap pass untouched we
// re-multiply here on the final iteration via the `remodulate` flag. This keeps
// crisp texture/albedo detail while smoothing the noisy lighting.

struct DenoiseUniforms {
    width: u32,
    height: u32,
    // Tap spacing for this iteration (1, 2, 4, ...).
    step: i32,
    // 1 on the first iteration: `src` is the raw accumulation buffer and must be
    // albedo-demodulated on read.
    demodulate: u32,
    // 1 on the last iteration: re-multiply the filtered lighting by the albedo
    // so `dst` holds final HDR radiance again.
    remodulate: u32,
    // Edge-stopping strength for the normal guide (power exponent).
    sigma_normal: f32,
    // Edge-stopping strength for luminance (larger = more blurring).
    sigma_luminance: f32,
    pad0: f32,
};

@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
// The shared accumulation buffer (radiance + guides). The first `width*height`
// pixels are radiance, the next are the albedo guide, the next the normal guide.
// `src` aliases its radiance region on the first iteration; on later iterations
// `src` is a scratch buffer, but the guides always come from here.
@group(0) @binding(2) var<storage, read> guides: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> u: DenoiseUniforms;

const EPS: f32 = 1e-3;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// First-hit albedo guide at pixel `idx` (region 1 of the shared buffer).
fn guide_albedo(idx: u32) -> vec3<f32> {
    return guides[u.width * u.height + idx].rgb;
}

// First-hit world-normal guide at pixel `idx` (region 2 of the shared buffer).
fn guide_normal(idx: u32) -> vec3<f32> {
    return guides[2u * u.width * u.height + idx].rgb;
}

// Returns the (demodulated, if requested) lighting at pixel `idx`.
fn read_lighting(idx: u32) -> vec3<f32> {
    let radiance = src[idx].rgb;
    if (u.demodulate != 0u) {
        return radiance / (guide_albedo(idx) + vec3<f32>(EPS));
    }
    return radiance;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    let cidx = gid.y * u.width + gid.x;

    let center_light = read_lighting(cidx);
    let center_normal = guide_normal(cidx);
    let center_lum = luminance(center_light);

    // 5x5 B-spline (à-trous) kernel weights along one axis: 1/16, 1/4, 3/8.
    let kernel = array<f32, 3>(0.375, 0.25, 0.0625);

    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;

    for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
        for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
            let sx = i32(gid.x) + dx * u.step;
            let sy = i32(gid.y) + dy * u.step;
            if (sx < 0 || sy < 0 || sx >= i32(u.width) || sy >= i32(u.height)) {
                continue;
            }
            let sidx = u32(sy) * u.width + u32(sx);

            let sample_light = read_lighting(sidx);
            let sample_normal = guide_normal(sidx);

            // à-trous spatial kernel (separable weights multiplied).
            let h = kernel[abs(dx)] * kernel[abs(dy)];

            // Edge-stopping on the surface normal: only blur across pixels whose
            // first-hit normal is closely aligned with the center's.
            let n_dot = max(dot(center_normal, sample_normal), 0.0);
            let w_normal = pow(n_dot, u.sigma_normal);

            // Edge-stopping on luminance: preserve high-contrast lighting edges
            // (shadow boundaries, highlights) instead of smearing them.
            let lum_diff = abs(center_lum - luminance(sample_light));
            let w_lum = exp(-lum_diff / (u.sigma_luminance + EPS));

            let w = h * w_normal * w_lum;
            sum = sum + sample_light * w;
            weight_sum = weight_sum + w;
        }
    }

    var filtered = center_light;
    if (weight_sum > 0.0) {
        filtered = sum / weight_sum;
    }

    if (u.remodulate != 0u) {
        // Re-apply the albedo so `dst` is HDR radiance the tonemap pass expects.
        filtered = filtered * (guide_albedo(cidx) + vec3<f32>(EPS));
    }

    dst[cidx] = vec4<f32>(filtered, 1.0);
}
