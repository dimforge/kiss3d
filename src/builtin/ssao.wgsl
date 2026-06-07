// Screen-space ambient occlusion from the view-position prepass.
//
// For each pixel it reconstructs the view-space normal from neighboring
// positions, samples a hemisphere of points oriented by that normal, projects
// each back to screen, and counts how many are occluded by nearer geometry. The
// result (1 = unoccluded) darkens the ambient term in the main shader.

struct SsaoUniforms {
    proj: mat4x4<f32>,
    inv_resolution: vec2<f32>,
    radius: f32,
    bias: f32,
    intensity: f32,
    power: f32,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var t_viewpos: texture_2d<f32>;
@group(0) @binding(1) var s_pt: sampler;
@group(0) @binding(2) var<uniform> u: SsaoUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var c = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let xy = c[vid];
    var o: VsOut;
    o.pos = vec4<f32>(xy, 0.0, 1.0);
    o.uv = vec2<f32>((xy.x + 1.0) * 0.5, (1.0 - xy.y) * 0.5);
    return o;
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// 16-sample hemisphere kernel (z up), weighted toward the origin.
fn kernel(i: i32) -> vec3<f32> {
    var k = array<vec3<f32>, 16>(
        vec3<f32>(0.04, 0.02, 0.05), vec3<f32>(-0.06, 0.03, 0.08),
        vec3<f32>(0.09, -0.05, 0.06), vec3<f32>(-0.02, -0.08, 0.10),
        vec3<f32>(0.12, 0.10, 0.14), vec3<f32>(-0.14, -0.06, 0.12),
        vec3<f32>(0.05, -0.16, 0.18), vec3<f32>(-0.18, 0.12, 0.16),
        vec3<f32>(0.22, 0.04, 0.20), vec3<f32>(-0.10, 0.24, 0.22),
        vec3<f32>(0.18, -0.22, 0.28), vec3<f32>(-0.28, -0.14, 0.26),
        vec3<f32>(0.30, 0.26, 0.34), vec3<f32>(-0.34, 0.20, 0.36),
        vec3<f32>(0.12, -0.40, 0.42), vec3<f32>(-0.20, -0.36, 0.50),
    );
    return k[i];
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let center = textureSampleLevel(t_viewpos, s_pt, in.uv, 0.0);
    if center.a < 0.5 {
        return vec4<f32>(1.0); // background: unoccluded
    }
    let p = center.xyz;

    // Reconstruct the view-space normal from neighboring positions.
    let px = textureSampleLevel(t_viewpos, s_pt, in.uv + vec2<f32>(u.inv_resolution.x, 0.0), 0.0).xyz;
    let py = textureSampleLevel(t_viewpos, s_pt, in.uv + vec2<f32>(0.0, u.inv_resolution.y), 0.0).xyz;
    var n = normalize(cross(px - p, py - p));
    if n.z < 0.0 {
        n = -n;
    }

    // Per-pixel rotation to decorrelate the kernel (cuts banding; blurred later).
    let ang = hash(in.uv / u.inv_resolution) * 6.2831853;
    let rvec = vec3<f32>(cos(ang), sin(ang), 0.0);
    let t = normalize(rvec - n * dot(rvec, n));
    let b = cross(n, t);
    let tbn = mat3x3<f32>(t, b, n);

    var occlusion = 0.0;
    for (var i = 0; i < 16; i = i + 1) {
        let sample_pos = p + (tbn * kernel(i)) * u.radius;
        let clip = u.proj * vec4<f32>(sample_pos, 1.0);
        if clip.w <= 0.0 {
            continue;
        }
        let ndc = clip.xyz / clip.w;
        let suv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
        if suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0 {
            continue;
        }
        let sampled = textureSampleLevel(t_viewpos, s_pt, suv, 0.0);
        if sampled.a < 0.5 {
            continue;
        }
        // View-space z grows toward the camera (it is negative ahead). Geometry
        // in front of the sample (larger z) occludes it.
        let range = smoothstep(0.0, 1.0, u.radius / max(abs(p.z - sampled.z), 1e-3));
        if sampled.z >= sample_pos.z + u.bias {
            occlusion += range;
        }
    }

    let ao = clamp(1.0 - (occlusion / 16.0) * u.intensity, 0.0, 1.0);
    return vec4<f32>(pow(ao, u.power));
}
