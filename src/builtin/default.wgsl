// Default material shader for kiss3d
// Implements Cook-Torrance PBR with texture support, instancing, and multi-light

const PI: f32 = 3.14159265359;
const MAX_LIGHTS: u32 = 8u;

// Light type constants
const LIGHT_TYPE_POINT: u32 = 0u;
const LIGHT_TYPE_DIRECTIONAL: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;

// Single light data structure
struct LightData {
    position: vec3<f32>,
    light_type: u32,
    direction: vec3<f32>,
    intensity: f32,
    color: vec3<f32>,
    inner_cone_cos: f32,
    outer_cone_cos: f32,
    attenuation_radius: f32,
    _padding: vec2<f32>,
}

// Bind group 0: Frame uniforms (view, projection, lights)
struct FrameUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    lights: array<LightData, MAX_LIGHTS>,
    num_lights: u32,
    ambient_intensity: f32,
    _padding: vec2<f32>,
    ambient_color: vec4<f32>,
    fog_color: vec4<f32>,
    // (mode, param_a, param_b, height_falloff): mode 0 off / 1 linear / 2 exp / 3 exp2.
    fog_params: vec4<f32>,
}

// Blends `color` toward the fog color by an amount derived from the fragment's
// view-space distance and (optionally) its world height.
fn apply_fog(color: vec3<f32>, view_dist: f32, world_y: f32) -> vec3<f32> {
    let mode = frame.fog_params.x;
    if mode < 0.5 {
        return color;
    }
    var f = 0.0;
    if mode < 1.5 {
        // Linear: param_a = start, param_b = end.
        let start = frame.fog_params.y;
        let end = frame.fog_params.z;
        f = clamp((view_dist - start) / max(end - start, 1e-4), 0.0, 1.0);
    } else if mode < 2.5 {
        // Exponential: param_a = density.
        f = 1.0 - exp(-frame.fog_params.y * view_dist);
    } else {
        // Exponential squared.
        let d = frame.fog_params.y * view_dist;
        f = 1.0 - exp(-d * d);
    }
    // Optional height thinning: less fog higher up.
    let hf = frame.fog_params.w;
    if hf > 0.0 {
        f *= exp(-max(world_y, 0.0) * hf);
    }
    return mix(color, frame.fog_color.rgb, clamp(f, 0.0, 1.0) * frame.fog_color.a);
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

// Bind group 1: Object uniforms (transform, scale, color, PBR properties)
struct ObjectUniforms {
    transform: mat4x4<f32>,
    ntransform: mat3x3<f32>,
    scale: mat3x3<f32>,
    color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    _pad0: vec2<f32>,
    emissive: vec4<f32>,
    has_normal_map: f32,
    has_metallic_roughness_map: f32,
    has_ao_map: f32,
    has_emissive_map: f32,
    reflectance: f32,
    clearcoat: f32,
    clearcoat_roughness: f32,
    anisotropy: f32,
    anisotropy_rotation: f32,
    transmission: f32,
    // Alpha mode (0 opaque / 1 mask / 2 blend / 3 premultiplied) + mask cutoff.
    alpha_mode: f32,
    alpha_cutoff: f32,
    specular_tint: vec4<f32>,
}

@group(1) @binding(0)
var<uniform> object: ObjectUniforms;

// Bind group 2: material textures — albedo plus the PBR maps. Albedo and the PBR
// maps are merged into a single group so the pipeline uses only 4 bind groups,
// staying within WebGPU's `maxBindGroups` limit of 4 (browsers expose exactly 4).
@group(2) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(2) @binding(1)
var s_diffuse: sampler;
@group(2) @binding(2)
var t_normal: texture_2d<f32>;
@group(2) @binding(3)
var s_normal: sampler;
@group(2) @binding(4)
var t_metallic_roughness: texture_2d<f32>;
@group(2) @binding(5)
var s_metallic_roughness: sampler;
@group(2) @binding(6)
var t_ao: texture_2d<f32>;
@group(2) @binding(7)
var s_ao: sampler;
@group(2) @binding(8)
var t_emissive: texture_2d<f32>;
@group(2) @binding(9)
var s_emissive: sampler;

// === SHADOW MAPPING (group 3) — localized block for easy merging ===
// Maximum number of atlas views (must match builtin/shadow.rs MAX_SHADOW_VIEWS).
const MAX_SHADOW_VIEWS: u32 = 16u;

// Per-light shadow metadata (mirrors GpuLightShadow in builtin/shadow.rs).
struct LightShadow {
    base_view: u32,
    num_views: u32,
    light_type: u32,
    enabled: f32,
    light_pos: vec3<f32>,
    far_plane: f32,
}

struct ShadowUniforms {
    view_proj: array<mat4x4<f32>, MAX_SHADOW_VIEWS>,
    lights: array<LightShadow, MAX_LIGHTS>,
    shadows_enabled: f32,
    texel_size: f32,
    depth_bias: f32,
    // 1.0 when translucent casters tinted the colored transmittance atlas.
    transmittance_enabled: f32,
    // PCF kernel scale (shadow softness/blur): 1.0 = default, larger = softer.
    softness: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    // Far view-space distance of each directional cascade (0..num_cascades).
    cascade_splits: vec4<f32>,
}

@group(3) @binding(0)
var t_shadow_atlas: texture_depth_2d_array;
@group(3) @binding(1)
var s_shadow: sampler_comparison;
@group(3) @binding(2)
var<uniform> shadow: ShadowUniforms;
// Colored transmittance atlas: RGB transmittance of translucent occluders in
// front of the nearest opaque surface (white where nothing translucent occludes).
@group(3) @binding(3)
var t_shadow_transmittance: texture_2d_array<f32>;
@group(3) @binding(4)
var s_shadow_color: sampler;

// Colored visibility of a light at a shadow texel: the opaque PCF visibility
// (`vis`, in [0,1]) and the RGB transmittance of any translucent occluders.
struct ShadowSample {
    vis: f32,
    transmit: vec3<f32>,
}

// Samples the colored transmittance atlas at `uv` of `layer` (white when no
// translucent caster contributed this frame).
fn sample_transmittance(layer: u32, uv: vec2<f32>) -> vec3<f32> {
    if shadow.transmittance_enabled < 0.5 {
        return vec3<f32>(1.0);
    }
    return textureSampleLevel(t_shadow_transmittance, s_shadow_color, uv, i32(layer), 0.0).rgb;
}

// One tap, with the receiver-plane depth bias applied: the compare depth follows
// the receiver's own plane at the tap (`base_z + grad . (tap_uv - ref_uv)`), so a
// wide kernel tests against the surface's actual depth there instead of the flat
// center depth — eliminating self-shadow acne without flattening contacts.
fn shadow_tap_rpdb(
    layer: u32, tap_uv: vec2<f32>, ref_uv: vec2<f32>, base_z: f32, grad: vec2<f32>,
) -> f32 {
    let compare = base_z + dot(grad, tap_uv - ref_uv);
    return textureSampleCompareLevel(t_shadow_atlas, s_shadow, tap_uv, i32(layer), compare);
}

// Castaño 2013 ("Shadow Mapping Summary Part 1") optimized PCF: a tent-weighted
// 3x3 grid of hardware bilinear comparison taps. Each tap is placed at a fractional
// offset so the GPU's 2x2 bilinear comparison, combined with the tent weights,
// reproduces a smooth ~5x5 PCF using only 9 taps — smoother AND crisper than a
// naive box PCF for the same cost. Returns visibility in [0,1].
//
// `base_z` is the receiver depth at `uv` (already nudged by the small constant
// bias) and `grad = (dz/du, dz/dv)` is the receiver-plane depth gradient, so each
// tap is compared against the plane depth at its own position (receiver-plane
// depth bias). This is what keeps wide/soft kernels free of acne.
fn shadow_pcf(layer: u32, uv: vec2<f32>, base_z: f32, grad: vec2<f32>) -> f32 {
    let map_size = vec2<f32>(textureDimensions(t_shadow_atlas));
    let inv_size = 1.0 / map_size;

    let coord = uv * map_size;
    var base_uv = floor(coord + 0.5);
    let s = coord.x + 0.5 - base_uv.x;
    let t = coord.y + 0.5 - base_uv.y;
    base_uv = (base_uv - 0.5) * inv_size;

    // `softness` scales the tap spacing about the kernel center: 1.0 is the
    // default penumbra, larger blurs the edge, 0.0 collapses to a hard edge.
    let spread = inv_size * shadow.softness;

    let uw0 = 4.0 - 3.0 * s;
    let uw1 = 7.0;
    let uw2 = 1.0 + 3.0 * s;
    let u0 = (3.0 - 2.0 * s) / uw0 - 2.0;
    let u1 = (3.0 + s) / uw1;
    let u2 = s / uw2 + 2.0;

    let vw0 = 4.0 - 3.0 * t;
    let vw1 = 7.0;
    let vw2 = 1.0 + 3.0 * t;
    let v0 = (3.0 - 2.0 * t) / vw0 - 2.0;
    let v1 = (3.0 + t) / vw1;
    let v2 = t / vw2 + 2.0;

    let p00 = base_uv + vec2<f32>(u0, v0) * spread;
    let p10 = base_uv + vec2<f32>(u1, v0) * spread;
    let p20 = base_uv + vec2<f32>(u2, v0) * spread;
    let p01 = base_uv + vec2<f32>(u0, v1) * spread;
    let p11 = base_uv + vec2<f32>(u1, v1) * spread;
    let p21 = base_uv + vec2<f32>(u2, v1) * spread;
    let p02 = base_uv + vec2<f32>(u0, v2) * spread;
    let p12 = base_uv + vec2<f32>(u1, v2) * spread;
    let p22 = base_uv + vec2<f32>(u2, v2) * spread;

    var sum = 0.0;
    sum += uw0 * vw0 * shadow_tap_rpdb(layer, p00, uv, base_z, grad);
    sum += uw1 * vw0 * shadow_tap_rpdb(layer, p10, uv, base_z, grad);
    sum += uw2 * vw0 * shadow_tap_rpdb(layer, p20, uv, base_z, grad);

    sum += uw0 * vw1 * shadow_tap_rpdb(layer, p01, uv, base_z, grad);
    sum += uw1 * vw1 * shadow_tap_rpdb(layer, p11, uv, base_z, grad);
    sum += uw2 * vw1 * shadow_tap_rpdb(layer, p21, uv, base_z, grad);

    sum += uw0 * vw2 * shadow_tap_rpdb(layer, p02, uv, base_z, grad);
    sum += uw1 * vw2 * shadow_tap_rpdb(layer, p12, uv, base_z, grad);
    sum += uw2 * vw2 * shadow_tap_rpdb(layer, p22, uv, base_z, grad);

    return sum * (1.0 / 144.0);
}

// Out-of-map default: fully lit, no tint.
fn shadow_sample_lit() -> ShadowSample {
    return ShadowSample(1.0, vec3<f32>(1.0));
}

// Receiver-plane depth gradient `(dz/du, dz/dv)` for `layer`: how the receiver's
// light-space depth changes per unit of shadow-map UV, so PCF can bias each tap
// onto the actual receiver plane (option 3).
//
// `dpos_dx/dpos_dy` are the screen-space derivatives of the world position
// (taken once in uniform control flow, in `shade`). Projecting them through this
// layer's `view_proj` gives the screen-space derivatives of `(uv, depth)`; we
// then invert the 2x2 UV Jacobian to change basis from screen space to UV space.
// Computing it from the world derivatives (rather than `dpdx` of the per-layer
// uv) keeps it valid in non-uniform control flow and consistent across cascade /
// cube-face seams. `det == 0` (degenerate, e.g. edge-on) falls back to no slope.
fn receiver_plane_grad(
    layer: u32, clip: vec4<f32>, ndc: vec3<f32>, dpos_dx: vec3<f32>, dpos_dy: vec3<f32>,
) -> vec2<f32> {
    let m = shadow.view_proj[layer];
    // d(clip)/d(screen) = (columns 0..2 of view_proj) . d(world)/d(screen).
    let dclip_dx = m[0] * dpos_dx.x + m[1] * dpos_dx.y + m[2] * dpos_dx.z;
    let dclip_dy = m[0] * dpos_dy.x + m[1] * dpos_dy.y + m[2] * dpos_dy.z;

    // Through the perspective divide: d(ndc) = (d(clip).xyz - ndc * d(clip).w) / w.
    let inv_w = 1.0 / clip.w;
    let dndc_dx = (dclip_dx.xyz - ndc * dclip_dx.w) * inv_w;
    let dndc_dy = (dclip_dy.xyz - ndc * dclip_dy.w) * inv_w;

    // uv = (ndc.x*0.5+0.5, -ndc.y*0.5+0.5); depth compared = ndc.z.
    let du_dx = dndc_dx.x * 0.5;
    let dv_dx = -dndc_dx.y * 0.5;
    let du_dy = dndc_dy.x * 0.5;
    let dv_dy = -dndc_dy.y * 0.5;

    let det = du_dx * dv_dy - dv_dx * du_dy;
    if abs(det) < 1e-12 {
        return vec2<f32>(0.0);
    }
    let inv_det = 1.0 / det;
    // Solve [du_dx dv_dx; du_dy dv_dy] * [dz/du; dz/dv] = [dz/dx; dz/dy].
    let dz_du = (dv_dy * dndc_dx.z - dv_dx * dndc_dy.z) * inv_det;
    let dz_dv = (du_dx * dndc_dy.z - du_dy * dndc_dx.z) * inv_det;

    // Clamp the slope so a degenerate gradient (silhouette edge, grazing angle)
    // can't over-bias into light leaking: cap the depth change to `MAX` per texel.
    let res = f32(textureDimensions(t_shadow_atlas).x);
    let per_texel = max(abs(dz_du), abs(dz_dv)) / res;
    let grad = vec2<f32>(dz_du, dz_dv);
    // Generous cap: only rein in truly degenerate gradients (near edge-on faces,
    // silhouette spikes) that would over-bias into light leaking. Legitimate
    // grazing slopes are large, so clamping too low reintroduces acne.
    let max_per_texel = 0.5;
    if per_texel > max_per_texel {
        return grad * (max_per_texel / per_texel);
    }
    return grad;
}

// Samples one atlas layer at a world position (project + bounds-check + PCF +
// transmittance), used by spot and point lights. Fully lit when out of the map.
fn sample_shadow_layer(
    layer: u32, world_pos: vec3<f32>, dpos_dx: vec3<f32>, dpos_dy: vec3<f32>,
) -> ShadowSample {
    let light_clip = shadow.view_proj[layer] * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 {
        return shadow_sample_lit();
    }
    let ndc = light_clip.xyz / light_clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0 || ndc.z < 0.0 {
        return shadow_sample_lit();
    }
    let grad = receiver_plane_grad(layer, light_clip, ndc, dpos_dx, dpos_dy);
    return ShadowSample(
        shadow_pcf(layer, uv, ndc.z - shadow.depth_bias, grad),
        sample_transmittance(layer, uv),
    );
}

// Projects `world_pos` into one cascade's atlas layer and samples it.
// Fully lit if the fragment falls outside that layer's map.
fn sample_one_cascade(
    layer: u32, world_pos: vec3<f32>, dpos_dx: vec3<f32>, dpos_dy: vec3<f32>,
) -> ShadowSample {
    return sample_shadow_layer(layer, world_pos, dpos_dx, dpos_dy);
}

// Cascaded shadow maps for a directional light: the `num_cascades` layers from
// `base_view` are nested frustum slices ordered near -> far. Select the cascade by
// the fragment's view-space depth, and cross-fade into the next cascade over a band
// before each boundary so the resolution change isn't a hard seam.
fn sample_directional_cascades(
    base_view: u32, num_cascades: u32, view_depth: f32, world_pos: vec3<f32>,
    dpos_dx: vec3<f32>, dpos_dy: vec3<f32>,
) -> ShadowSample {
    // Pick the first cascade whose far bound is beyond the fragment depth.
    var c = num_cascades - 1u;
    for (var i = 0u; i < num_cascades; i = i + 1u) {
        if view_depth < shadow.cascade_splits[i] {
            c = i;
            break;
        }
    }

    let s = sample_one_cascade(base_view + c, world_pos, dpos_dx, dpos_dy);

    // Blend into the next cascade across a band before this cascade's far split.
    if c + 1u < num_cascades {
        let split = shadow.cascade_splits[c];
        let band = split * 0.2;
        if view_depth > split - band {
            let s_next = sample_one_cascade(base_view + c + 1u, world_pos, dpos_dx, dpos_dy);
            let t = clamp((view_depth - (split - band)) / band, 0.0, 1.0);
            return ShadowSample(
                mix(s.vis, s_next.vis, t),
                mix(s.transmit, s_next.transmit, t),
            );
        }
    }
    return s;
}

// Selects the point-light cube face (atlas layer) for a light->fragment vector.
// Face order matches builtin/shadow.rs: +X,-X,+Y,-Y,+Z,-Z.
fn point_cube_face(dir: vec3<f32>) -> u32 {
    let a = abs(dir);
    if a.x >= a.y && a.x >= a.z {
        if dir.x > 0.0 { return 0u; } else { return 1u; }
    } else if a.y >= a.z {
        if dir.y > 0.0 { return 2u; } else { return 3u; }
    } else {
        if dir.z > 0.0 { return 4u; } else { return 5u; }
    }
}

// Returns the colored light visibility for light `light_index` at `world_pos`:
// the opaque-shadow visibility in [0,1] tinted by any translucent occluders.
// `vec3(1.0)` means fully lit.
fn compute_shadow(
    light_index: u32, world_pos: vec3<f32>, dpos_dx: vec3<f32>, dpos_dy: vec3<f32>,
) -> vec3<f32> {
    if shadow.shadows_enabled < 0.5 {
        return vec3<f32>(1.0);
    }
    let ls = shadow.lights[light_index];
    if ls.enabled < 0.5 {
        return vec3<f32>(1.0);
    }

    var s: ShadowSample;
    if ls.light_type == LIGHT_TYPE_POINT {
        let face = point_cube_face(world_pos - ls.light_pos);
        s = sample_shadow_layer(ls.base_view + face, world_pos, dpos_dx, dpos_dy);
    } else if ls.light_type == LIGHT_TYPE_DIRECTIONAL {
        // Cascaded shadow maps: select/blend cascades by view-space depth (the
        // distance in front of the camera; view -z is forward).
        let view_depth = -(frame.view * vec4<f32>(world_pos, 1.0)).z;
        s = sample_directional_cascades(
            ls.base_view, ls.num_views, view_depth, world_pos, dpos_dx, dpos_dy,
        );
    } else {
        // Spot lights use a single perspective view.
        s = sample_shadow_layer(ls.base_view, world_pos, dpos_dx, dpos_dy);
    }

    // Opaque visibility scales the (colored) translucent transmittance.
    return vec3<f32>(s.vis) * s.transmit;
}
// === END SHADOW MAPPING block ===

// Vertex input
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coord: vec2<f32>,
    @location(2) normal: vec3<f32>,
}

// Instance input
struct InstanceInput {
    @location(3) inst_tra: vec3<f32>,
    @location(4) inst_color: vec4<f32>,
    @location(5) inst_def_0: vec3<f32>,
    @location(6) inst_def_1: vec3<f32>,
    @location(7) inst_def_2: vec3<f32>,
}

// Vertex output / Fragment input
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) vert_color: vec4<f32>,
    @location(4) view_pos: vec3<f32>,
}

// === PBR BRDF Functions ===

// Normal Distribution Function (GGX/Trowbridge-Reitz)
fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;

    var denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return a2 / max(denom, 0.0001);
}

// Geometry function (Smith's Schlick-GGX)
fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;  // Direct lighting

    return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx2 = geometry_schlick_ggx(NdotV, roughness);
    let ggx1 = geometry_schlick_ggx(NdotL, roughness);

    return ggx1 * ggx2;
}

// Fresnel (Schlick approximation)
fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Scalar Schlick Fresnel (used by the dielectric clearcoat lobe).
fn fresnel_schlick_scalar(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Isotropic GGX/Trowbridge-Reitz NDF parameterized by linear roughness `alpha`.
fn d_ggx_alpha(NoH: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Anisotropic GGX NDF. `at`/`ab` are the tangent/bitangent roughnesses.
fn d_ggx_aniso(at: f32, ab: f32, ToH: f32, BoH: f32, NoH: f32) -> f32 {
    let a2 = at * ab;
    let v = vec3<f32>(ab * ToH, at * BoH, a2 * NoH);
    let v2 = dot(v, v);
    let w2 = a2 / max(v2, 1e-7);
    return a2 * w2 * w2 * (1.0 / PI);
}

// Height-correlated Smith visibility (includes the 1/(4·NoV·NoL) term).
fn v_smith_correlated(NoV: f32, NoL: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let lv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    let ll = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / max(lv + ll, 1e-7);
}

// Anisotropic height-correlated Smith visibility.
fn v_smith_correlated_aniso(
    at: f32, ab: f32, ToV: f32, BoV: f32, ToL: f32, BoL: f32, NoV: f32, NoL: f32,
) -> f32 {
    let lv = NoL * length(vec3<f32>(at * ToV, ab * BoV, NoV));
    let ll = NoV * length(vec3<f32>(at * ToL, ab * BoL, NoL));
    return 0.5 / max(lv + ll, 1e-7);
}

// Kelemen visibility for the clearcoat lobe.
fn v_kelemen(LoH: f32) -> f32 {
    return 0.25 / max(LoH * LoH, 1e-7);
}

// Attenuation functions
fn calculate_point_attenuation(dist: f32, radius: f32) -> f32 {
    // Smooth falloff that reaches zero at the attenuation radius
    let normalized_dist = clamp(dist / radius, 0.0, 1.0);
    let attenuation = 1.0 - normalized_dist * normalized_dist;
    return attenuation * attenuation;
}

fn calculate_spot_attenuation(
    L: vec3<f32>,
    spot_direction: vec3<f32>,
    dist: f32,
    inner_cone_cos: f32,
    outer_cone_cos: f32,
    radius: f32
) -> f32 {
    // Angular attenuation
    let cos_angle = dot(-L, spot_direction);
    let angular_attenuation = clamp(
        (cos_angle - outer_cone_cos) / max(inner_cone_cos - outer_cone_cos, 0.0001),
        0.0,
        1.0
    );

    // Distance attenuation
    let dist_attenuation = calculate_point_attenuation(dist, radius);

    return angular_attenuation * angular_attenuation * dist_attenuation;
}

// === Vertex Shader ===

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    // Build deformation matrix from instance data
    let deformation = mat3x3<f32>(
        instance.inst_def_0,
        instance.inst_def_1,
        instance.inst_def_2
    );

    // Transform position
    let scaled_pos = object.scale * vertex.position;
    let deformed_pos = deformation * scaled_pos;
    let model_pos = object.transform * vec4<f32>(deformed_pos, 1.0);
    let world_pos = vec4<f32>(instance.inst_tra, 0.0) + model_pos;

    out.clip_position = frame.proj * frame.view * world_pos;
    out.world_pos = world_pos.xyz;

    // Transform normal to world space
    out.world_normal = normalize(deformation * object.ntransform * vertex.normal);

    // View-space position for lighting calculations
    let view_pos = frame.view * world_pos;
    out.view_pos = view_pos.xyz / view_pos.w;

    out.tex_coord = vertex.tex_coord;
    out.vert_color = instance.inst_color;

    return out;
}

// === Fragment Shader ===

// Shades the fragment, returning (linear HDR color, alpha). Shared by the opaque
// pass (`fs_main`) and the weighted-blended OIT pass (`fs_oit`).
fn shade(in: VertexOutput) -> vec4<f32> {
    // Screen-space derivatives of the world position, taken here in uniform
    // control flow (before any branching) so they are valid. The shadow code
    // projects these per light to derive the receiver-plane depth bias.
    let dpos_dx = dpdx(in.world_pos);
    let dpos_dy = dpdy(in.world_pos);

    // Sample albedo texture and combine with vertex/object color
    let albedo_tex = textureSample(t_diffuse, s_diffuse, in.tex_coord);
    let base_color = in.vert_color * object.color;
    let albedo = (albedo_tex * base_color).rgb;

    // Get PBR parameters - either from textures or uniforms
    var metallic = object.metallic;
    var roughness = object.roughness;

    if object.has_metallic_roughness_map > 0.5 {
        let mr = textureSample(t_metallic_roughness, s_metallic_roughness, in.tex_coord);
        // glTF convention: B = metallic, G = roughness
        metallic = mr.b;
        roughness = mr.g;
    }

    // Clamp roughness to prevent artifacts
    roughness = clamp(roughness, 0.04, 1.0);

    // Get normal - either from normal map or geometry
    var N = normalize(in.world_normal);

    if object.has_normal_map > 0.5 {
        let normal_sample = textureSample(t_normal, s_normal, in.tex_coord).rgb;
        let tangent_normal = normal_sample * 2.0 - 1.0;

        // Generate tangent space (simple method based on normal)
        var tangent: vec3<f32>;
        let c1 = cross(N, vec3<f32>(0.0, 0.0, 1.0));
        let c2 = cross(N, vec3<f32>(0.0, 1.0, 0.0));
        if length(c1) > length(c2) {
            tangent = normalize(c1);
        } else {
            tangent = normalize(c2);
        }
        let bitangent = normalize(cross(N, tangent));
        let TBN = mat3x3<f32>(tangent, bitangent, N);
        N = normalize(TBN * tangent_normal);
    }

    // Transform normal to view space for lighting
    let view_mat3 = mat3x3<f32>(
        frame.view[0].xyz,
        frame.view[1].xyz,
        frame.view[2].xyz
    );
    let N_view = normalize(view_mat3 * N);

    // Sample ambient occlusion
    var ao = 1.0;
    if object.has_ao_map > 0.5 {
        ao = textureSample(t_ao, s_ao, in.tex_coord).r;
    }

    // Sample emissive
    var emissive = object.emissive.rgb;
    if object.has_emissive_map > 0.5 {
        let emissive_sample = textureSample(t_emissive, s_emissive, in.tex_coord).rgb;
        emissive = emissive * emissive_sample;
    }

    // View vector (in view space)
    let V = normalize(-in.view_pos);

    // Linear (alpha) roughness for the analytic lobes.
    let alpha = max(roughness * roughness, 1e-4);

    // Reflectance at normal incidence (F0). Dielectrics use the reflectance
    // remap `0.16·reflectance²` (0.04 at the default 0.5) tinted by specular_tint;
    // metals use the albedo.
    let dielectric_f0 = 0.16 * object.reflectance * object.reflectance * object.specular_tint.rgb;
    let F0 = mix(dielectric_f0, albedo, metallic);

    // Anisotropy basis: a tangent/bitangent in view space, rotated about the
    // normal by `anisotropy_rotation`.
    let aniso = object.anisotropy;
    var aniso_t = cross(vec3<f32>(0.0, 1.0, 0.0), N_view);
    if dot(aniso_t, aniso_t) < 1e-6 {
        aniso_t = cross(vec3<f32>(1.0, 0.0, 0.0), N_view);
    }
    aniso_t = normalize(aniso_t);
    let ar = object.anisotropy_rotation;
    aniso_t = normalize(aniso_t * cos(ar) + cross(N_view, aniso_t) * sin(ar));
    let aniso_b = cross(N_view, aniso_t);
    let at = max(alpha * (1.0 + aniso), 1e-4);
    let ab = max(alpha * (1.0 - aniso), 1e-4);

    // Clearcoat linear roughness.
    let cc_alpha = max(object.clearcoat_roughness * object.clearcoat_roughness, 1e-4);

    // Accumulate lighting from all lights
    var Lo = vec3<f32>(0.0);

    for (var i = 0u; i < frame.num_lights; i++) {
        let light = frame.lights[i];

        // Calculate light direction and attenuation based on light type
        var L: vec3<f32>;
        var attenuation: f32 = 1.0;
        var light_intensity = light.intensity;

        if light.light_type == LIGHT_TYPE_POINT {
            // Point light: calculate direction from light position to fragment
            let light_pos_view = (frame.view * vec4<f32>(light.position, 1.0)).xyz;
            let light_vec = light_pos_view - in.view_pos;
            let dist = length(light_vec);
            L = normalize(light_vec);
            attenuation = calculate_point_attenuation(dist, light.attenuation_radius);
        } else if light.light_type == LIGHT_TYPE_DIRECTIONAL {
            // Directional light: use light direction directly
            let light_dir_view = normalize(view_mat3 * light.direction);
            L = -light_dir_view;  // Light direction points FROM light, we need TO light
        } else {
            // Spot light: calculate direction and angular attenuation
            let light_pos_view = (frame.view * vec4<f32>(light.position, 1.0)).xyz;
            let light_dir_view = normalize(view_mat3 * light.direction);
            let light_vec = light_pos_view - in.view_pos;
            let dist = length(light_vec);
            L = normalize(light_vec);
            attenuation = calculate_spot_attenuation(
                L,
                light_dir_view,
                dist,
                light.inner_cone_cos,
                light.outer_cone_cos,
                light.attenuation_radius
            );
        }

        // Skip if no contribution
        if attenuation <= 0.0 {
            continue;
        }

        let H = normalize(V + L);
        let NoV = max(dot(N_view, V), 1e-4);
        let NdotL_raw = dot(N_view, L);
        let NoL = max(NdotL_raw, 0.0);
        let NoH = clamp(dot(N_view, H), 0.0, 1.0);
        let LoH = clamp(dot(L, H), 0.0, 1.0);

        let F = fresnel_schlick(max(dot(H, V), 0.0), F0);

        // Base specular lobe — anisotropic GGX when anisotropy is set, otherwise
        // the isotropic height-correlated form.
        var D: f32;
        var Vis: f32;
        if abs(aniso) > 1e-3 {
            let ToV = dot(aniso_t, V);
            let BoV = dot(aniso_b, V);
            let ToL = dot(aniso_t, L);
            let BoL = dot(aniso_b, L);
            let ToH = dot(aniso_t, H);
            let BoH = dot(aniso_b, H);
            D = d_ggx_aniso(at, ab, ToH, BoH, NoH);
            Vis = v_smith_correlated_aniso(at, ab, ToV, BoV, ToL, BoL, NoV, NoL);
        } else {
            D = d_ggx_alpha(NoH, alpha);
            Vis = v_smith_correlated(NoV, NoL, alpha);
        }
        var specular = D * Vis * F;

        // Clearcoat: a thin, smooth dielectric lobe layered on top of the base,
        // which it attenuates by (1 - Fc).
        var cc_atten = 1.0;
        if object.clearcoat > 0.0 {
            let dc = d_ggx_alpha(NoH, cc_alpha);
            let vc = v_kelemen(LoH);
            let fc = fresnel_schlick_scalar(LoH, 0.04) * object.clearcoat;
            specular = specular * (1.0 - fc) + vec3<f32>(dc * vc * fc);
            cc_atten = 1.0 - fc;
        }

        // Energy conservation: diffuse gets what the specular + clearcoat leave.
        let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic) * cc_atten;

        // Wrapped diffuse (half-Lambert) for softer terminators on back faces.
        let wrap = 0.2;
        let diffuse_wrap = max(NdotL_raw * (1.0 - wrap) + wrap, 0.0);

        // === SHADOW MAPPING: attenuate this light by its shadow-map visibility ===
        let shadow_factor = compute_shadow(i, in.world_pos, dpos_dx, dpos_dy);
        // === END SHADOW MAPPING ===

        let radiance = light.color * light_intensity * attenuation * shadow_factor;

        // Diffuse transmission: a fraction of the diffuse is instead lit from the
        // back side of the surface (cheap translucency).
        let diffuse_contrib =
            kD * albedo / PI * diffuse_wrap * (1.0 - object.transmission);
        var transmit_contrib = vec3<f32>(0.0);
        if object.transmission > 0.0 {
            let back = max(-NdotL_raw, 0.0);
            transmit_contrib = object.transmission * albedo / PI * back;
        }
        let specular_contrib = specular * NoL;
        Lo += (diffuse_contrib + transmit_contrib + specular_contrib) * radiance;
    }

    // Ambient lighting using configurable intensity + color from frame uniforms
    let ambient = frame.ambient_color.rgb * frame.ambient_intensity * albedo * ao;

    // Final color
    var color = ambient + Lo + emissive;

    // Distance fog (applied to the lit color; uses view distance + world height).
    color = apply_fog(color, length(in.view_pos), in.world_pos.y);

    return vec4<f32>(color, albedo_tex.a * base_color.a);
}

// Opaque pass: write the shaded color straight into the HDR film. Handles the
// opaque-phase alpha modes: Opaque (alpha forced to 1) and Mask (cutout discard).
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = shade(in);
    let mode = u32(object.alpha_mode + 0.5);
    // Mask: discard fragments below the cutoff.
    if mode == 1u && c.a < object.alpha_cutoff {
        discard;
    }
    // Opaque and Mask write a fully opaque pixel.
    if mode == 0u || mode == 1u {
        return vec4<f32>(c.rgb, 1.0);
    }
    return c;
}

// Weighted-Blended OIT output (McGuire & Bavoil 2013): an additive
// premultiplied-weighted color accumulator and a multiplicative revealage.
struct OitOutput {
    @location(0) accum: vec4<f32>,
    @location(1) reveal: f32,
}

// Transparent pass: emit the weighted-blended OIT contributions instead of
// blending directly, so transparency is order-independent (no sorting).
@fragment
fn fs_oit(in: VertexOutput) -> OitOutput {
    let c = shade(in);
    let a = c.a;
    // Depth-based weight: nearer fragments dominate (McGuire eq. 9). `view_pos.z`
    // is negative in front of the camera, so use its magnitude.
    let z = abs(in.view_pos.z);
    let w = clamp(10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0)), 1e-2, 3e3);

    // Premultiplied-alpha surfaces already carry color * alpha; straight-alpha
    // (Blend) surfaces are premultiplied here for the weighted accumulation.
    let mode = u32(object.alpha_mode + 0.5);
    var premult = c.rgb * a;
    if mode == 3u {
        premult = c.rgb;
    }

    var out: OitOutput;
    out.accum = vec4<f32>(premult, a) * w;
    out.reveal = a;
    return out;
}
