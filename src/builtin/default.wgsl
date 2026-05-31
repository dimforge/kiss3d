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
    _padding: f32,
    // Far view-space distance of each directional cascade (0..num_cascades).
    cascade_splits: vec4<f32>,
}

@group(3) @binding(0)
var t_shadow_atlas: texture_depth_2d_array;
@group(3) @binding(1)
var s_shadow: sampler_comparison;
@group(3) @binding(2)
var<uniform> shadow: ShadowUniforms;

// PCF (5x5) comparison sample of one atlas layer at a shadow-map `uv` + depth.
// Returns 1.0 = fully lit, 0.0 = fully shadowed. The comparison sampler does a
// bilinear (2x2) hardware test per tap, so a 5x5 tap grid yields a smooth ~6-texel
// penumbra that hides residual stair-stepping on shadow edges.
// One hardware bilinear (2x2) comparison sample of an atlas layer.
fn shadow_tap(layer: u32, uv: vec2<f32>, compare_depth: f32) -> f32 {
    return textureSampleCompareLevel(t_shadow_atlas, s_shadow, uv, i32(layer), compare_depth);
}

// Castaño 2013 ("Shadow Mapping Summary Part 1") optimized PCF: a tent-weighted
// 3x3 grid of hardware bilinear comparison taps. Each tap is placed at a fractional
// offset so the GPU's 2x2 bilinear comparison, combined with the tent weights,
// reproduces a smooth ~5x5 PCF using only 9 taps — smoother AND crisper than a
// naive box PCF for the same cost. Returns visibility in [0,1].
fn shadow_pcf(layer: u32, uv: vec2<f32>, compare_depth: f32) -> f32 {
    let map_size = vec2<f32>(textureDimensions(t_shadow_atlas));
    let inv_size = 1.0 / map_size;

    let coord = uv * map_size;
    var base_uv = floor(coord + 0.5);
    let s = coord.x + 0.5 - base_uv.x;
    let t = coord.y + 0.5 - base_uv.y;
    base_uv = (base_uv - 0.5) * inv_size;

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

    var sum = 0.0;
    sum += uw0 * vw0 * shadow_tap(layer, base_uv + vec2<f32>(u0, v0) * inv_size, compare_depth);
    sum += uw1 * vw0 * shadow_tap(layer, base_uv + vec2<f32>(u1, v0) * inv_size, compare_depth);
    sum += uw2 * vw0 * shadow_tap(layer, base_uv + vec2<f32>(u2, v0) * inv_size, compare_depth);

    sum += uw0 * vw1 * shadow_tap(layer, base_uv + vec2<f32>(u0, v1) * inv_size, compare_depth);
    sum += uw1 * vw1 * shadow_tap(layer, base_uv + vec2<f32>(u1, v1) * inv_size, compare_depth);
    sum += uw2 * vw1 * shadow_tap(layer, base_uv + vec2<f32>(u2, v1) * inv_size, compare_depth);

    sum += uw0 * vw2 * shadow_tap(layer, base_uv + vec2<f32>(u0, v2) * inv_size, compare_depth);
    sum += uw1 * vw2 * shadow_tap(layer, base_uv + vec2<f32>(u1, v2) * inv_size, compare_depth);
    sum += uw2 * vw2 * shadow_tap(layer, base_uv + vec2<f32>(u2, v2) * inv_size, compare_depth);

    return sum * (1.0 / 144.0);
}

// Samples one atlas layer at a world position (project + bounds-check + PCF), used
// by spot and point lights. Returns 1.0 (lit) when the fragment is out of the map.
fn sample_shadow_layer(layer: u32, world_pos: vec3<f32>) -> f32 {
    let light_clip = shadow.view_proj[layer] * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = light_clip.xyz / light_clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0 || ndc.z < 0.0 {
        return 1.0;
    }
    return shadow_pcf(layer, uv, ndc.z - shadow.depth_bias);
}

// Projects `world_pos` into one cascade's atlas layer and samples it with PCF.
// Returns 1.0 (fully lit) if the fragment falls outside that layer's map.
fn sample_one_cascade(layer: u32, world_pos: vec3<f32>) -> f32 {
    let light_clip = shadow.view_proj[layer] * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = light_clip.xyz / light_clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0 || ndc.z < 0.0 {
        return 1.0;
    }
    return shadow_pcf(layer, uv, ndc.z - shadow.depth_bias);
}

// Cascaded shadow maps for a directional light: the `num_cascades` layers from
// `base_view` are nested frustum slices ordered near -> far. Select the cascade by
// the fragment's view-space depth, and cross-fade into the next cascade over a band
// before each boundary so the resolution change isn't a hard seam.
fn sample_directional_cascades(
    base_view: u32, num_cascades: u32, view_depth: f32, world_pos: vec3<f32>,
) -> f32 {
    // Pick the first cascade whose far bound is beyond the fragment depth.
    var c = num_cascades - 1u;
    for (var i = 0u; i < num_cascades; i = i + 1u) {
        if view_depth < shadow.cascade_splits[i] {
            c = i;
            break;
        }
    }

    let s = sample_one_cascade(base_view + c, world_pos);

    // Blend into the next cascade across a band before this cascade's far split.
    if c + 1u < num_cascades {
        let split = shadow.cascade_splits[c];
        let band = split * 0.2;
        if view_depth > split - band {
            let s_next = sample_one_cascade(base_view + c + 1u, world_pos);
            let t = clamp((view_depth - (split - band)) / band, 0.0, 1.0);
            return mix(s, s_next, t);
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

// Returns the light visibility factor in [0,1] for light `light_index` at `world_pos`.
fn compute_shadow(light_index: u32, world_pos: vec3<f32>) -> f32 {
    if shadow.shadows_enabled < 0.5 {
        return 1.0;
    }
    let ls = shadow.lights[light_index];
    if ls.enabled < 0.5 {
        return 1.0;
    }

    if ls.light_type == LIGHT_TYPE_POINT {
        let face = point_cube_face(world_pos - ls.light_pos);
        return sample_shadow_layer(ls.base_view + face, world_pos);
    }
    if ls.light_type == LIGHT_TYPE_DIRECTIONAL {
        // Cascaded shadow maps: select/blend cascades by view-space depth (the
        // distance in front of the camera; view -z is forward).
        let view_depth = -(frame.view * vec4<f32>(world_pos, 1.0)).z;
        return sample_directional_cascades(ls.base_view, ls.num_views, view_depth, world_pos);
    }
    // Spot lights use a single perspective view.
    return sample_shadow_layer(ls.base_view, world_pos);
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

    // Calculate reflectance at normal incidence (F0)
    // Dielectric: 0.04, Metal: albedo
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

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

        // Cook-Torrance BRDF
        let NDF = distribution_ggx(N_view, H, roughness);
        let G = geometry_smith(N_view, V, L, roughness);
        let F = fresnel_schlick(max(dot(H, V), 0.0), F0);

        // Specular contribution
        let numerator = NDF * G * F;
        let denominator = 4.0 * max(dot(N_view, V), 0.0) * max(dot(N_view, L), 0.0) + 0.0001;
        let specular = numerator / denominator;

        // Energy conservation: diffuse + specular = 1
        let kS = F;
        var kD = vec3<f32>(1.0) - kS;
        kD *= 1.0 - metallic;  // Metals have no diffuse

        // Lighting calculation
        let NdotL_raw = dot(N_view, L);
        let NdotL = max(NdotL_raw, 0.0);

        // Wrapped diffuse (half-Lambert) for softer shadows on back-facing triangles
        let wrap = 0.2;
        let NdotL_wrapped = (NdotL_raw * (1.0 - wrap) + wrap);
        let diffuse_wrap = max(NdotL_wrapped, 0.0);

        // === SHADOW MAPPING: attenuate this light by its shadow-map visibility ===
        let shadow_factor = compute_shadow(i, in.world_pos);
        // === END SHADOW MAPPING ===

        // Combine lighting with light color
        let radiance = light.color * light_intensity * attenuation * shadow_factor;
        let diffuse_contrib = kD * albedo / PI * diffuse_wrap;
        let specular_contrib = specular * NdotL;
        Lo += (diffuse_contrib + specular_contrib) * radiance;
    }

    // Ambient lighting using configurable intensity from frame uniforms
    let ambient = vec3<f32>(frame.ambient_intensity) * albedo * ao;

    // Final color
    var color = ambient + Lo + emissive;

    return vec4<f32>(color, albedo_tex.a * base_color.a);
}

// Opaque pass: write the shaded color straight into the HDR film.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return shade(in);
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

    var out: OitOutput;
    out.accum = vec4<f32>(c.rgb * a, a) * w;
    out.reveal = a;
    return out;
}
