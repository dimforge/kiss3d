import package::common::{unpack_mat2, unpack_mat3};
// 2D lit material: diffuse + specular shading from dynamic 2D lights, with optional
// normal mapping. Lights live slightly above the plane (their height), so a
// normal-mapped sprite reacts with per-pixel shading; without a normal map the
// surface is flat (+Z) and lights contribute a smooth radial falloff.

const MAX_LIGHTS: u32 = 16u;

struct Light {
    // position.xy, height, kind (0 = point, 1 = spot)
    pos_height: vec4<f32>,
    // color.rgb, intensity
    color_intensity: vec4<f32>,
    // direction.xy, cos(inner_angle), cos(outer_angle)
    dir_cone: vec4<f32>,
    // radius, _, _, _
    radius: vec4<f32>,
}

struct FrameUniforms {
    view_0: vec4<f32>,
    view_1: vec4<f32>,
    view_2: vec4<f32>,
    proj_0: vec4<f32>,
    proj_1: vec4<f32>,
    proj_2: vec4<f32>,
    // ambient.rgb, num_lights
    ambient_count: vec4<f32>,
    lights: array<Light, MAX_LIGHTS>,
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

struct ObjectUniforms {
    model_0: vec4<f32>,
    model_1: vec4<f32>,
    model_2: vec4<f32>,
    scale_0: vec4<f32>,
    scale_1: vec4<f32>,
    color: vec4<f32>,
    // specular_strength, shininess, normal_strength, has_normal_map
    params: vec4<f32>,
}

@group(1) @binding(0)
var<uniform> obj: ObjectUniforms;

@group(2) @binding(0)
var t_albedo: texture_2d<f32>;
@group(2) @binding(1)
var s_albedo: sampler;
@group(2) @binding(2)
var t_normal: texture_2d<f32>;
@group(2) @binding(3)
var s_normal: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coord: vec2<f32>,
}

struct InstanceInput {
    @location(2) inst_tra: vec2<f32>,
    @location(3) inst_color: vec4<f32>,
    @location(4) inst_def_0: vec2<f32>,
    @location(5) inst_def_1: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) world: vec2<f32>,
    @location(2) inst_color: vec4<f32>,
}

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    let view = unpack_mat3(frame.view_0, frame.view_1, frame.view_2);
    let proj = unpack_mat3(frame.proj_0, frame.proj_1, frame.proj_2);
    let model = unpack_mat3(obj.model_0, obj.model_1, obj.model_2);
    let scale = unpack_mat2(obj.scale_0, obj.scale_1);
    let def = mat2x2<f32>(instance.inst_def_0, instance.inst_def_1);

    let deformed = def * (scale * vertex.position);
    let model_pos = model * vec3<f32>(deformed, 1.0);
    let world = vec3<f32>(instance.inst_tra, 0.0) + model_pos;
    var projected = proj * view * world;
    projected.z = 0.0;

    out.clip_position = vec4<f32>(projected, 1.0);
    out.tex_coord = vertex.tex_coord;
    out.world = world.xy;
    out.inst_color = instance.inst_color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo = textureSample(t_albedo, s_albedo, in.tex_coord) * obj.color * in.inst_color;

    let specular_strength = obj.params.x;
    let shininess = max(obj.params.y, 1.0);
    let normal_strength = obj.params.z;
    let has_normal_map = obj.params.w;

    var n = vec3<f32>(0.0, 0.0, 1.0);
    if (has_normal_map > 0.5) {
        let sampled = textureSample(t_normal, s_normal, in.tex_coord).xyz * 2.0 - vec3<f32>(1.0);
        n = normalize(vec3<f32>(sampled.xy * normal_strength, max(sampled.z, 0.05)));
    }

    let ambient = frame.ambient_count.rgb;
    var lit = ambient * albedo.rgb;

    let frag = vec3<f32>(in.world, 0.0);
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    let count = u32(frame.ambient_count.w);

    for (var i = 0u; i < count; i = i + 1u) {
        let light = frame.lights[i];
        let radius = light.radius.x;
        let to_light = vec3<f32>(light.pos_height.xy, light.pos_height.z) - frag;
        let planar_dist = length(to_light.xy);
        if (planar_dist >= radius) {
            continue;
        }
        let l = normalize(to_light);

        // Smooth distance attenuation (quadratic falloff to zero at the radius).
        var atten = clamp(1.0 - planar_dist / radius, 0.0, 1.0);
        atten = atten * atten;

        // Spot cone factor (1 for point lights).
        var cone = 1.0;
        if (light.pos_height.w > 0.5) {
            let spot_dir = normalize(light.dir_cone.xy);
            let frag_dir = normalize(in.world - light.pos_height.xy);
            let cos_ang = dot(frag_dir, spot_dir);
            cone = smoothstep(light.dir_cone.w, light.dir_cone.z, cos_ang);
        }

        let n_dot_l = max(dot(n, l), 0.0);
        let diffuse = n_dot_l * atten * cone;

        var spec = 0.0;
        if (n_dot_l > 0.0 && specular_strength > 0.0) {
            let h = normalize(l + view_dir);
            spec = pow(max(dot(n, h), 0.0), shininess) * specular_strength * atten * cone;
        }

        let radiance = light.color_intensity.rgb * light.color_intensity.w;
        lit += radiance * (diffuse * albedo.rgb + spec);
    }

    return vec4<f32>(lit, albedo.a);
}
