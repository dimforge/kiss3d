// Default material shader for kiss3d
// Implements Blinn-Phong lighting with texture support and instancing

// Bind group 0: Frame uniforms (view, projection, light)
struct FrameUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    light_position: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

// Bind group 1: Object uniforms (transform, scale, color)
struct ObjectUniforms {
    transform: mat4x4<f32>,
    ntransform: mat3x3<f32>,
    scale: mat3x3<f32>,
    color: vec3<f32>,
    _padding: f32,
}

@group(1) @binding(0)
var<uniform> object: ObjectUniforms;

// Bind group 2: Texture and sampler
@group(2) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(2) @binding(1)
var s_diffuse: sampler;

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
    @location(1) normal_interp: vec3<f32>,
    @location(2) vert_pos: vec3<f32>,
    @location(3) vert_color: vec4<f32>,
    @location(4) local_light_position: vec3<f32>,
}

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

    // View-space position for lighting
    let view_pos = frame.view * world_pos;
    out.vert_pos = view_pos.xyz / view_pos.w;

    // Transform normal to view space
    let view_mat3 = mat3x3<f32>(
        frame.view[0].xyz,
        frame.view[1].xyz,
        frame.view[2].xyz
    );
    out.normal_interp = view_mat3 * deformation * object.ntransform * vertex.normal;

    out.tex_coord = vertex.tex_coord;
    out.local_light_position = (frame.view * vec4<f32>(frame.light_position, 1.0)).xyz;
    out.vert_color = instance.inst_color;

    return out;
}

// Convert linear RGB to sRGB for display.
// This is applied manually because we use a non-sRGB framebuffer for
// consistent behavior across native and web platforms.
fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = linear < vec3<f32>(0.0031308);
    let lower = linear * 12.92;
    let higher = pow(linear, vec3<f32>(1.0 / 2.4)) * 1.055 - vec3<f32>(0.055);
    return select(higher, lower, cutoff);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let specular_color = vec3<f32>(0.4, 0.4, 0.4);

    let normal = normalize(in.normal_interp);
    let light_dir = normalize(in.local_light_position - in.vert_pos);

    // Lambertian diffuse
    let lambertian = max(dot(light_dir, normal), 0.0);

    // Blinn-Phong specular
    var specular = 0.0;
    if lambertian > 0.0 {
        let view_dir = normalize(-in.vert_pos);
        let half_dir = normalize(light_dir + view_dir);
        let spec_angle = max(dot(half_dir, normal), 0.0);
        specular = pow(spec_angle, 30.0);
    }

    let base_color = in.vert_color * vec4<f32>(object.color, 1.0);
    let tex_color = textureSample(t_diffuse, s_diffuse, in.tex_coord);

    let final_color = tex_color * vec4<f32>(
        base_color.xyz / 3.0 +
        lambertian * base_color.xyz / 3.0 +
        specular * specular_color / 3.0,
        base_color.w
    );

    return vec4<f32>(linear_to_srgb(final_color.rgb), final_color.a);
}
