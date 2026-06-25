import package::common::fullscreen_uv_from_clip;
// Full-resolution composite for 2D global illumination: samples the rendered scene
// and the low-resolution irradiance field (bilinearly upsampled by the sampler) and
// modulates the scene by ambient + irradiance — soft shadows and colored bleed.

@group(0) @binding(0)
var t_scene: texture_2d<f32>;
@group(0) @binding(1)
var s_scene: sampler;

@group(1) @binding(0)
var t_gi: texture_2d<f32>;
@group(1) @binding(1)
var s_gi: sampler;

struct CompositeUniforms {
    // ambient.rgb, _
    ambient: vec4<f32>,
}

@group(2) @binding(0)
var<uniform> u: CompositeUniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(vertex.position, 0.0, 1.0);
    out.tex_coord = fullscreen_uv_from_clip(vertex.position);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(t_scene, s_scene, in.tex_coord).rgb;
    // The GI field is lower-resolution; the linear sampler upsamples it bilinearly.
    let irradiance = textureSampleLevel(t_gi, s_gi, in.tex_coord, 0.0).rgb;
    let lit = scene * (u.ambient.rgb + irradiance);
    return vec4<f32>(lit, 1.0);
}
