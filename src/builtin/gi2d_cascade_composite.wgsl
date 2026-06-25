import package::common::fullscreen_uv_from_clip;
// Resolves cascade 0 of the radiance cascades into per-pixel irradiance and modulates
// the scene by it. For each pixel we find the surrounding cascade-0 probes (grid
// spacing `s0`), average each probe's `e0 x e0` direction tile into that probe's
// irradiance, bilinearly interpolate across the four nearest probes, and multiply the
// scene by ambient + irradiance.

@group(0) @binding(0)
var t_scene: texture_2d<f32>;
@group(0) @binding(1)
var s_scene: sampler;

@group(1) @binding(0)
var t_cascade0: texture_2d<f32>;

struct CompositeUniforms {
    // e0, dir_count0 (= e0*e0), probesX0, probesY0
    v0: vec4<f32>,
    // fieldW, fieldH, s0, _
    v1: vec4<f32>,
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

fn probe_irradiance(probe: vec2<i32>, e: i32, dir_count: i32, probes: vec2<i32>) -> vec3<f32> {
    let pc = clamp(probe, vec2<i32>(0), probes - vec2<i32>(1));
    var sum = vec3<f32>(0.0);
    for (var d = 0; d < dir_count; d = d + 1) {
        let coord = pc * e + vec2<i32>(d % e, d / e);
        sum += textureLoad(t_cascade0, coord, 0).rgb;
    }
    return sum / f32(dir_count);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let e0 = i32(u.v0.x);
    let dir_count = i32(u.v0.y);
    let probes = vec2<i32>(i32(u.v0.z), i32(u.v0.w));
    let field_size = u.v1.xy;
    let s0 = u.v1.z;

    let pix = in.tex_coord * field_size;
    let pf = pix / s0 - vec2<f32>(0.5);
    let p0 = vec2<i32>(floor(pf));
    let fr = pf - floor(pf);

    let i00 = probe_irradiance(p0 + vec2<i32>(0, 0), e0, dir_count, probes);
    let i10 = probe_irradiance(p0 + vec2<i32>(1, 0), e0, dir_count, probes);
    let i01 = probe_irradiance(p0 + vec2<i32>(0, 1), e0, dir_count, probes);
    let i11 = probe_irradiance(p0 + vec2<i32>(1, 1), e0, dir_count, probes);
    let irradiance = mix(mix(i00, i10, fr.x), mix(i01, i11, fr.x), fr.y);

    let scene = textureSample(t_scene, s_scene, in.tex_coord).rgb;
    return vec4<f32>(scene * (u.ambient.rgb + irradiance), 1.0);
}
