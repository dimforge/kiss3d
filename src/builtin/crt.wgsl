import package::common::fullscreen_uv_from_clip;
// CRT stylization post-process: screen curvature (barrel distortion), chromatic
// aberration, scanlines and a vignette. Reads the rendered scene and writes the
// stylized image to the output. Knobs come from `CrtUniforms`; any term can be
// disabled by zeroing its strength.

@group(0) @binding(0)
var t_fbo: texture_2d<f32>;
@group(0) @binding(1)
var s_fbo: sampler;

struct CrtUniforms {
    // Barrel-distortion strength (0 = flat).
    curvature: f32,
    // Chromatic-aberration strength (UV units at the screen edge).
    aberration: f32,
    // Scanline darkening intensity in [0, 1].
    scanline_intensity: f32,
    // Number of scanlines down the screen.
    scanline_count: f32,
    // Vignette strength in [0, 1].
    vignette: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(1) @binding(0)
var<uniform> uniforms: CrtUniforms;

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

// Barrel-distort UVs around the screen center to fake a curved CRT tube.
fn curve(uv: vec2<f32>, amount: f32) -> vec2<f32> {
    var c = uv * 2.0 - vec2<f32>(1.0);
    let offset = c.yx * c.yx * amount;
    c += c * offset;
    return c * 0.5 + vec2<f32>(0.5);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = curve(in.tex_coord, uniforms.curvature);

    // Outside the curved tube reads as black (the bezel).
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // Chromatic aberration: offset the R/B taps outward from the center, scaled by
    // distance so the fringing grows toward the edges.
    let from_center = uv - vec2<f32>(0.5);
    let offset = from_center * uniforms.aberration;
    let r = textureSample(t_fbo, s_fbo, uv + offset).r;
    let g = textureSample(t_fbo, s_fbo, uv).g;
    let b = textureSample(t_fbo, s_fbo, uv - offset).b;
    var color = vec3<f32>(r, g, b);

    // Scanlines: a sinusoidal darkening along rows.
    let scan = sin(uv.y * uniforms.scanline_count * 3.14159265);
    color *= 1.0 - uniforms.scanline_intensity * (0.5 + 0.5 * scan);

    // Vignette: darken toward the corners.
    let vig = 1.0 - uniforms.vignette * dot(from_center, from_center) * 2.0;
    color *= clamp(vig, 0.0, 1.0);

    return vec4<f32>(color, 1.0);
}
