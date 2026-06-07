// Box-downsamples one mip of an equirectangular environment map into the next,
// used to build the mip chain that the rasterizer's image-based lighting samples
// (coarser mips approximate rougher pre-filtered reflections / irradiance).

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let xy = corners[vid];
    var out: VsOut;
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    out.uv = vec2<f32>((xy.x + 1.0) * 0.5, (1.0 - xy.y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Bilinear sample at mip 0 of the bound (previous-mip) view averages a 2x2
    // block of the source into each destination texel.
    return textureSampleLevel(src, samp, in.uv, 0.0);
}
