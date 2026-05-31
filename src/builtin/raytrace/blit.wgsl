// Minimal fullscreen blit used to pack arbitrarily-sized object textures into a
// single fixed-resolution 2D-array layer for the path tracer. A triangle-strip
// quad covers the target layer and samples the source with bilinear filtering.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Two triangles covering the viewport.
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    let c = corners[vid];
    var out: VsOut;
    out.pos = vec4<f32>(c, 0.0, 1.0);
    out.uv = vec2<f32>(0.5 * c.x + 0.5, 0.5 - 0.5 * c.y);
    return out;
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(src_tex, src_sampler, in.uv, 0.0);
}
