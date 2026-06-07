// Equirectangular skybox for the rasterizer.
//
// Draws a full-screen triangle into the HDR scene target before the opaque pass.
// For each pixel it reconstructs the world-space view ray from the inverse
// view-projection matrix and samples an equirectangular environment map. The
// direction→UV mapping is the shared `equirect_dir_to_uv` (so the rasterizer, SSR
// and the path tracer all show the same sky); the Y-axis rotation matches too.

import package::pbr_env::equirect_dir_to_uv;
import package::common::fullscreen_triangle_xy;

struct SkyUniforms {
    inv_view_proj: mat4x4<f32>,
    // (cos(rotation), sin(rotation), intensity, unused)
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SkyUniforms;
@group(0) @binding(1) var env_tex: texture_2d<f32>;
@group(0) @binding(2) var env_samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // z = 1 keeps the sky at the far plane; the pass uses no depth test/write.
    let xy = fullscreen_triangle_xy(vid);
    var out: VsOut;
    out.pos = vec4<f32>(xy, 1.0, 1.0);
    out.ndc = xy;
    return out;
}

fn env_rotate(rd: vec3<f32>) -> vec3<f32> {
    let c = u.params.x;
    let sn = u.params.y;
    return vec3<f32>(c * rd.x + sn * rd.z, rd.y, -sn * rd.x + c * rd.z);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Reconstruct the world-space ray direction through this pixel. The clip-space
    // z range is GL-style [-1, 1] to match kiss3d's `perspective_rh_gl` cameras.
    let near = u.inv_view_proj * vec4<f32>(in.ndc, -1.0, 1.0);
    let far = u.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - near.xyz / near.w);

    let uv = equirect_dir_to_uv(env_rotate(dir));
    let c = textureSampleLevel(env_tex, env_samp, uv, 0.0).rgb * u.params.z;
    return vec4<f32>(c, 1.0);
}
