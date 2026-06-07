// 4x4 box blur of the raw SSAO buffer, to smooth the per-pixel kernel rotation
// noise before the AO modulates ambient lighting.

@group(0) @binding(0) var t_ao: texture_2d<f32>;
@group(0) @binding(1) var s_pt: sampler;
@group(0) @binding(2) var<uniform> inv_resolution: vec4<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var c = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let xy = c[vid];
    var o: VsOut;
    o.pos = vec4<f32>(xy, 0.0, 1.0);
    o.uv = vec2<f32>((xy.x + 1.0) * 0.5, (1.0 - xy.y) * 0.5);
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var sum = 0.0;
    for (var x = -2; x < 2; x = x + 1) {
        for (var y = -2; y < 2; y = y + 1) {
            let off = vec2<f32>(f32(x), f32(y)) * inv_resolution.xy;
            sum += textureSampleLevel(t_ao, s_pt, in.uv + off, 0.0).r;
        }
    }
    return vec4<f32>(sum / 16.0);
}
