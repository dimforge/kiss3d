// 2D point rendering shader for kiss3d
// Used for rendering debug points with configurable size in 2D scenes
//
// Uses storage buffer for point data (position + color).
// Draw call: draw(0..(6 * num_points), 0..1)

// Frame uniforms (bind group 0)
// Note: mat3x3 is stored as array<vec4<f32>, 3> for proper alignment
struct FrameUniforms {
    view_0: vec4<f32>,
    view_1: vec4<f32>,
    view_2: vec4<f32>,
    proj_0: vec4<f32>,
    proj_1: vec4<f32>,
    proj_2: vec4<f32>,
    viewport: vec4<f32>, // x, y, width, height
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

// Point data storage buffer (bind group 0, binding 1)
struct PointData {
    position: vec2<f32>,
    size: f32,       // Per-point size (uses default if <= 0)
    _pad: f32,
    color: vec3<f32>,
    _pad2: f32,
}

@group(0) @binding(1)
var<storage, read> points: array<PointData>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

// Reconstruct mat3x3 from padded vec4 columns
fn unpack_mat3(col0: vec4<f32>, col1: vec4<f32>, col2: vec4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        col0.xyz,
        col1.xyz,
        col2.xyz
    );
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Compute which point from vertex_index
    // Draw call: draw(0..(6 * num_points), 0..1)
    // Each 6 vertices form one point quad
    let point_index = vertex_index / 6u;
    let point = points[point_index];

    // 6 vertices per point forming 2 triangles (centered quad)
    // x, y are offsets from center (-0.5 to 0.5)
    var positions = array<vec2<f32>, 6u>(
        vec2(-0.5, -0.5),
        vec2( 0.5, -0.5),
        vec2( 0.5,  0.5),
        vec2(-0.5, -0.5),
        vec2( 0.5,  0.5),
        vec2(-0.5,  0.5)
    );
    let offset = positions[vertex_index % 6u];

    // Reconstruct matrices from uniform data
    let view_mat = unpack_mat3(frame.view_0, frame.view_1, frame.view_2);
    let proj_mat = unpack_mat3(frame.proj_0, frame.proj_1, frame.proj_2);

    // Transform through view and projection
    let view_proj = proj_mat * view_mat;
    let clip = view_proj * vec3(point.position, 1.0);

    // Calculate screen-space position
    let resolution = vec2(frame.viewport.z, frame.viewport.w);
    let screen_center = resolution * (0.5 * clip.xy + 0.5);

    // Calculate offset from point center (offset is -0.5 to 0.5, multiply by size)
    let pt = screen_center + offset * point.size;

    var out: VertexOutput;
    out.clip_position = vec4((2.0 * pt) / resolution - 1.0, 0.0, 1.0);
    out.color = point.color;
    return out;
}

// Convert linear RGB to sRGB for display.
fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = linear < vec3<f32>(0.0031308);
    let lower = linear * 12.92;
    let higher = pow(linear, vec3<f32>(1.0 / 2.4)) * 1.055 - vec3<f32>(0.055);
    return select(higher, lower, cutoff);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(linear_to_srgb(in.color), 1.0);
}
