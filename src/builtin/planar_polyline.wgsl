// Planar (2D) polyline shader for thick line rendering
// Based on bevy_polyline but simplified for 2D (no near-plane clipping, no depth)
//
// Uses instanced rendering where each instance is a line segment.
// Material data (color, width) is passed per-instance via vertex attributes.

// View uniforms
// Note: mat3x3 is stored as array<vec4<f32>, 3> for proper alignment
struct ViewUniforms {
    view_0: vec4<f32>,
    view_1: vec4<f32>,
    view_2: vec4<f32>,
    proj_0: vec4<f32>,
    proj_1: vec4<f32>,
    proj_2: vec4<f32>,
    viewport: vec4<f32>, // x, y, width, height
}

@group(0) @binding(0)
var<uniform> view: ViewUniforms;

// Vertex input - line segment endpoints with material data (per-instance)
struct VertexInput {
    @location(0) point_a: vec2<f32>,
    @location(1) width: f32,
    @location(2) point_b: vec2<f32>,
    @location(3) color: vec4<f32>,
    @builtin(vertex_index) index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
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
fn vs_main(vertex: VertexInput) -> VertexOutput {
    // 6 vertices per line segment forming 2 triangles
    // Positions encode: y = perpendicular offset (-0.5 or 0.5), z = along line (0 or 1)
    var positions = array<vec3<f32>, 6u>(
        vec3(0.0, -0.5, 0.0),
        vec3(0.0, -0.5, 1.0),
        vec3(0.0, 0.5, 1.0),
        vec3(0.0, -0.5, 0.0),
        vec3(0.0, 0.5, 1.0),
        vec3(0.0, 0.5, 0.0)
    );
    let position = positions[vertex.index % 6u];

    // Reconstruct view and projection matrices
    let view_mat = unpack_mat3(view.view_0, view.view_1, view.view_2);
    let proj_mat = unpack_mat3(view.proj_0, view.proj_1, view.proj_2);
    let view_proj = proj_mat * view_mat;

    // Transform endpoints to clip space
    let clip0 = view_proj * vec3(vertex.point_a, 1.0);
    let clip1 = view_proj * vec3(vertex.point_b, 1.0);

    // For 2D, w is typically 1.0, so screen space is just xy
    let resolution = vec2(view.viewport.z, view.viewport.w);
    let screen0 = resolution * (0.5 * clip0.xy + 0.5);
    let screen1 = resolution * (0.5 * clip1.xy + 0.5);

    // Calculate basis vectors for the line in screen space
    let line_dir = screen1 - screen0;
    let line_length = length(line_dir);

    // Handle degenerate case where points are at the same location
    var x_basis: vec2<f32>;
    var y_basis: vec2<f32>;
    if line_length > 0.001 {
        x_basis = line_dir / line_length;
        y_basis = vec2(-x_basis.y, x_basis.x);
    } else {
        x_basis = vec2(1.0, 0.0);
        y_basis = vec2(0.0, 1.0);
    }

    let line_width = vertex.width;

    // Calculate offset from line center
    let pt_offset = line_width * (position.x * x_basis + position.y * y_basis);
    let pt0 = screen0 + pt_offset;
    let pt1 = screen1 + pt_offset;
    let pt = mix(pt0, pt1, position.z);

    var out: VertexOutput;
    // Convert back from screen space to clip space
    out.clip_position = vec4((2.0 * pt) / resolution - 1.0, 0.0, 1.0);
    out.color = vertex.color;
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
    return vec4<f32>(linear_to_srgb(in.color.rgb), in.color.a);
}
