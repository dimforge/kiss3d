// Polyline shader for thick line rendering
// Based on bevy_polyline (https://github.com/ForesightMiningSoftwareCorporation/bevy_polyline)
//
// Uses instanced rendering where each instance is a line segment.
// Material data (color, width, depth_bias) is passed per-instance via vertex attributes.
// Lines are drawn in world space (no model transform).

// View uniforms (bind group 0)
struct ViewUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport: vec4<f32>, // x, y, width, height
}

@group(0) @binding(0)
var<uniform> view: ViewUniforms;

// Vertex input - line segment endpoints with material data (per-instance)
struct VertexInput {
    @location(0) point_a: vec3<f32>,
    @location(1) width: f32,
    @location(2) point_b: vec3<f32>,
    @location(3) depth_bias: f32,
    @location(4) color: vec4<f32>,
    @location(5) perspective: u32,
    @builtin(vertex_index) index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

// Clip a point against the near plane
fn clip_near_plane(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    // Move a if a is behind the near plane and b is in front.
    if a.z > a.w && b.z <= b.w {
        // Interpolate a towards b until it's at the near plane.
        let distance_a = a.z - a.w;
        let distance_b = b.z - b.w;
        let t = distance_a / (distance_a - distance_b);
        return a + (b - a) * t;
    }
    return a;
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

    // Points are already in world space
    let world_a = vec4(vertex.point_a, 1.0);
    let world_b = vec4(vertex.point_b, 1.0);

    // Transform to clip space
    let view_proj = view.proj * view.view;
    var clip0 = view_proj * world_a;
    var clip1 = view_proj * world_b;

    // Manual near plane clipping to avoid errors when doing the perspective divide
    clip0 = clip_near_plane(clip0, clip1);
    clip1 = clip_near_plane(clip1, clip0);

    // Interpolate along the line based on position.z
    let clip = mix(clip0, clip1, position.z);

    // Calculate screen-space positions
    let resolution = vec2(view.viewport.z, view.viewport.w);
    let screen0 = resolution * (0.5 * clip0.xy / clip0.w + 0.5);
    let screen1 = resolution * (0.5 * clip1.xy / clip1.w + 0.5);

    // Calculate basis vectors for the line in screen space
    let line_dir = screen1 - screen0;
    let line_length = length(line_dir);

    // Handle degenerate case where points project to same location
    var x_basis: vec2<f32>;
    var y_basis: vec2<f32>;
    if line_length > 0.001 {
        x_basis = line_dir / line_length;
        y_basis = vec2(-x_basis.y, x_basis.x);
    } else {
        x_basis = vec2(1.0, 0.0);
        y_basis = vec2(0.0, 1.0);
    }

    var line_width = vertex.width;
    var color = vertex.color;

    // Perspective mode: width varies with distance (thinner when further away)
    if vertex.perspective != 0u {
        line_width = line_width / clip.w;
        // Line thinness fade for anti-aliasing when line becomes sub-pixel
        if line_width > 0.0 && line_width < 1.0 {
            color.a = color.a * line_width;
            line_width = 1.0;
        }
    }

    // Calculate offset from line center
    let pt_offset = line_width * (position.x * x_basis + position.y * y_basis);
    let pt0 = screen0 + pt_offset;
    let pt1 = screen1 + pt_offset;
    let pt = mix(pt0, pt1, position.z);

    // Apply depth bias
    var depth = clip.z;
    let depth_bias = vertex.depth_bias;
    if depth_bias >= 0.0 {
        depth = depth * (1.0 - depth_bias);
    } else {
        let epsilon = 4.88e-04;
        // Exponential depth bias for easier user control
        depth = depth * exp2(-depth_bias * log2(clip.w / depth - epsilon));
    }

    var out: VertexOutput;
    out.clip_position = vec4(clip.w * ((2.0 * pt) / resolution - 1.0), depth, clip.w);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
