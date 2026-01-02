// Point rendering shader for kiss3d
// Used for rendering debug points with configurable size
//
// Uses storage buffer for point data (position + color).
// Draw call: draw(0..(6 * num_points), 0..1)

// Frame uniforms (bind group 0)
struct FrameUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport: vec4<f32>, // x, y, width, height
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

// Point data storage buffer (bind group 0, binding 1)
struct PointData {
    position: vec3<f32>,
    size: f32,       // Per-point size (uses default if <= 0)
    color: vec4<f32>,
}

@group(0) @binding(1)
var<storage, read> points: array<PointData>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
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

    // Transform to clip space
    let view_proj = frame.proj * frame.view;
    let clip = view_proj * vec4(point.position, 1.0);

    // Skip points behind camera
    if clip.w <= 0.0 {
        var out: VertexOutput;
        out.clip_position = vec4(0.0, 0.0, -1.0, 1.0); // Behind near plane
        out.color = vec4(0.0);
        return out;
    }

    // Calculate screen-space position
    let resolution = vec2(frame.viewport.z, frame.viewport.w);
    let screen_center = resolution * (0.5 * clip.xy / clip.w + 0.5);

    // Calculate offset from point center (offset is -0.5 to 0.5, multiply by size)
    let pt = screen_center + offset * point.size;

    var out: VertexOutput;
    out.clip_position = vec4(clip.w * ((2.0 * pt) / resolution - 1.0), clip.z, clip.w);
    out.color = point.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
