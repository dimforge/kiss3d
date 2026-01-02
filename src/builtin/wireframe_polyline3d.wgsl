// Wireframe polyline shader for thick line rendering of mesh edges
// Based on bevy_polyline (https://github.com/ForesightMiningSoftwareCorporation/bevy_polyline)
//
// Uses storage buffer for edges and vertex buffers for instances (reusing InstancesBuffer).
// Draw call: draw(0..6, 0..(num_edges * num_instances))
// We compute which edge and which object instance from the instance_index.

// View uniforms (bind group 0)
struct ViewUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport: vec4<f32>, // x, y, width, height
}

// Model uniforms for per-object transform (bind group 1)
struct ModelUniforms {
    transform: mat4x4<f32>,  // Combined model transform (rotation + translation)
    scale: vec3<f32>,        // Non-uniform scale
    num_edges: u32,          // Number of edges in the edge buffer
    default_color: vec4<f32>, // Default wireframe color (used when instance alpha == 0)
    default_width: f32,      // Default wireframe width (used when instance width < 0)
    use_perspective: u32,    // Whether to scale width with distance (1 = yes, 0 = no)
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> view: ViewUniforms;

@group(1) @binding(0)
var<uniform> model: ModelUniforms;

// Edge storage buffer (bind group 1, binding 1)
struct Edge {
    point_a: vec3<f32>,
    _pad_a: f32,
    point_b: vec3<f32>,
    _pad_b: f32,
}

@group(1) @binding(1)
var<storage, read> edges: array<Edge>;

// Instance input from vertex buffers (reuses InstancesBuffer layout)
struct InstanceInput {
    @location(0) inst_position: vec3<f32>,      // positions buffer
    @location(1) inst_color: vec4<f32>,          // colors buffer (mesh color, not used for wireframe)
    @location(2) inst_def_0: vec3<f32>,          // deformations buffer (col 0)
    @location(3) inst_def_1: vec3<f32>,          // deformations buffer (col 1)
    @location(4) inst_def_2: vec3<f32>,          // deformations buffer (col 2)
    @location(5) inst_lines_color: vec4<f32>,    // lines_colors buffer
    @location(6) inst_lines_width: f32,          // lines_widths buffer
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

// Clip a point against the near plane
fn clip_near_plane(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    if a.z > a.w && b.z <= b.w {
        let distance_a = a.z - a.w;
        let distance_b = b.z - b.w;
        let t = distance_a / (distance_a - distance_b);
        return a + (b - a) * t;
    }
    return a;
}

@vertex
fn vs_main(
    instance: InstanceInput,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    // Compute which edge from vertex_index
    // Draw call: draw(0..(6 * num_edges), 0..num_instances)
    // Each 6 vertices form one edge quad
    let edge_index = vertex_index / 6u;
    let edge = edges[edge_index];

    // 6 vertices per line segment forming 2 triangles
    var positions = array<vec3<f32>, 6u>(
        vec3(0.0, -0.5, 0.0),
        vec3(0.0, -0.5, 1.0),
        vec3(0.0, 0.5, 1.0),
        vec3(0.0, -0.5, 0.0),
        vec3(0.0, 0.5, 1.0),
        vec3(0.0, 0.5, 0.0)
    );
    let position = positions[vertex_index % 6u];

    // Build deformation matrix from instance data
    let deformation = mat3x3<f32>(
        instance.inst_def_0,
        instance.inst_def_1,
        instance.inst_def_2
    );

    // Apply deformation, scale, model transform, and instance offset
    let deformed_a = deformation * edge.point_a;
    let deformed_b = deformation * edge.point_b;
    let scaled_a = deformed_a * model.scale;
    let scaled_b = deformed_b * model.scale;
    let model_a = model.transform * vec4(scaled_a, 1.0);
    let model_b = model.transform * vec4(scaled_b, 1.0);
    let world_a = model_a + vec4(instance.inst_position, 0.0);
    let world_b = model_b + vec4(instance.inst_position, 0.0);

    // Transform to clip space
    let view_proj = view.proj * view.view;
    var clip0 = view_proj * world_a;
    var clip1 = view_proj * world_b;

    // Manual near plane clipping
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

    var x_basis: vec2<f32>;
    var y_basis: vec2<f32>;
    if line_length > 0.001 {
        x_basis = line_dir / line_length;
        y_basis = vec2(-x_basis.y, x_basis.x);
    } else {
        x_basis = vec2(1.0, 0.0);
        y_basis = vec2(0.0, 1.0);
    }

    // Use instance width if >= 0, otherwise use default
    var line_width = instance.inst_lines_width;
    if line_width < 0.0 {
        line_width = model.default_width;
    }

    // Apply perspective scaling if enabled (scale width by 1/w to shrink with distance)
    if model.use_perspective != 0u {
        // Use the interpolated clip.w for perspective-correct scaling
        line_width = line_width / clip.w;
    }

    // Use instance color if alpha > 0, otherwise use default
    var color = instance.inst_lines_color;
    if color.a == 0.0 {
        color = model.default_color;
    }

    // Calculate offset from line center
    let pt_offset = line_width * (position.x * x_basis + position.y * y_basis);
    let pt0 = screen0 + pt_offset;
    let pt1 = screen1 + pt_offset;
    let pt = mix(pt0, pt1, position.z);

    var out: VertexOutput;
    out.clip_position = vec4(clip.w * ((2.0 * pt) / resolution - 1.0), clip.z, clip.w);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
