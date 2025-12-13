// Planar wireframe polyline shader for thick line rendering of 2D mesh edges
// Adapted from 3D wireframe_polyline.wgsl for 2D planar rendering
//
// Uses storage buffer for edges and vertex buffers for instances (reusing PlanarInstancesBuffer).
// Draw call: draw(0..(6 * num_edges), 0..num_instances)
// We compute which edge from vertex_index.

// View uniforms (bind group 0)
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

// Model uniforms for per-object transform (bind group 1)
struct ModelUniforms {
    model_0: vec4<f32>,
    model_1: vec4<f32>,
    model_2: vec4<f32>,
    scale_0: vec4<f32>,
    scale_1: vec4<f32>,
    num_edges: u32,
    default_width: f32,
    use_perspective: u32,
    _padding1: f32,
    default_color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> view: ViewUniforms;

@group(1) @binding(0)
var<uniform> model: ModelUniforms;

// Edge storage buffer (bind group 1, binding 1)
struct Edge {
    point_a: vec2<f32>,
    point_b: vec2<f32>,
}

@group(1) @binding(1)
var<storage, read> edges: array<Edge>;

// Instance input from vertex buffers (reuses PlanarInstancesBuffer layout)
struct InstanceInput {
    @location(0) inst_position: vec2<f32>,      // positions buffer
    @location(1) inst_color: vec4<f32>,          // colors buffer (mesh color, not used for wireframe)
    @location(2) inst_def_0: vec2<f32>,          // deformations buffer (col 0)
    @location(3) inst_def_1: vec2<f32>,          // deformations buffer (col 1)
    @location(4) inst_lines_color: vec4<f32>,    // lines_colors buffer
    @location(5) inst_lines_width: f32,          // lines_widths buffer
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

// Reconstruct mat2x2 from padded vec4 columns
fn unpack_mat2(col0: vec4<f32>, col1: vec4<f32>) -> mat2x2<f32> {
    return mat2x2<f32>(
        col0.xy,
        col1.xy
    );
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

    // Reconstruct matrices from uniform data
    let view_mat = unpack_mat3(view.view_0, view.view_1, view.view_2);
    let proj_mat = unpack_mat3(view.proj_0, view.proj_1, view.proj_2);
    let model_mat = unpack_mat3(model.model_0, model.model_1, model.model_2);
    let scale_mat = unpack_mat2(model.scale_0, model.scale_1);

    // Build deformation matrix from instance data
    let deformation = mat2x2<f32>(
        instance.inst_def_0,
        instance.inst_def_1
    );

    // Apply deformation, scale, model transform, and instance offset
    let scaled_a = scale_mat * edge.point_a;
    let scaled_b = scale_mat * edge.point_b;
    let deformed_a = deformation * scaled_a;
    let deformed_b = deformation * scaled_b;
    let model_a = model_mat * vec3(deformed_a, 1.0);
    let model_b = model_mat * vec3(deformed_b, 1.0);
    let world_a = vec3(instance.inst_position, 0.0) + model_a;
    let world_b = vec3(instance.inst_position, 0.0) + model_b;

    // Transform through view and projection
    let view_proj = proj_mat * view_mat;

    // Use instance width if >= 0, otherwise use default
    var line_width = instance.inst_lines_width;
    if line_width < 0.0 {
        line_width = model.default_width;
    }

    // Use instance color if alpha > 0, otherwise use default
    var color = instance.inst_lines_color;
    if color.a == 0.0 {
        color = model.default_color;
    }

    let resolution = vec2(view.viewport.z, view.viewport.w);

    var pt: vec2<f32>;
    if model.use_perspective != 0u {
        // Perspective mode: add offset in world space before projection
        // Calculate perpendicular direction in world space
        let world_dir = (world_b - world_a).xy;
        let world_len = length(world_dir);
        var world_perp: vec2<f32>;
        if world_len > 0.0001 {
            let world_tangent = world_dir / world_len;
            world_perp = vec2(-world_tangent.y, world_tangent.x);
        } else {
            world_perp = vec2(0.0, 1.0);
        }

        // Add world-space offset (position.y is the perpendicular offset -0.5 to 0.5)
        let world_offset = vec3(world_perp * position.y * line_width, 0.0);
        let world_pos = mix(world_a, world_b, position.z) + world_offset;
        let clip = view_proj * world_pos;
        pt = resolution * (0.5 * clip.xy + 0.5);
    } else {
        // Non-perspective mode: add offset in screen space (constant pixel width)
        let clip0 = view_proj * world_a;
        let clip1 = view_proj * world_b;
        let screen0 = resolution * (0.5 * clip0.xy + 0.5);
        let screen1 = resolution * (0.5 * clip1.xy + 0.5);

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

        // Calculate offset from line center in screen space
        let pt_offset = line_width * (position.x * x_basis + position.y * y_basis);
        let pt0 = screen0 + pt_offset;
        let pt1 = screen1 + pt_offset;
        pt = mix(pt0, pt1, position.z);
    }

    var out: VertexOutput;
    out.clip_position = vec4((2.0 * pt) / resolution - 1.0, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
