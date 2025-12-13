// Points shader for thick point rendering of mesh vertices
// Each vertex expands to a 6-vertex quad (2 triangles) for configurable point size.
//
// Uses storage buffer for vertex positions and vertex buffers for instances.
// Draw call: draw(0..(6 * num_vertices), 0..num_instances)

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
    num_vertices: u32,       // Number of vertices in the vertex buffer
    default_color: vec4<f32>, // Default point color (used when instance alpha == 0)
    default_size: f32,       // Default point size (used when instance size < 0)
    use_perspective: u32,    // Whether to scale size with distance (1 = yes, 0 = no)
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> view: ViewUniforms;

@group(1) @binding(0)
var<uniform> model: ModelUniforms;

// Vertex storage buffer (bind group 1, binding 1)
struct Vertex {
    position: vec3<f32>,
    _pad: f32,
}

@group(1) @binding(1)
var<storage, read> vertices: array<Vertex>;

// Instance input from vertex buffers (reuses InstancesBuffer layout)
struct InstanceInput {
    @location(0) inst_position: vec3<f32>,      // positions buffer
    @location(1) inst_color: vec4<f32>,          // colors buffer (mesh color, not used for points)
    @location(2) inst_def_0: vec3<f32>,          // deformations buffer (col 0)
    @location(3) inst_def_1: vec3<f32>,          // deformations buffer (col 1)
    @location(4) inst_def_2: vec3<f32>,          // deformations buffer (col 2)
    @location(5) inst_points_color: vec4<f32>,   // points_colors buffer
    @location(6) inst_points_size: f32,          // points_sizes buffer
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(
    instance: InstanceInput,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    // Compute which mesh vertex from vertex_index
    // Draw call: draw(0..(6 * num_vertices), 0..num_instances)
    // Each 6 vertices form one point quad
    let point_index = vertex_index / 6u;
    let vertex = vertices[point_index];

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

    // Build deformation matrix from instance data
    let deformation = mat3x3<f32>(
        instance.inst_def_0,
        instance.inst_def_1,
        instance.inst_def_2
    );

    // Apply deformation, scale, model transform, and instance offset
    let deformed_pos = deformation * vertex.position;
    let scaled_pos = deformed_pos * model.scale;
    let model_pos = model.transform * vec4(scaled_pos, 1.0);
    let world_pos = model_pos + vec4(instance.inst_position, 0.0);

    // Transform to clip space
    let view_proj = view.proj * view.view;
    let clip = view_proj * world_pos;

    // Skip points behind camera
    if clip.w <= 0.0 {
        var out: VertexOutput;
        out.clip_position = vec4(0.0, 0.0, -1.0, 1.0); // Behind near plane
        out.color = vec4(0.0);
        return out;
    }

    // Calculate screen-space position
    let resolution = vec2(view.viewport.z, view.viewport.w);
    let screen_center = resolution * (0.5 * clip.xy / clip.w + 0.5);

    // Use instance size if >= 0, otherwise use default
    var point_size = instance.inst_points_size;
    if point_size < 0.0 {
        point_size = model.default_size;
    }

    // Apply perspective scaling if enabled (scale size by 1/w to shrink with distance)
    if model.use_perspective != 0u {
        point_size = point_size / clip.w;
    }

    // Use instance color if alpha > 0, otherwise use default
    var color = instance.inst_points_color;
    if color.a == 0.0 {
        color = model.default_color;
    }

    // Calculate offset from point center (offset is -0.5 to 0.5, multiply by size)
    let pt = screen_center + offset * point_size;

    var out: VertexOutput;
    out.clip_position = vec4(clip.w * ((2.0 * pt) / resolution - 1.0), clip.z, clip.w);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
