// Planar points shader for thick point rendering of 2D mesh vertices
// Adapted from 3D wireframe_points.wgsl for 2D planar rendering
//
// Uses storage buffer for vertex positions and vertex buffers for instances.
// Draw call: draw(0..(6 * num_vertices), 0..num_instances)

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
    num_vertices: u32,
    default_size: f32,
    use_perspective: u32,
    _padding1: f32,
    default_color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> view: ViewUniforms;

@group(1) @binding(0)
var<uniform> model: ModelUniforms;

// Vertex storage buffer (bind group 1, binding 1)
struct Vertex {
    position: vec2<f32>,
}

@group(1) @binding(1)
var<storage, read> vertices: array<Vertex>;

// Instance input from vertex buffers (reuses PlanarInstancesBuffer layout)
struct InstanceInput {
    @location(0) inst_position: vec2<f32>,      // positions buffer
    @location(1) inst_color: vec4<f32>,          // colors buffer (mesh color, not used for points)
    @location(2) inst_def_0: vec2<f32>,          // deformations buffer (col 0)
    @location(3) inst_def_1: vec2<f32>,          // deformations buffer (col 1)
    @location(4) inst_points_color: vec4<f32>,   // points_colors buffer
    @location(5) inst_points_size: f32,          // points_sizes buffer
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

    // Apply scale, deformation, model transform, and instance offset
    let scaled_pos = scale_mat * vertex.position;
    let deformed_pos = deformation * scaled_pos;
    let model_pos = model_mat * vec3(deformed_pos, 1.0);
    let world_pos = vec3(instance.inst_position, 0.0) + model_pos;

    // Transform through view and projection
    let view_proj = proj_mat * view_mat;

    // Use instance size if >= 0, otherwise use default
    var point_size = instance.inst_points_size;
    if point_size < 0.0 {
        point_size = model.default_size;
    }

    // Use instance color if alpha > 0, otherwise use default
    var color = instance.inst_points_color;
    if color.a == 0.0 {
        color = model.default_color;
    }

    let resolution = vec2(view.viewport.z, view.viewport.w);

    var pt: vec2<f32>;
    if model.use_perspective != 0u {
        // Perspective mode: add offset in world space before projection
        // The offset scales naturally with the camera
        let world_offset = vec3(offset * point_size, 0.0);
        let world_pos_offset = world_pos + world_offset;
        let clip_offset = view_proj * world_pos_offset;
        pt = resolution * (0.5 * clip_offset.xy + 0.5);
    } else {
        // Non-perspective mode: add offset in screen space (constant pixel size)
        let clip = view_proj * world_pos;
        let screen_center = resolution * (0.5 * clip.xy + 0.5);
        pt = screen_center + offset * point_size;
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
