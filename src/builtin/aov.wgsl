// Auxiliary-output (AOV) shaders for kiss3d's rasterizer.
//
// A single WGSL module with three fragment entry points, one per auxiliary
// render output. They share the vertex stage and uniform layout so the host
// can drive all three passes from the same scene-graph traversal.
//
// All passes render into non-multisampled targets so that read-back is exact.

// Bind group 0: per-frame uniforms (view, projection, mode flags).
struct FrameUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    // x = 1.0 to emit camera-space normals, 0.0 for world-space normals.
    // Remaining components are reserved/padding.
    flags: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

// Bind group 1: per-object uniforms (transform, scale, segmentation id).
struct ObjectUniforms {
    transform: mat4x4<f32>,
    scale: mat3x3<f32>,
    // x = segmentation id (bit-cast from u32). y/z/w reserved/padding.
    extra: vec4<u32>,
}

@group(1) @binding(0)
var<uniform> object: ObjectUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // World-space normal (object transform applied, ignoring non-uniform scale).
    @location(0) ws_normal: vec3<f32>,
    // Eye-space (camera-space) position; -z is the positive linear depth.
    @location(1) eye_pos: vec3<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let scaled_pos = object.scale * vertex.position;
    let world_pos = object.transform * vec4<f32>(scaled_pos, 1.0);
    let eye_pos = frame.view * world_pos;

    out.clip_position = frame.proj * eye_pos;
    out.eye_pos = eye_pos.xyz;
    // Transform the normal by the object's rotation (the transform's upper-left
    // 3x3). Uniform meshes use this directly; this matches NormalsMaterial which
    // outputs object-local normals, but here we want world space for AOVs.
    let rot = mat3x3<f32>(
        object.transform[0].xyz,
        object.transform[1].xyz,
        object.transform[2].xyz,
    );
    out.ws_normal = rot * vertex.normal;

    return out;
}

// Linear (metric, eye-space) depth: positive distance in front of the camera.
@fragment
fn fs_depth(in: VertexOutput) -> @location(0) f32 {
    return -in.eye_pos.z;
}

// Surface normals, encoded from [-1, 1] into [0, 1]. World-space by default,
// camera-space when frame.flags.x is set.
@fragment
fn fs_normals(in: VertexOutput) -> @location(0) vec4<f32> {
    var n = normalize(in.ws_normal);
    if (frame.flags.x > 0.5) {
        // Rotate the world normal into camera space.
        let view_rot = mat3x3<f32>(
            frame.view[0].xyz,
            frame.view[1].xyz,
            frame.view[2].xyz,
        );
        n = normalize(view_rot * n);
    }
    let color = (n + vec3<f32>(1.0)) / 2.0;
    return vec4<f32>(color, 1.0);
}

// Segmentation: raw integer object id into an R32Uint target.
@fragment
fn fs_segmentation(in: VertexOutput) -> @location(0) u32 {
    return object.extra.x;
}
