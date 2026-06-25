import package::common::unpack_mat3;
// 2D GPU skinning: each vertex is transformed by a weighted blend of up to four
// joint matrices, then by the object model transform and the camera. Joint matrices
// (bone world × inverse-bind, as 2D affine 3x3) are supplied per frame.

const MAX_JOINTS: u32 = 32u;

struct FrameUniforms {
    view_0: vec4<f32>,
    view_1: vec4<f32>,
    view_2: vec4<f32>,
    proj_0: vec4<f32>,
    proj_1: vec4<f32>,
    proj_2: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

struct ObjectUniforms {
    model_0: vec4<f32>,
    model_1: vec4<f32>,
    model_2: vec4<f32>,
    color: vec4<f32>,
    // Each joint is three vec4 (padded 3x3 columns): joints[3*j + {0,1,2}].
    joints: array<vec4<f32>, 96>,
}

@group(1) @binding(0)
var<uniform> obj: ObjectUniforms;

@group(2) @binding(0)
var t_albedo: texture_2d<f32>;
@group(2) @binding(1)
var s_albedo: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coord: vec2<f32>,
    @location(2) joints: vec4<u32>,
    @location(3) weights: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}

fn joint_mat(j: u32) -> mat3x3<f32> {
    let b = 3u * j;
    return unpack_mat3(obj.joints[b], obj.joints[b + 1u], obj.joints[b + 2u]);
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let p = vec3<f32>(vertex.position, 1.0);
    // Weighted blend of the four influencing joints.
    var skinned = vec3<f32>(0.0);
    skinned += vertex.weights.x * (joint_mat(vertex.joints.x) * p);
    skinned += vertex.weights.y * (joint_mat(vertex.joints.y) * p);
    skinned += vertex.weights.z * (joint_mat(vertex.joints.z) * p);
    skinned += vertex.weights.w * (joint_mat(vertex.joints.w) * p);

    let view = unpack_mat3(frame.view_0, frame.view_1, frame.view_2);
    let proj = unpack_mat3(frame.proj_0, frame.proj_1, frame.proj_2);
    let model = unpack_mat3(obj.model_0, obj.model_1, obj.model_2);

    let world = model * vec3<f32>(skinned.xy, 1.0);
    var projected = proj * view * world;
    projected.z = 0.0;

    out.clip_position = vec4<f32>(projected, 1.0);
    out.tex_coord = vertex.tex_coord;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_albedo, s_albedo, in.tex_coord) * obj.color;
}
