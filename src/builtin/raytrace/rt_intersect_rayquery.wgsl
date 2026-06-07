// Hardware-backend intersection: inline ray queries against a TLAS with
// instancing. Exposes the same contract as the compute backend:
//     trace_closest(origin, dir, tmax) -> Hit
//     trace_any(origin, dir, tmax)     -> bool
//
// The `enable wgpu_ray_query;` directive is emitted by the root kernel module
// (gated `@if(hardware)`). The TLAS holds one instance per scene copy; each
// instance points at a shared per-mesh BLAS and carries (via its custom index) an
// index into the `instances` buffer, where the hit shader reads the mesh +
// material. A committed `primitive_index` is the triangle within that mesh's BLAS;
// the mesh's `tri_offset` maps it to the global (mesh-local) triangle/vertex tables.

// Shared scene data + types from the preamble module (unused items strip away).
import package::rt_preamble::{
    RtVertex, RtTriangle, RtMaterial, RtLight, FrameUniforms, Hit, RtEmitter,
    BSDF_OPAQUE, BSDF_GLASS, BSDF_METAL, BSDF_EMISSIVE, PI, EPS, T_MAX,
    frame, pixels, vertices, triangles, materials, lights, emitters,
    tex_array, tex_sampler, env_tex, env_sampler
};

struct RtMeshDesc {
    node_offset: u32,
    tri_offset: u32,
    pad0: u32,
    pad1: u32,
};

struct RtInstance {
    world_to_object: mat4x4<f32>,
    object_to_world: mat4x4<f32>,
    mesh_id: u32,
    material_id: u32,
    pad0: u32,
    pad1: u32,
};

@group(1) @binding(4) var tlas: acceleration_structure;
@group(1) @binding(11) var<storage, read> meshes: array<RtMeshDesc>;
@group(1) @binding(12) var<storage, read> instances: array<RtInstance>;

const RAY_FLAG_NONE: u32 = 0u;
const RAY_FLAG_TERMINATE_ON_FIRST_HIT: u32 = 0x04u;
const RAY_QUERY_INTERSECTION_NONE: u32 = 0u;

fn trace_closest(ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> Hit {
    var hit: Hit;
    hit.valid = false;
    hit.t = tmax;
    hit.normal = vec3<f32>(0.0, 1.0, 0.0);
    hit.geom_normal = vec3<f32>(0.0, 1.0, 0.0);
    hit.material_id = 0u;
    hit.tri_area = 0.0;
    hit.uv = vec2<f32>(0.0);

    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, RayDesc(RAY_FLAG_NONE, 0xFFu, EPS, tmax, ro, rd));
    // All geometry is flagged OPAQUE, so the driver commits hits internally and this
    // loop normally runs zero or one iterations. The `guard` is a safety valve: on a
    // misbehaving (experimental) ray-query backend a never-terminating proceed would
    // otherwise hang the GPU — and macOS has no compute watchdog, so that freezes the
    // whole machine. Starting small to test whether bounding the loop is what stops
    // the hang; raise it if a correct backend ever truncates traversal.
    var guard = 0u;
    while (rayQueryProceed(&rq) && guard < 32u) { guard = guard + 1u; }
    let isect = rayQueryGetCommittedIntersection(&rq);

    if (isect.kind != RAY_QUERY_INTERSECTION_NONE) {
        let inst = instances[isect.instance_custom_data];
        let mesh = meshes[inst.mesh_id];
        let tri = triangles[mesh.tri_offset + isect.primitive_index];
        let b = isect.barycentrics;
        let w = 1.0 - b.x - b.y;

        // Mesh-local positions/normal/uv, interpolated at the hit.
        let p0 = vertices[tri.v0].position;
        let p1 = vertices[tri.v1].position;
        let p2 = vertices[tri.v2].position;
        let local_n = vertices[tri.v0].normal * w
                    + vertices[tri.v1].normal * b.x
                    + vertices[tri.v2].normal * b.y;

        // World-space triangle (via the instance transform the query provides) for
        // the geometric normal + area.
        let pw0 = isect.object_to_world * vec4<f32>(p0, 1.0);
        let pw1 = isect.object_to_world * vec4<f32>(p1, 1.0);
        let pw2 = isect.object_to_world * vec4<f32>(p2, 1.0);
        let ng = cross(pw1 - pw0, pw2 - pw0);

        // Shading normal: inverse-transpose of object→world = transpose of the
        // 3x3 part of world→object.
        let w2o = isect.world_to_object;
        let nmat = transpose(mat3x3<f32>(w2o[0], w2o[1], w2o[2]));

        hit.valid = true;
        hit.t = isect.t;
        hit.normal = normalize(nmat * local_n);
        hit.geom_normal = normalize(ng);
        hit.material_id = inst.material_id;
        hit.tri_area = 0.5 * length(ng);
        let uv0 = vec2<f32>(vertices[tri.v0].u, vertices[tri.v0].v);
        let uv1 = vec2<f32>(vertices[tri.v1].u, vertices[tri.v1].v);
        let uv2 = vec2<f32>(vertices[tri.v2].u, vertices[tri.v2].v);
        hit.uv = uv0 * w + uv1 * b.x + uv2 * b.y;
    }

    return hit;
}

fn trace_any(ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> bool {
    var rq: ray_query;
    rayQueryInitialize(
        &rq,
        tlas,
        RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0xFFu, EPS, tmax, ro, rd),
    );
    // See `trace_closest`: bounded for safety against a non-terminating proceed.
    var guard = 0u;
    while (rayQueryProceed(&rq) && guard < 32u) { guard = guard + 1u; }
    let isect = rayQueryGetCommittedIntersection(&rq);
    return isect.kind != RAY_QUERY_INTERSECTION_NONE;
}
