// Hardware-backend intersection: inline ray queries against a TLAS. Exposes the
// same contract as the compute backend:
//     trace_closest(origin, dir, tmax) -> Hit
//     trace_any(origin, dir, tmax)     -> bool
//
// The module is assembled with `enable wgpu_ray_query;` prepended (see
// pipeline.rs). Geometry is a single BLAS holding the whole world-space scene in
// triangle order, so `primitive_index` indexes the `triangles` table directly.

@group(1) @binding(4) var tlas: acceleration_structure;

const RAY_FLAG_NONE: u32 = 0u;
const RAY_FLAG_TERMINATE_ON_FIRST_HIT: u32 = 0x04u;
const RAY_QUERY_INTERSECTION_NONE: u32 = 0u;

fn trace_closest(ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> Hit {
    var hit: Hit;
    hit.valid = false;
    hit.t = tmax;
    hit.normal = vec3<f32>(0.0, 1.0, 0.0);
    hit.material_id = 0u;

    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, RayDesc(RAY_FLAG_NONE, 0xFFu, EPS, tmax, ro, rd));
    while (rayQueryProceed(&rq)) {}
    let isect = rayQueryGetCommittedIntersection(&rq);

    if (isect.kind != RAY_QUERY_INTERSECTION_NONE) {
        let tri = triangles[isect.primitive_index];
        let b = isect.barycentrics;
        let w = 1.0 - b.x - b.y;
        let n = vertices[tri.v0].normal * w
              + vertices[tri.v1].normal * b.x
              + vertices[tri.v2].normal * b.y;
        hit.valid = true;
        hit.t = isect.t;
        hit.normal = normalize(n);
        hit.material_id = tri.material_id;
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
    while (rayQueryProceed(&rq)) {}
    let isect = rayQueryGetCommittedIntersection(&rq);
    return isect.kind != RAY_QUERY_INTERSECTION_NONE;
}
