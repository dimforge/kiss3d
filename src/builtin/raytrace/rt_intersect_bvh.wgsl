// Compute-backend intersection: a TWO-LEVEL BVH traversed in a compute shader.
//
//   - Top level (TLAS): a BVH over instances (binding 4). Each leaf references a
//     run of instances; each instance carries a world->object transform and a
//     reference to a shared bottom-level mesh.
//   - Bottom level (BLAS): one BVH per unique mesh (binding 10), stored in mesh
//     LOCAL space and shared by every instance of that mesh.
//
// A world-space ray is transformed into each instance's local space (the
// direction is NOT renormalized, so the parametric `t` is identical in both
// spaces and can be compared directly across instances). Hits are reported back
// in WORLD space, matching the contract the kernel expects:
//     trace_closest(origin, dir, tmax) -> Hit
//     trace_any(origin, dir, tmax)     -> bool

// Shared scene data + types from the preamble module (unused items strip away).
import package::rt_preamble::{
    RtVertex, RtTriangle, RtMaterial, RtLight, FrameUniforms, Hit, RtEmitter,
    BSDF_OPAQUE, BSDF_GLASS, BSDF_METAL, BSDF_EMISSIVE, PI, EPS, T_MAX,
    frame, pixels, vertices, triangles, materials, lights, emitters,
    tex_array, tex_sampler, env_tex, env_sampler
};

struct BvhNode {
    aabb_min: vec3<f32>,
    left_first: u32,
    aabb_max: vec3<f32>,
    count: u32,
};

struct RtInstance {
    world_to_object: mat4x4<f32>,
    object_to_world: mat4x4<f32>,
    mesh_id: u32,      // unused on the GPU (CPU bookkeeping only)
    material_id: u32,
    node_offset: u32,  // base of this mesh's BLAS nodes in the merged `bvh_nodes`
    tri_offset: u32,   // base of this mesh's triangles in the reordered triangles
};

// Merged two-level BVH: TLAS nodes first, then every mesh's BLAS nodes. Top-level
// traversal starts at node 0; bottom-level uses each instance's `node_offset`
// (which already includes the TLAS-node count).
@group(1) @binding(4) var<storage, read> bvh_nodes: array<BvhNode>;
@group(1) @binding(12) var<storage, read> instances: array<RtInstance>;

// Möller–Trumbore. Returns vec4(t, u, v, hit) where hit > 0.5 means a valid hit.
fn intersect_tri(ro: vec3<f32>, rd: vec3<f32>, p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> vec4<f32> {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let pv = cross(rd, e2);
    let det = dot(e1, pv);
    if (abs(det) < 1e-12) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let inv_det = 1.0 / det;
    let tv = ro - p0;
    let u = dot(tv, pv) * inv_det;
    if (u < 0.0 || u > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let qv = cross(tv, e1);
    let v = dot(rd, qv) * inv_det;
    if (v < 0.0 || u + v > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let t = dot(e2, qv) * inv_det;
    return vec4<f32>(t, u, v, 1.0);
}

// Slab test. Returns the (clamped) entry distance if the ray intersects the box
// within `(0, tmax)`, or -1.0 if it misses. Entry distance clamped to 0 so a ray
// originating inside the box still reports a hit (see the long note in git
// history: secondary rays start on surfaces inside the scene bounds).
fn intersect_aabb(ro: vec3<f32>, inv_rd: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, tmax: f32) -> f32 {
    let t0 = (lo - ro) * inv_rd;
    let t1 = (hi - ro) * inv_rd;
    let tsmall = min(t0, t1);
    let tbig = max(t0, t1);
    let tnear = max(max(tsmall.x, tsmall.y), tsmall.z);
    let tfar = min(min(tbig.x, tbig.y), tbig.z);
    if (tnear <= tfar && tfar > 0.0 && tnear < tmax) {
        return max(tnear, 0.0);
    }
    return -1.0;
}

// Upper-left 3x3 of a mat4x4 (column-major).
fn mat3_of(m: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
}

// Traverses one instance's bottom-level BVH in mesh-local space and updates `hit`
// (in world space) if it finds a closer intersection. `lo`/`ld` are the ray in
// the instance's object space (ld un-normalized, so `t` matches world space).
fn intersect_instance(inst: RtInstance, ro: vec3<f32>, rd: vec3<f32>, hit: ptr<function, Hit>) {
    let lo = (inst.world_to_object * vec4<f32>(ro, 1.0)).xyz;
    let ld = (inst.world_to_object * vec4<f32>(rd, 0.0)).xyz;
    let inv_ld = 1.0 / ld;
    // Normal matrix = inverse-transpose of object->world = transpose(mat3(world_to_object)).
    let nmat = transpose(mat3_of(inst.world_to_object));

    var stack: array<u32, 32>;
    var sp: i32 = 0;
    stack[0] = 0u;
    sp = 1;

    loop {
        if (sp <= 0) { break; }
        sp = sp - 1;
        let ni = stack[sp];
        let node = bvh_nodes[inst.node_offset + ni];

        if (intersect_aabb(lo, inv_ld, node.aabb_min, node.aabb_max, (*hit).t) < 0.0) {
            continue;
        }

        if (node.count > 0u) {
            for (var i = 0u; i < node.count; i = i + 1u) {
                let ti = inst.tri_offset + node.left_first + i;
                let tri = triangles[ti];
                let p0 = vertices[tri.v0].position;
                let p1 = vertices[tri.v1].position;
                let p2 = vertices[tri.v2].position;
                let r = intersect_tri(lo, ld, p0, p1, p2);
                if (r.w > 0.5 && r.x > EPS && r.x < (*hit).t) {
                    let w = 1.0 - r.y - r.z;
                    let local_n = vertices[tri.v0].normal * w
                                + vertices[tri.v1].normal * r.y
                                + vertices[tri.v2].normal * r.z;
                    // World-space triangle for the geometric normal + area.
                    let pw0 = (inst.object_to_world * vec4<f32>(p0, 1.0)).xyz;
                    let pw1 = (inst.object_to_world * vec4<f32>(p1, 1.0)).xyz;
                    let pw2 = (inst.object_to_world * vec4<f32>(p2, 1.0)).xyz;
                    let ng = cross(pw1 - pw0, pw2 - pw0);

                    (*hit).valid = true;
                    (*hit).t = r.x;
                    (*hit).normal = normalize(nmat * local_n);
                    (*hit).geom_normal = normalize(ng);
                    (*hit).material_id = inst.material_id;
                    (*hit).tri_area = 0.5 * length(ng);
                    let uv0 = vec2<f32>(vertices[tri.v0].u, vertices[tri.v0].v);
                    let uv1 = vec2<f32>(vertices[tri.v1].u, vertices[tri.v1].v);
                    let uv2 = vec2<f32>(vertices[tri.v2].u, vertices[tri.v2].v);
                    (*hit).uv = uv0 * w + uv1 * r.y + uv2 * r.z;
                }
            }
        } else {
            if (sp < 31) { stack[sp] = ni + 1u; sp = sp + 1; }
            if (sp < 31) { stack[sp] = node.left_first; sp = sp + 1; }
        }
    }
}

// Any-hit test of one instance's bottom-level BVH (mesh-local space).
fn instance_any_hit(inst: RtInstance, ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> bool {
    let lo = (inst.world_to_object * vec4<f32>(ro, 1.0)).xyz;
    let ld = (inst.world_to_object * vec4<f32>(rd, 0.0)).xyz;
    let inv_ld = 1.0 / ld;

    var stack: array<u32, 32>;
    var sp: i32 = 0;
    stack[0] = 0u;
    sp = 1;

    loop {
        if (sp <= 0) { break; }
        sp = sp - 1;
        let ni = stack[sp];
        let node = bvh_nodes[inst.node_offset + ni];

        if (intersect_aabb(lo, inv_ld, node.aabb_min, node.aabb_max, tmax) < 0.0) {
            continue;
        }

        if (node.count > 0u) {
            for (var i = 0u; i < node.count; i = i + 1u) {
                let tri = triangles[inst.tri_offset + node.left_first + i];
                let p0 = vertices[tri.v0].position;
                let p1 = vertices[tri.v1].position;
                let p2 = vertices[tri.v2].position;
                let r = intersect_tri(lo, ld, p0, p1, p2);
                if (r.w > 0.5 && r.x > EPS && r.x < tmax) {
                    return true;
                }
            }
        } else {
            if (sp < 31) { stack[sp] = ni + 1u; sp = sp + 1; }
            if (sp < 31) { stack[sp] = node.left_first; sp = sp + 1; }
        }
    }
    return false;
}

fn trace_closest(ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> Hit {
    var hit: Hit;
    hit.valid = false;
    hit.t = tmax;
    hit.normal = vec3<f32>(0.0, 1.0, 0.0);
    hit.geom_normal = vec3<f32>(0.0, 1.0, 0.0);
    hit.material_id = 0u;
    hit.tri_area = 0.0;
    hit.uv = vec2<f32>(0.0);

    if (frame.num_triangles == 0u) {
        return hit;
    }

    let inv_rd = 1.0 / rd;
    var stack: array<u32, 32>;
    var sp: i32 = 0;
    stack[0] = 0u;
    sp = 1;

    loop {
        if (sp <= 0) { break; }
        sp = sp - 1;
        let ni = stack[sp];
        let node = bvh_nodes[ni];

        if (intersect_aabb(ro, inv_rd, node.aabb_min, node.aabb_max, hit.t) < 0.0) {
            continue;
        }

        if (node.count > 0u) {
            for (var i = 0u; i < node.count; i = i + 1u) {
                intersect_instance(instances[node.left_first + i], ro, rd, &hit);
            }
        } else {
            if (sp < 31) { stack[sp] = ni + 1u; sp = sp + 1; }
            if (sp < 31) { stack[sp] = node.left_first; sp = sp + 1; }
        }
    }

    return hit;
}

fn trace_any(ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> bool {
    if (frame.num_triangles == 0u) {
        return false;
    }

    let inv_rd = 1.0 / rd;
    var stack: array<u32, 32>;
    var sp: i32 = 0;
    stack[0] = 0u;
    sp = 1;

    loop {
        if (sp <= 0) { break; }
        sp = sp - 1;
        let ni = stack[sp];
        let node = bvh_nodes[ni];

        if (intersect_aabb(ro, inv_rd, node.aabb_min, node.aabb_max, tmax) < 0.0) {
            continue;
        }

        if (node.count > 0u) {
            for (var i = 0u; i < node.count; i = i + 1u) {
                if (instance_any_hit(instances[node.left_first + i], ro, rd, tmax)) {
                    return true;
                }
            }
        } else {
            if (sp < 31) { stack[sp] = ni + 1u; sp = sp + 1; }
            if (sp < 31) { stack[sp] = node.left_first; sp = sp + 1; }
        }
    }

    return false;
}
