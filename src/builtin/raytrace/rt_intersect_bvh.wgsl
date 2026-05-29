// Compute-backend intersection: traverse a CPU-built BVH stored in a storage
// buffer. Exposes the backend-independent contract used by the kernel:
//     trace_closest(origin, dir, tmax) -> Hit
//     trace_any(origin, dir, tmax)     -> bool

struct BvhNode {
    aabb_min: vec3<f32>,
    left_first: u32,
    aabb_max: vec3<f32>,
    count: u32,
};

@group(1) @binding(4) var<storage, read> bvh_nodes: array<BvhNode>;

// Möller–Trumbore. Returns vec4(t, u, v, hit) where hit > 0.5 means a valid hit.
fn intersect_tri(ro: vec3<f32>, rd: vec3<f32>, p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> vec4<f32> {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let pv = cross(rd, e2);
    let det = dot(e1, pv);
    if (abs(det) < 1e-9) {
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
// within `(0, tmax)`, or -1.0 if it misses.
//
// The entry distance is clamped to 0 so that a ray whose origin is *inside* the
// box (true for every shadow ray and bounce ray, which start on a surface inside
// the scene bounds) still reports a hit — otherwise `tnear` would be negative and
// the caller, which treats a negative result as a miss, would skip the node and
// the whole subtree, losing all shadows and indirect lighting.
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

fn trace_closest(ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> Hit {
    var hit: Hit;
    hit.valid = false;
    hit.t = tmax;
    hit.normal = vec3<f32>(0.0, 1.0, 0.0);
    hit.material_id = 0u;

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
            // Leaf: test each triangle.
            for (var i = 0u; i < node.count; i = i + 1u) {
                let tri = triangles[node.left_first + i];
                let p0 = vertices[tri.v0].position;
                let p1 = vertices[tri.v1].position;
                let p2 = vertices[tri.v2].position;
                let r = intersect_tri(ro, rd, p0, p1, p2);
                if (r.w > 0.5 && r.x > EPS && r.x < hit.t) {
                    hit.valid = true;
                    hit.t = r.x;
                    let w = 1.0 - r.y - r.z;
                    let n = vertices[tri.v0].normal * w
                          + vertices[tri.v1].normal * r.y
                          + vertices[tri.v2].normal * r.z;
                    hit.normal = normalize(n);
                    hit.material_id = tri.material_id;
                }
            }
        } else {
            // Interior: left child is ni+1, right child is node.left_first.
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
                let tri = triangles[node.left_first + i];
                let p0 = vertices[tri.v0].position;
                let p1 = vertices[tri.v1].position;
                let p2 = vertices[tri.v2].position;
                let r = intersect_tri(ro, rd, p0, p1, p2);
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
