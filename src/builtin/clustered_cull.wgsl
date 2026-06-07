// Clustered forward+ : light culling pass.
//
// One invocation per cluster. Tests every clustered light's bounding sphere
// (centre = view-space position, radius = attenuation_radius) against the
// cluster's view-space AABB, reserves a contiguous slice in the global index
// list via an atomic counter, and records (offset, count) for the cluster.

struct ClusterUniforms {
    inv_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    grid: vec4<u32>,    // (grid_x, grid_y, grid_z, num_clustered_lights)
    screen: vec4<f32>,  // (width_px, height_px, tile_w_px, tile_h_px)
    depth: vec4<f32>,   // (z_near, z_far, ln(z_far/z_near), unused)
};

struct ClusterAABB {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};

struct GpuLight {
    position: vec3<f32>,
    light_type: u32,
    direction: vec3<f32>,
    intensity: f32,
    color: vec3<f32>,
    inner_cone_cos: f32,
    outer_cone_cos: f32,
    attenuation_radius: f32,
    shadow_slot: u32,
    // Light-layer bitmask (lighting channels); unused by culling, kept for layout
    // parity with GpuLight in object_material.rs / LightData in default.wgsl.
    layers: u32,
};

@group(0) @binding(0) var<uniform> u: ClusterUniforms;
@group(0) @binding(1) var<storage, read> aabbs: array<ClusterAABB>;
@group(0) @binding(2) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(3) var<storage, read_write> grid: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> index_list: array<u32>;

// Must match `MAX_LIGHTS_PER_CLUSTER` in clustered.rs.
const MAX_PER_CLUSTER: u32 = 256u;

// Squared distance from `c` to the AABB [lo, hi]; 0 if inside.
fn sphere_hits_aabb(c: vec3<f32>, r: f32, lo: vec3<f32>, hi: vec3<f32>) -> bool {
    let closest = clamp(c, lo, hi);
    let d = closest - c;
    return dot(d, d) <= r * r;
}

@compute @workgroup_size(64)
fn cull_lights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cluster = gid.x;
    let total = u.grid.x * u.grid.y * u.grid.z;
    if cluster >= total {
        return;
    }

    let lo = aabbs[cluster].min_pt.xyz;
    let hi = aabbs[cluster].max_pt.xyz;
    let n = u.grid.w;

    // Each cluster owns a fixed `MAX_PER_CLUSTER` slice of the index list (one
    // thread per cluster, so no atomics or per-thread scratch array needed). The
    // slice base is `cluster * MAX_PER_CLUSTER`; lights past the cap are dropped.
    let base = cluster * MAX_PER_CLUSTER;
    var count = 0u;
    for (var i = 0u; i < n; i = i + 1u) {
        if count >= MAX_PER_CLUSTER {
            break;
        }
        let l = lights[i];
        let pos_vs = (u.view * vec4<f32>(l.position, 1.0)).xyz;
        if sphere_hits_aabb(pos_vs, l.attenuation_radius, lo, hi) {
            index_list[base + count] = i;
            count = count + 1u;
        }
    }
    grid[cluster] = vec2<u32>(base, count);
}
