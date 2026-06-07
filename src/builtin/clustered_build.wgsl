// Clustered forward+ : cluster AABB build pass.
//
// One invocation per cluster. Computes the cluster's view-space AABB from the
// inverse projection (screen tile corners) and the cluster's exponential depth
// slice. Only re-run when the projection or viewport changes.

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

@group(0) @binding(0) var<uniform> u: ClusterUniforms;
@group(0) @binding(1) var<storage, read_write> aabbs: array<ClusterAABB>;

// Clip-space point (z arbitrary on the same pixel ray) → view space.
fn clip_to_view(clip: vec4<f32>) -> vec3<f32> {
    let v = u.inv_proj * clip;
    return v.xyz / v.w;
}

// Pixel coordinate (top-left origin) → a view-space point on that pixel's eye ray.
fn screen_to_view(px: vec2<f32>) -> vec3<f32> {
    let tex = px / u.screen.xy;
    // wgpu clip space: xy in [-1, 1] with y up; pixel y is top-down, so flip y.
    let ndc = vec2<f32>(tex.x * 2.0 - 1.0, 1.0 - tex.y * 2.0);
    return clip_to_view(vec4<f32>(ndc, 0.0, 1.0));
}

// Intersect the ray (origin -> p) with the constant view-space plane z = zp.
fn z_plane(p: vec3<f32>, zp: f32) -> vec3<f32> {
    let t = zp / p.z;
    return p * t;
}

@compute @workgroup_size(4, 4, 4)
fn build_aabbs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.grid.x || gid.y >= u.grid.y || gid.z >= u.grid.z {
        return;
    }
    let idx = gid.x + gid.y * u.grid.x + gid.z * u.grid.x * u.grid.y;

    let tile = u.screen.zw;
    let min_px = vec2<f32>(f32(gid.x), f32(gid.y)) * tile;
    let max_px = vec2<f32>(f32(gid.x + 1u), f32(gid.y + 1u)) * tile;

    let min_vs = screen_to_view(min_px);
    let max_vs = screen_to_view(max_px);

    let near = u.depth.x;
    let far = u.depth.y;
    let ratio = far / near;
    // View space looks down -Z, so cluster z planes are negative.
    let z_near = -near * pow(ratio, f32(gid.z) / f32(u.grid.z));
    let z_far = -near * pow(ratio, f32(gid.z + 1u) / f32(u.grid.z));

    let p0 = z_plane(min_vs, z_near);
    let p1 = z_plane(min_vs, z_far);
    let p2 = z_plane(max_vs, z_near);
    let p3 = z_plane(max_vs, z_far);

    let lo = min(min(p0, p1), min(p2, p3));
    let hi = max(max(p0, p1), max(p2, p3));

    aabbs[idx].min_pt = vec4<f32>(lo, 0.0);
    aabbs[idx].max_pt = vec4<f32>(hi, 0.0);
}
