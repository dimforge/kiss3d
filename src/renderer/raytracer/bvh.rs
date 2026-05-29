//! A simple CPU bounding-volume hierarchy over the scene triangles.
//!
//! The BVH is built with a median split along the largest centroid axis and
//! flattened into a contiguous array of [`BvhNode`]s that the WGSL kernel can
//! traverse with an explicit stack. The triangle list is reordered so that the
//! triangles referenced by each leaf are contiguous.

use bytemuck::{Pod, Zeroable};
use glamx::Vec3;

use super::scene_data::{RtTriangle, RtVertex};

/// A flattened BVH node, 32 bytes, matching the WGSL `BvhNode` layout.
///
/// `count == 0` marks an interior node. Its left child is always the next node
/// (`self_index + 1`, depth-first layout) and its right child is at `left_first`.
/// `count > 0` marks a leaf whose triangles occupy
/// `[left_first, left_first + count)` in the reordered triangle array.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct BvhNode {
    /// Minimum corner of the node's AABB.
    pub aabb_min: [f32; 3],
    /// Interior: left child index. Leaf: first triangle index.
    pub left_first: u32,
    /// Maximum corner of the node's AABB.
    pub aabb_max: [f32; 3],
    /// Number of triangles (0 = interior node).
    pub count: u32,
}

const LEAF_SIZE: usize = 4;

#[inline]
fn tri_centroid(v: &[RtVertex], t: &RtTriangle) -> Vec3 {
    let a = Vec3::from_array(v[t.v0 as usize].position);
    let b = Vec3::from_array(v[t.v1 as usize].position);
    let c = Vec3::from_array(v[t.v2 as usize].position);
    (a + b + c) / 3.0
}

#[inline]
fn tri_bounds(v: &[RtVertex], t: &RtTriangle) -> (Vec3, Vec3) {
    let a = Vec3::from_array(v[t.v0 as usize].position);
    let b = Vec3::from_array(v[t.v1 as usize].position);
    let c = Vec3::from_array(v[t.v2 as usize].position);
    (a.min(b).min(c), a.max(b).max(c))
}

/// Builds a BVH over `triangles` and returns `(nodes, ordered_triangles)`.
///
/// `ordered_triangles` is a permutation of the input triangles such that each
/// leaf's triangles are contiguous; upload it (not the original) alongside the
/// node array. An empty input yields a single empty leaf so the GPU buffers
/// remain bindable.
pub fn build(vertices: &[RtVertex], triangles: &[RtTriangle]) -> (Vec<BvhNode>, Vec<RtTriangle>) {
    if triangles.is_empty() {
        return (
            vec![BvhNode {
                aabb_min: [0.0; 3],
                left_first: 0,
                aabb_max: [0.0; 3],
                count: 0,
            }],
            Vec::new(),
        );
    }

    let centroids: Vec<Vec3> = triangles.iter().map(|t| tri_centroid(vertices, t)).collect();
    let bounds: Vec<(Vec3, Vec3)> = triangles.iter().map(|t| tri_bounds(vertices, t)).collect();

    let mut indices: Vec<u32> = (0..triangles.len() as u32).collect();
    let mut nodes: Vec<BvhNode> = Vec::with_capacity(triangles.len() * 2);

    build_recursive(
        &mut nodes,
        &mut indices,
        &centroids,
        &bounds,
        0,
        triangles.len(),
    );

    let ordered: Vec<RtTriangle> = indices.iter().map(|&i| triangles[i as usize]).collect();
    (nodes, ordered)
}

/// Number of bins used by the SAH sweep.
const NUM_BINS: usize = 12;

#[inline]
fn surface_area(min: Vec3, max: Vec3) -> f32 {
    let d = (max - min).max(Vec3::ZERO);
    2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
}

fn build_recursive(
    nodes: &mut Vec<BvhNode>,
    indices: &mut [u32],
    centroids: &[Vec3],
    bounds: &[(Vec3, Vec3)],
    start: usize,
    end: usize,
) -> u32 {
    let node_index = nodes.len() as u32;
    nodes.push(BvhNode::default());

    // AABB over all triangles in this range (and the centroid bounds).
    let mut bmin = Vec3::splat(f32::INFINITY);
    let mut bmax = Vec3::splat(f32::NEG_INFINITY);
    let mut cmin = Vec3::splat(f32::INFINITY);
    let mut cmax = Vec3::splat(f32::NEG_INFINITY);
    for &i in &indices[start..end] {
        let (lo, hi) = bounds[i as usize];
        bmin = bmin.min(lo);
        bmax = bmax.max(hi);
        let c = centroids[i as usize];
        cmin = cmin.min(c);
        cmax = cmax.max(c);
    }

    let count = end - start;
    let make_leaf = |nodes: &mut Vec<BvhNode>| {
        nodes[node_index as usize] = BvhNode {
            aabb_min: bmin.to_array(),
            left_first: start as u32,
            aabb_max: bmax.to_array(),
            count: count as u32,
        };
        node_index
    };

    let extent = cmax - cmin;
    // Degenerate (all centroids coincide) or tiny: leaf.
    if count <= LEAF_SIZE || extent.max_element() <= 0.0 {
        return make_leaf(nodes);
    }

    // Binned SAH: for each axis, bin triangles by centroid and sweep the
    // candidate split planes, scoring `area(L)*count(L) + area(R)*count(R)`.
    let mut best_cost = f32::INFINITY;
    let mut best_axis = usize::MAX;
    let mut best_split = 0usize;
    for axis in 0..3 {
        let lo = cmin[axis];
        let ext = extent[axis];
        if ext <= 0.0 {
            continue;
        }
        let scale = NUM_BINS as f32 / ext;

        let mut bin_count = [0u32; NUM_BINS];
        let mut bin_min = [Vec3::splat(f32::INFINITY); NUM_BINS];
        let mut bin_max = [Vec3::splat(f32::NEG_INFINITY); NUM_BINS];
        for &i in &indices[start..end] {
            let b = (((centroids[i as usize][axis] - lo) * scale) as usize).min(NUM_BINS - 1);
            bin_count[b] += 1;
            let (tlo, thi) = bounds[i as usize];
            bin_min[b] = bin_min[b].min(tlo);
            bin_max[b] = bin_max[b].max(thi);
        }

        // Prefix (left) and suffix (right) sweeps over the NUM_BINS-1 planes.
        let mut left_area = [0.0f32; NUM_BINS - 1];
        let mut left_count = [0u32; NUM_BINS - 1];
        let mut right_area = [0.0f32; NUM_BINS - 1];
        let mut right_count = [0u32; NUM_BINS - 1];
        {
            let mut lmin = Vec3::splat(f32::INFINITY);
            let mut lmax = Vec3::splat(f32::NEG_INFINITY);
            let mut lcount = 0u32;
            for s in 0..NUM_BINS - 1 {
                lcount += bin_count[s];
                lmin = lmin.min(bin_min[s]);
                lmax = lmax.max(bin_max[s]);
                left_count[s] = lcount;
                left_area[s] = if lcount > 0 { surface_area(lmin, lmax) } else { 0.0 };
            }
            let mut rmin = Vec3::splat(f32::INFINITY);
            let mut rmax = Vec3::splat(f32::NEG_INFINITY);
            let mut rcount = 0u32;
            for s in (0..NUM_BINS - 1).rev() {
                rcount += bin_count[s + 1];
                rmin = rmin.min(bin_min[s + 1]);
                rmax = rmax.max(bin_max[s + 1]);
                right_count[s] = rcount;
                right_area[s] = if rcount > 0 { surface_area(rmin, rmax) } else { 0.0 };
            }
        }

        for s in 0..NUM_BINS - 1 {
            if left_count[s] == 0 || right_count[s] == 0 {
                continue;
            }
            let cost =
                left_area[s] * left_count[s] as f32 + right_area[s] * right_count[s] as f32;
            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_split = s;
            }
        }
    }

    // Leaf if no useful split, or splitting isn't worth it for a small node.
    let node_area = surface_area(bmin, bmax);
    let leaf_cost = count as f32 * node_area;
    if best_axis == usize::MAX || (count <= 8 && best_cost >= leaf_cost) {
        return make_leaf(nodes);
    }

    // Partition `indices[start..end]` by the chosen plane.
    let lo = cmin[best_axis];
    let scale = NUM_BINS as f32 / extent[best_axis];
    let split_bin = best_split;
    let mut mid;
    {
        let mut i = start;
        let mut j = end;
        while i < j {
            let b = (((centroids[indices[i] as usize][best_axis] - lo) * scale) as usize)
                .min(NUM_BINS - 1);
            if b <= split_bin {
                i += 1;
            } else {
                j -= 1;
                indices.swap(i, j);
            }
        }
        mid = i;
    }

    // Guard against a degenerate partition (everything on one side).
    if mid == start || mid == end {
        mid = start + count / 2;
        indices[start..end].select_nth_unstable_by(count / 2, |&a, &b| {
            centroids[a as usize][best_axis]
                .partial_cmp(&centroids[b as usize][best_axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Depth-first: the left child lands at `node_index + 1`; the right child
    // follows the whole left subtree, so its index is stored explicitly.
    let _left = build_recursive(nodes, indices, centroids, bounds, start, mid);
    let right = build_recursive(nodes, indices, centroids, bounds, mid, end);

    nodes[node_index as usize] = BvhNode {
        aabb_min: bmin.to_array(),
        left_first: right,
        aabb_max: bmax.to_array(),
        count: 0,
    };
    node_index
}
