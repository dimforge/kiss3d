//! Utilities useful for various generations tasks.

use glamx::{Vec2, Vec3};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// A wrapper for Vec3 that allows it to be used as a HashMap key.
/// Uses bit representation of f32 values for hashing.
#[derive(Clone, Copy)]
struct HashableVec3(Vec3);

impl PartialEq for HashableVec3 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for HashableVec3 {}

impl Hash for HashableVec3 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.x.to_bits().hash(state);
        self.0.y.to_bits().hash(state);
        self.0.z.to_bits().hash(state);
    }
}

// TODO: remove that in favor of `push_xy_circle` ?
/// Pushes a discretized counterclockwise circle to a buffer.
#[inline]
pub fn push_circle(radius: f32, nsubdiv: u32, dtheta: f32, y: f32, out: &mut Vec<Vec3>) {
    let mut curr_theta = 0.0f32;

    for _ in 0..nsubdiv {
        out.push(Vec3::new(
            curr_theta.cos() * radius,
            y,
            curr_theta.sin() * radius,
        ));
        curr_theta += dtheta;
    }
}

/// Pushes a discretized counterclockwise circle to a buffer.
/// The circle is contained on the plane spanned by the `x` and `y` axis.
#[inline]
pub fn push_xy_arc(radius: f32, nsubdiv: u32, dtheta: f32, out: &mut Vec<Vec2>) {
    let mut curr_theta = 0.0f32;

    for _ in 0..nsubdiv {
        out.push(Vec2::new(
            curr_theta.cos() * radius,
            curr_theta.sin() * radius,
        ));
        curr_theta += dtheta;
    }
}

/// Creates the faces from two circles with the same discretization.
#[inline]
pub fn push_ring_indices(
    base_lower_circle: u32,
    base_upper_circle: u32,
    nsubdiv: u32,
    out: &mut Vec<[u32; 3]>,
) {
    push_open_ring_indices(base_lower_circle, base_upper_circle, nsubdiv, out);

    // adjust the last two triangles
    push_rectangle_indices(
        base_upper_circle,
        base_upper_circle + nsubdiv - 1,
        base_lower_circle,
        base_lower_circle + nsubdiv - 1,
        out,
    );
}

/// Creates the faces from two circles with the same discretization.
#[inline]
pub fn push_open_ring_indices(
    base_lower_circle: u32,
    base_upper_circle: u32,
    nsubdiv: u32,
    out: &mut Vec<[u32; 3]>,
) {
    assert!(nsubdiv > 0);

    for i in 0..nsubdiv - 1 {
        let bl_i = base_lower_circle + i;
        let bu_i = base_upper_circle + i;
        push_rectangle_indices(bu_i + 1, bu_i, bl_i + 1, bl_i, out);
    }
}

/// Creates the faces from a circle and a point that is shared by all triangle.
#[inline]
pub fn push_degenerate_top_ring_indices(
    base_circle: u32,
    point: u32,
    nsubdiv: u32,
    out: &mut Vec<[u32; 3]>,
) {
    push_degenerate_open_top_ring_indices(base_circle, point, nsubdiv, out);

    out.push([base_circle + nsubdiv - 1, point, base_circle]);
}

/// Creates the faces from a circle and a point that is shared by all triangle.
#[inline]
pub fn push_degenerate_open_top_ring_indices(
    base_circle: u32,
    point: u32,
    nsubdiv: u32,
    out: &mut Vec<[u32; 3]>,
) {
    assert!(nsubdiv > 0);

    for i in 0..nsubdiv - 1 {
        out.push([base_circle + i, point, base_circle + i + 1]);
    }
}

/// Pushes indices so that a circle is filled with triangles. Each triangle will have the
/// `base_circle` point in common.
/// Pushes `nsubdiv - 2` elements to `out`.
#[inline]
pub fn push_filled_circle_indices(base_circle: u32, nsubdiv: u32, out: &mut Vec<[u32; 3]>) {
    for i in base_circle + 1..base_circle + nsubdiv - 1 {
        out.push([base_circle, i, i + 1]);
    }
}

/// Given four corner points, pushes to two counterclockwise triangles to `out`.
///
/// # Arguments:
/// * `ul` - the up-left point.
/// * `dl` - the down-left point.
/// * `dr` - the down-left point.
/// * `ur` - the up-left point.
#[inline]
pub fn push_rectangle_indices(ul: u32, ur: u32, dl: u32, dr: u32, out: &mut Vec<[u32; 3]>) {
    out.push([ul, dl, dr]);
    out.push([dr, ur, ul]);
}

/// Reverses the clockwising of a set of faces.
#[inline]
pub fn reverse_clockwising(indices: &mut [[u32; 3]]) {
    for i in indices.iter_mut() {
        i.swap(0, 1);
    }
}

/// Duplicates the indices of each triangle on the given index buffer.
///
/// For example: [ [0, 1, 2] ] becomes: [ [[0, 0, 0], [1, 1, 1], [2, 2, 2]] ].
#[inline]
pub fn split_index_buffer(indices: &[[u32; 3]]) -> Vec<[[u32; 3]; 3]> {
    let mut resi = Vec::new();

    for vertex in indices.iter() {
        resi.push([
            [vertex[0], vertex[0], vertex[0]],
            [vertex[1], vertex[1], vertex[1]],
            [vertex[2], vertex[2], vertex[2]],
        ]);
    }

    resi
}

/// Duplicates the indices of each triangle on the given index buffer, giving the same id to each
/// identical vertex.
#[inline]
pub fn split_index_buffer_and_recover_topology(
    indices: &[[u32; 3]],
    coords: &[Vec3],
) -> (Vec<[[u32; 3]; 3]>, Vec<Vec3>) {
    let mut vtx_to_id: HashMap<HashableVec3, u32> = HashMap::default();
    let mut new_coords = Vec::with_capacity(coords.len());
    let mut out = Vec::with_capacity(indices.len());

    fn resolve_coord_id(
        coord: Vec3,
        vtx_to_id: &mut HashMap<HashableVec3, u32>,
        new_coords: &mut Vec<Vec3>,
    ) -> u32 {
        let key = HashableVec3(coord);
        let id = match vtx_to_id.entry(key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(new_coords.len() as u32),
        };

        if *id == new_coords.len() as u32 {
            new_coords.push(coord);
        }

        *id
    }

    for t in indices.iter() {
        let va = resolve_coord_id(coords[t[0] as usize], &mut vtx_to_id, &mut new_coords);
        let oa = t[0];

        let vb = resolve_coord_id(coords[t[1] as usize], &mut vtx_to_id, &mut new_coords);
        let ob = t[1];

        let vc = resolve_coord_id(coords[t[2] as usize], &mut vtx_to_id, &mut new_coords);
        let oc = t[2];

        out.push([[va, oa, oa], [vb, ob, ob], [vc, oc, oc]]);
    }

    new_coords.shrink_to_fit();

    (out, new_coords)
}

// TODO: check at compile-time that we are in 3D?
/// Computes the normals of a set of vertices.
#[inline]
pub fn compute_normals(coordinates: &[Vec3], faces: &[[u32; 3]], normals: &mut Vec<Vec3>) {
    let mut divisor: Vec<f32> = vec![0.0; coordinates.len()];

    // Shrink the output buffer if it is too big.
    if normals.len() > coordinates.len() {
        normals.truncate(coordinates.len())
    }

    // Reinit all normals to zero.
    normals.clear();
    normals.extend(std::iter::repeat_n(Vec3::ZERO, coordinates.len()));

    // Accumulate normals ...
    for f in faces.iter() {
        let edge1 = coordinates[f[1] as usize] - coordinates[f[0] as usize];
        let edge2 = coordinates[f[2] as usize] - coordinates[f[0] as usize];
        let cross = edge1.cross(edge2);

        let normal = if cross.length_squared() > 0.0 {
            cross.normalize()
        } else {
            cross
        };

        normals[f[0] as usize] += normal;
        normals[f[1] as usize] += normal;
        normals[f[2] as usize] += normal;

        divisor[f[0] as usize] += 1.0;
        divisor[f[1] as usize] += 1.0;
        divisor[f[2] as usize] += 1.0;
    }

    // ... and compute the mean
    for (n, divisor) in normals.iter_mut().zip(divisor.iter()) {
        *n /= *divisor
    }
}
