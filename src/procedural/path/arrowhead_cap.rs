use crate::procedural::path::PolylineCompatibleCap;
use crate::procedural::utils;
use glamx::{Pose3, Vec3};

/// A cap that looks like an arrow.
pub struct ArrowheadCap {
    radius_scale: f32,
    front_dist_to_head: f32,
    back_dist_to_head: f32,
}

impl ArrowheadCap {
    /// Creates a cap that looks like an arrow.
    ///
    /// # Arguments:
    /// * `radius_scale` - scale factor of the cap base.
    /// * `front_dist_to_head` - distance from the path endpoint and the arrow tip.
    /// * `back_dist_to_head` - distance from the path endpoint and the cap base.
    pub fn new(radius_scale: f32, front_dist_to_head: f32, back_dist_to_head: f32) -> ArrowheadCap {
        ArrowheadCap {
            radius_scale,
            front_dist_to_head,
            back_dist_to_head,
        }
    }

    fn do_gen_cap(
        &self,
        attach_id: u32,
        pattern: &[Vec3],
        pt: Vec3,
        dir: Vec3,
        closed: bool,
        negative_shifts: bool,
        coords: &mut Vec<Vec3>,
        indices: &mut Vec<[u32; 3]>,
    ) {
        let front_dist_to_head = if negative_shifts {
            -self.front_dist_to_head
        } else {
            self.front_dist_to_head
        };
        let back_dist_to_head = if negative_shifts {
            -self.back_dist_to_head
        } else {
            self.back_dist_to_head
        };
        let pointy_thing = pt + dir * front_dist_to_head;
        let start_id = coords.len() as u32;
        let npts = pattern.len() as u32;
        let mut attach_id = attach_id;

        if !(self.radius_scale == 1.0) || back_dist_to_head != 0.0 {
            let mut new_pattern: Vec<Vec3> =
                pattern.iter().map(|p| *p * self.radius_scale).collect();

            // NOTE: this is done exactly the same on the PolylinePattern::stroke method.
            // Refactor?

            let back_shift = dir * back_dist_to_head;

            let transform = if dir.x == 0.0 && dir.z == 0.0 {
                // TODO: this might not be enough to avoid singularities.
                Pose3::face_towards(pt - back_shift, pt + dir, Vec3::X)
            } else {
                Pose3::face_towards(pt - back_shift, pt + dir, Vec3::Y)
            };

            for p in &mut new_pattern {
                *p = transform * *p
            }

            coords.extend(new_pattern);

            if closed {
                utils::push_ring_indices(attach_id, start_id, npts, indices)
            } else {
                utils::push_open_ring_indices(attach_id, start_id, npts, indices)
            }

            attach_id = start_id;
        }

        if closed {
            utils::push_degenerate_top_ring_indices(attach_id, coords.len() as u32, npts, indices);
        } else {
            utils::push_degenerate_open_top_ring_indices(
                attach_id,
                coords.len() as u32,
                npts,
                indices,
            );
        }

        coords.push(pointy_thing);
    }
}

impl PolylineCompatibleCap for ArrowheadCap {
    fn gen_end_cap(
        &self,
        attach_id: u32,
        pattern: &[Vec3],
        pt: Vec3,
        dir: Vec3,
        closed: bool,
        coords: &mut Vec<Vec3>,
        indices: &mut Vec<[u32; 3]>,
    ) {
        let start_indices_id = indices.len();

        self.do_gen_cap(attach_id, pattern, pt, dir, closed, false, coords, indices);
        utils::reverse_clockwising(&mut indices[start_indices_id..])
    }

    fn gen_start_cap(
        &self,
        attach_id: u32,
        pattern: &[Vec3],
        pt: Vec3,
        dir: Vec3,
        closed: bool,
        coords: &mut Vec<Vec3>,
        indices: &mut Vec<[u32; 3]>,
    ) {
        self.do_gen_cap(attach_id, pattern, pt, dir, closed, true, coords, indices)
    }
}
