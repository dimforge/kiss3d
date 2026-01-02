use crate::procedural::path::{CurveSampler, PathSample};
use glamx::Vec3;

/// A path with its sample points given by a polyline.
///
/// This will return sequentially each vertex of the polyline.
pub struct PolylinePath<'a> {
    curr_len: f32,
    curr_dir: Vec3,
    curr_pt_id: usize,
    curr_pt: Vec3,
    polyline: &'a [Vec3],
}

impl<'a> PolylinePath<'a> {
    /// Creates a new polyline-based path.
    pub fn new(polyline: &'a [Vec3]) -> PolylinePath<'a> {
        assert!(
            polyline.len() > 1,
            "The polyline must have at least two points."
        );

        let diff = polyline[1] - polyline[0];
        let len = diff.length();
        let dir = diff / len;

        PolylinePath {
            curr_len: len,
            curr_dir: dir,
            curr_pt_id: 0,
            curr_pt: polyline[0],
            polyline,
        }
    }
}

impl CurveSampler for PolylinePath<'_> {
    fn next(&mut self) -> PathSample {
        let poly_coords = self.polyline;

        let result = if self.curr_pt_id == 0 {
            PathSample::StartPoint(self.curr_pt, self.curr_dir)
        } else if self.curr_pt_id < poly_coords.len() - 1 {
            PathSample::InnerPoint(self.curr_pt, self.curr_dir)
        } else if self.curr_pt_id == poly_coords.len() - 1 {
            PathSample::EndPoint(self.curr_pt, self.curr_dir)
        } else {
            PathSample::EndOfSample
        };

        self.curr_pt_id += 1;

        if self.curr_pt_id < poly_coords.len() {
            self.curr_pt = poly_coords[self.curr_pt_id];

            if self.curr_pt_id < poly_coords.len() - 1 {
                let curr_diff = poly_coords[self.curr_pt_id + 1] - poly_coords[self.curr_pt_id];
                self.curr_len = curr_diff.length();
                self.curr_dir = curr_diff / self.curr_len;
            }
        }

        result
    }
}
