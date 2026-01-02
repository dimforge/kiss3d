use glamx::{Pose2, Vec2};

/// Geometric description of a polyline.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RenderPolyline {
    /// Coordinates of the polyline vertices.
    coords: Vec<Vec2>,
    /// Coordinates of the polyline normals.
    normals: Option<Vec<Vec2>>,
}

impl RenderPolyline {
    /// Creates a new polyline.
    pub fn new(coords: Vec<Vec2>, normals: Option<Vec<Vec2>>) -> RenderPolyline {
        if let Some(ref ns) = normals {
            assert!(
                coords.len() == ns.len(),
                "There must be exactly one normal per vertex."
            );
        }

        RenderPolyline { coords, normals }
    }
}

impl RenderPolyline {
    /// Moves the polyline data out of it.
    pub fn unwrap(self) -> (Vec<Vec2>, Option<Vec<Vec2>>) {
        (self.coords, self.normals)
    }

    /// The coordinates of this polyline vertices.
    #[inline]
    pub fn coords(&self) -> &[Vec2] {
        &self.coords[..]
    }

    /// The mutable coordinates of this polyline vertices.
    #[inline]
    pub fn coords_mut(&mut self) -> &mut [Vec2] {
        &mut self.coords[..]
    }

    /// The normals of this polyline vertices.
    #[inline]
    pub fn normals(&self) -> Option<&[Vec2]> {
        self.normals.as_deref()
    }

    /// The mutable normals of this polyline vertices.
    #[inline]
    pub fn normals_mut(&mut self) -> Option<&mut [Vec2]> {
        self.normals.as_deref_mut()
    }

    /// Translates each vertex of this polyline.
    pub fn translate_by(&mut self, t: Vec2) {
        for c in self.coords.iter_mut() {
            *c += t;
        }
    }

    /// Rotates each vertex and normal of this polyline by an angle (in radians).
    pub fn rotate_by(&mut self, angle: f32) {
        let (sin, cos) = angle.sin_cos();
        for c in self.coords.iter_mut() {
            let x = cos * c.x - sin * c.y;
            let y = sin * c.x + cos * c.y;
            *c = Vec2::new(x, y);
        }

        for n in self.normals.iter_mut() {
            for n in n.iter_mut() {
                let x = cos * n.x - sin * n.y;
                let y = sin * n.x + cos * n.y;
                *n = Vec2::new(x, y);
            }
        }
    }

    /// Transforms each vertex and rotates each normal of this polyline.
    pub fn transform_by(&mut self, t: Pose2) {
        for c in self.coords.iter_mut() {
            *c = t * *c;
        }

        for n in self.normals.iter_mut() {
            for n in n.iter_mut() {
                *n = t.rotation * *n;
            }
        }
    }

    /// Apply a transformation to every vertex and normal of this polyline and returns it.
    #[inline]
    pub fn transformed(mut self, t: Pose2) -> Self {
        self.transform_by(t);
        self
    }

    /// Scales each vertex of this polyline.
    pub fn scale_by_scalar(&mut self, s: f32) {
        for c in self.coords.iter_mut() {
            *c *= s
        }
        // TODO: do something for the normals?
    }

    /// Scales each vertex of this mesh.
    #[inline]
    pub fn scale_by(&mut self, s: Vec2) {
        for c in self.coords.iter_mut() {
            *c *= s;
        }
        // TODO: do something for the normals?
    }

    /// Apply a scaling to every vertex and normal of this polyline and returns it.
    #[inline]
    pub fn scaled(mut self, s: Vec2) -> Self {
        self.scale_by(s);
        self
    }
}
