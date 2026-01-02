use crate::camera::Camera2d;
use crate::event::WindowEvent;
use crate::window::Canvas;
use glamx::{Mat3, Vec2, Vec3, Vec3Swizzles};

/// A camera that cannot move.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FixedView2d {
    proj: Mat3,
    inv_proj: Mat3,
}

impl Default for FixedView2d {
    fn default() -> Self {
        Self::new()
    }
}

impl FixedView2d {
    /// Create a new static camera.
    pub fn new() -> FixedView2d {
        FixedView2d {
            proj: Mat3::IDENTITY,
            inv_proj: Mat3::IDENTITY,
        }
    }
}

impl Camera2d for FixedView2d {
    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent) {
        let scale = canvas.scale_factor();

        if let WindowEvent::FramebufferSize(w, h) = *event {
            let diag = Vec3::new(
                2.0 * (scale as f32) / (w as f32),
                2.0 * (scale as f32) / (h as f32),
                1.0,
            );
            let inv_diag = Vec3::new(1.0 / diag.x, 1.0 / diag.y, 1.0);

            self.proj = Mat3::from_diagonal(diag);
            self.inv_proj = Mat3::from_diagonal(inv_diag);
        }
    }

    #[inline]
    fn view_transform_pair(&self) -> (Mat3, Mat3) {
        (Mat3::IDENTITY, self.proj)
    }

    fn update(&mut self, _: &Canvas) {}

    fn unproject(&self, window_coord: Vec2, size: Vec2) -> Vec2 {
        let normalized_coords = Vec2::new(
            2.0 * window_coord.x / size.x - 1.0,
            2.0 * -window_coord.y / size.y + 1.0,
        );

        let normalized_homogeneous = Vec3::new(normalized_coords.x, normalized_coords.y, 1.0);
        let unprojected_homogeneous = self.inv_proj * normalized_homogeneous;
        unprojected_homogeneous.xy() / unprojected_homogeneous.z
    }
}
