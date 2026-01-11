use glamx::{Mat3, Vec2, Vec3, Vec3Swizzles};

use crate::{camera::Camera2d, event::WindowEvent, window::Canvas};

/// A camera that cannot move, with top-left origin.
/// This will be familar to users coming from Raylib or macroquad
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FixedView2dTopLeft {
    proj: Mat3,
    inv_proj: Mat3,
}

impl Default for FixedView2dTopLeft {
    fn default() -> Self {
        Self::new()
    }
}

impl FixedView2dTopLeft {
    /// Create a new static camera.
    pub fn new() -> FixedView2dTopLeft {
        FixedView2dTopLeft {
            proj: Mat3::IDENTITY,
            inv_proj: Mat3::IDENTITY,
        }
    }
}

impl Camera2d for FixedView2dTopLeft {
    fn handle_event(&mut self, _canvas: &Canvas, event: &WindowEvent) {
        if let WindowEvent::FramebufferSize(w, h) = *event {
            let proj = Mat3::from_cols(
                Vec3::new(2.0 / w as f32, 0.0, 0.0),
                Vec3::new(0.0, -2.0 / h as f32, 0.0),
                Vec3::new(-1.0, 1.0, 1.0),
            );

            self.proj = proj;
            self.inv_proj = proj.inverse();
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
            1.0 - 2.0 * window_coord.y / size.y,
        );

        let normalized_homogeneous = Vec3::new(normalized_coords.x, normalized_coords.y, 1.0);
        let unprojected_homogeneous = self.inv_proj * normalized_homogeneous;
        unprojected_homogeneous.xy() / unprojected_homogeneous.z
    }
}
