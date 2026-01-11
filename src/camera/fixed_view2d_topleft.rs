use glamx::{Mat3, Vec2, Vec3, Vec3Swizzles};

use crate::{camera::Camera2d, event::WindowEvent, window::Canvas};

/// A camera with top left origin and pixel coordinates.
///
/// This camera is designed for 2D games and UI applications where:
/// - (0, 0) is at the top left corner of the window
/// - Y axis points downward (increasing Y moves down)
/// - Coordinates map directly to screen pixels (no HiDPI scaling)
///
/// This coordinate system will be familiar to users coming from:
/// - Raylib
/// - macroquad  
/// - HTML Canvas
///
/// # Note on HiDPI/Retina Displays
///
/// This camera does NOT apply HiDPI scaling and makes a few assumptions:
///
/// 1. Your game logic works with integer pixel coordinates
/// 2. You want behavior the same across X11, Wayland, Windows, macOS
/// 3. On Wayland, scale factors can be fractional (1.25, 1.5, etc.),
///    which would complicate pixel perfect rendering
/// 4. For tile based or grid based games, you want exact pixel alignment
///
/// On high DPI displays, your content will render at native resolution but may appear smaller.
///
/// If you need DPI aware scaling, consider:
/// - Using `FixedView2d` (center origin, includes scaling)
/// - Implementing your own scaling logic in game code
/// - Implementing a custom camera that handles fractional scaling
///
/// # Example
///
/// ```rust
/// let mut camera = FixedView2dTopLeft::new();
///
/// // Draw at pixel position (100, 50)
/// window.draw_circle_2d(Vec2::new(100.0, 50.0), 10.0, RED);
///
/// // Mouse position maps directly to world coordinates
/// let mouse_world = camera.unproject(mouse_pos, window_size);
///
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
