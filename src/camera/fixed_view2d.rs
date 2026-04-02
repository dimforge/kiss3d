use crate::camera::Camera2d;
use crate::event::WindowEvent;
use crate::window::Canvas;
use glamx::{Mat3, Vec2, Vec3, Vec3Swizzles};

/// The coordinate system used by a [`FixedView2d`] camera.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CoordinateSystem2d {
    /// Origin at the center of the screen, Y axis pointing upward.
    ///
    /// This is the default coordinate system, following mathematical conventions.
    #[default]
    CenterUp,
    /// Origin at the top-left corner, Y axis pointing downward.
    ///
    /// This coordinate system is familiar to users coming from Raylib, macroquad,
    /// HTML Canvas, etc. Coordinates map directly to screen pixels if HiDPI is disabled.
    TopLeftDown,
}

/// A static 2D camera with configurable coordinate system and HiDPI support.
///
/// # Coordinate systems
///
/// Use [`CoordinateSystem2d::CenterUp`] (default) for a centered origin with Y pointing up,
/// or [`CoordinateSystem2d::TopLeftDown`] for a top-left origin with Y pointing down
/// (pixel coordinates).
///
/// # HiDPI scaling
///
/// When `apply_hidpi` is `true` (the default), the camera applies the display's scale factor
/// so that coordinates are in logical pixels. When `false`, coordinates map directly to
/// physical pixels — useful for pixel-perfect rendering on high-DPI displays.
///
/// # Example
///
/// ```rust
/// # use kiss3d::prelude::{FixedView2d, CoordinateSystem2d};
/// // Center-origin camera with HiDPI (default):
/// let cam = FixedView2d::default();
///
/// // Top-left pixel camera without HiDPI:
/// let cam = FixedView2d::new(CoordinateSystem2d::TopLeftDown, false);
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FixedView2d {
    proj: Mat3,
    inv_proj: Mat3,
    coord_system: CoordinateSystem2d,
    apply_hidpi: bool,
}

impl Default for FixedView2d {
    fn default() -> Self {
        Self::new(CoordinateSystem2d::CenterUp, true)
    }
}

impl FixedView2d {
    /// Create a new static camera with the given coordinate system and HiDPI setting.
    pub fn new(coord_system: CoordinateSystem2d, apply_hidpi: bool) -> FixedView2d {
        FixedView2d {
            proj: Mat3::IDENTITY,
            inv_proj: Mat3::IDENTITY,
            coord_system,
            apply_hidpi,
        }
    }
}

impl Camera2d for FixedView2d {
    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent) {
        if let WindowEvent::FramebufferSize(w, h) = *event {
            let scale = if self.apply_hidpi {
                canvas.scale_factor() as f32
            } else {
                1.0
            };

            let w = w as f32;
            let h = h as f32;

            let proj = match self.coord_system {
                CoordinateSystem2d::CenterUp => {
                    let diag = Vec3::new(2.0 * scale / w, 2.0 * scale / h, 1.0);
                    Mat3::from_diagonal(diag)
                }
                CoordinateSystem2d::TopLeftDown => Mat3::from_cols(
                    Vec3::new(2.0 * scale / w, 0.0, 0.0),
                    Vec3::new(0.0, -2.0 * scale / h, 0.0),
                    Vec3::new(-1.0, 1.0, 1.0),
                ),
            };

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
            2.0 * -window_coord.y / size.y + 1.0,
        );

        let normalized_homogeneous = Vec3::new(normalized_coords.x, normalized_coords.y, 1.0);
        let unprojected_homogeneous = self.inv_proj * normalized_homogeneous;
        unprojected_homogeneous.xy() / unprojected_homogeneous.z
    }
}
