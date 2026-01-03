use crate::camera::Camera2d;
use crate::event::{Action, MouseButton, WindowEvent};
use crate::window::Canvas;
use glamx::{Mat3, Vec2, Vec3, Vec3Swizzles};
use num::Pow;

/// A 2D camera that can be zoomed and panned.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PanZoomCamera2d {
    at: Vec2,
    /// Distance from the camera to the `at` focus point.
    zoom: f32,

    /// Increment of the zoom per unit scrolling. The default value is 40.0.
    zoom_step: f32,
    drag_button: Option<MouseButton>,

    view: Mat3,
    proj: Mat3,
    scaled_proj: Mat3,
    inv_scaled_proj: Mat3,
    last_cursor_pos: Vec2,
}

impl Default for PanZoomCamera2d {
    fn default() -> Self {
        Self::new(Vec2::ZERO, 1.0)
    }
}

impl PanZoomCamera2d {
    /// Create a new arc-ball camera.
    pub fn new(eye: Vec2, zoom: f32) -> PanZoomCamera2d {
        let mut res = PanZoomCamera2d {
            at: eye,
            zoom,
            zoom_step: 0.9,
            drag_button: Some(MouseButton::Button2),
            view: Mat3::IDENTITY,
            proj: Mat3::IDENTITY,
            scaled_proj: Mat3::IDENTITY,
            inv_scaled_proj: Mat3::IDENTITY,
            last_cursor_pos: Vec2::ZERO,
        };

        res.update_projviews();

        res
    }

    /// The point the arc-ball is looking at.
    pub fn at(&self) -> Vec2 {
        self.at
    }

    /// Get a mutable reference to the point the camera is looking at.
    pub fn set_at(&mut self, at: Vec2) {
        self.at = at;
        self.update_projviews();
    }

    /// Gets the zoom of the camera.
    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    /// Sets the zoom of the camera.
    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom;

        self.update_restrictions();
        self.update_projviews();
    }

    /// Move the camera such that it is centered on a specific point.
    pub fn look_at(&mut self, at: Vec2, zoom: f32) {
        self.at = at;
        self.zoom = zoom;
        self.update_projviews();
    }

    /// Transformation applied by the camera without perspective.
    fn update_restrictions(&mut self) {
        if self.zoom < 0.00001 {
            self.zoom = 0.00001
        }
    }

    /// The button used to drag the PanZoomCamera2d camera.
    pub fn drag_button(&self) -> Option<MouseButton> {
        self.drag_button
    }

    /// Set the button used to drag the PanZoomCamera2d camera.
    /// Use None to disable dragging.
    pub fn rebind_drag_button(&mut self, new_button: Option<MouseButton>) {
        self.drag_button = new_button;
    }

    /// Move the camera based on drag from right mouse button
    /// `dpos` is assumed to be in window space so the y-axis is flipped
    fn handle_right_button_displacement(&mut self, dpos: Vec2) {
        self.at.x -= dpos.x / self.zoom;
        self.at.y += dpos.y / self.zoom;
        self.update_projviews();
    }

    fn handle_scroll(&mut self, off: f32) {
        #[cfg(target_arch = "wasm32")] // TODO: not sure why it’s weaker on wasm32
        let off = off * 10.0;
        self.zoom /= self.zoom_step.pow(off / 120.0);
        self.update_restrictions();
        self.update_projviews();
    }

    fn update_projviews(&mut self) {
        // Create translation matrix: translate by -at
        self.view = Mat3::from_cols(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-self.at.x, -self.at.y, 1.0),
        );

        self.scaled_proj = self.proj;
        // Scale x and y components (first two diagonal elements)
        self.scaled_proj.col_mut(0)[0] *= self.zoom;
        self.scaled_proj.col_mut(1)[1] *= self.zoom;

        self.inv_scaled_proj.col_mut(0)[0] = 1.0 / self.scaled_proj.col(0)[0];
        self.inv_scaled_proj.col_mut(1)[1] = 1.0 / self.scaled_proj.col(1)[1];
    }
}

impl Camera2d for PanZoomCamera2d {
    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent) {
        let scale = 1.0; // canvas.scale_factor();

        match *event {
            WindowEvent::CursorPos(x, y, _) => {
                let curr_pos = Vec2::new(x as f32, y as f32);

                if let Some(drag_button) = self.drag_button {
                    if canvas.get_mouse_button(drag_button) == Action::Press {
                        let dpos = curr_pos - self.last_cursor_pos;
                        self.handle_right_button_displacement(dpos)
                    }
                }

                self.last_cursor_pos = curr_pos;
            }
            WindowEvent::Scroll(_, off, _) => self.handle_scroll(off as f32),
            WindowEvent::FramebufferSize(w, h) => {
                self.proj = Mat3::from_cols(
                    Vec3::new(2.0 * (scale as f32) / (w as f32), 0.0, 0.0),
                    Vec3::new(0.0, 2.0 * (scale as f32) / (h as f32), 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                );
                self.update_projviews();
            }
            _ => {}
        }
    }

    #[inline]
    fn view_transform_pair(&self) -> (Mat3, Mat3) {
        (self.view, self.scaled_proj)
    }

    fn update(&mut self, _: &Canvas) {}

    /// Calculate the global position of the given window coordinate
    fn unproject(&self, window_coord: Vec2, size: Vec2) -> Vec2 {
        // Convert window coordinates (origin at top left) to normalized screen coordinates
        // (origin at the center of the screen)
        let normalized_coords = Vec2::new(
            2.0 * window_coord.x / size.x - 1.0,
            2.0 * -window_coord.y / size.y + 1.0,
        );

        // Project normalized screen coordinate to screen space
        let normalized_homogeneous = Vec3::new(normalized_coords.x, normalized_coords.y, 1.0);
        let unprojected_homogeneous = self.inv_scaled_proj * normalized_homogeneous;

        // Convert from screen space to global space
        let screen_pos = unprojected_homogeneous.xy() / unprojected_homogeneous.z;
        screen_pos + self.at
    }
}
