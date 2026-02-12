//! Canvas wrapper for DRM to satisfy camera interface requirements.

use crate::event::{Action, Key, MouseButton};
use crate::window::drm::DrmCanvas;

/// A wrapper around DrmCanvas that provides the same interface as Canvas
/// for camera compatibility.
///
/// In headless mode, there are no input events, so all key/mouse queries
/// return "not pressed" states.
pub struct DrmCanvasWrapper<'a> {
    canvas: &'a DrmCanvas,
}

impl<'a> DrmCanvasWrapper<'a> {
    /// Creates a new wrapper around a DrmCanvas.
    pub fn new(canvas: &'a DrmCanvas) -> Self {
        Self { canvas }
    }

    /// The size of the render target.
    pub fn size(&self) -> (u32, u32) {
        self.canvas.size()
    }

    /// Gets the surface format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.canvas.surface_format()
    }

    /// The current position of the cursor (always None in headless mode).
    pub fn cursor_pos(&self) -> Option<(f64, f64)> {
        None
    }

    /// The scale factor (always 1.0 in headless mode).
    pub fn scale_factor(&self) -> f64 {
        1.0
    }

    /// The state of a mouse button (always Released in headless mode).
    pub fn get_mouse_button(&self, _button: MouseButton) -> Action {
        Action::Release
    }

    /// The state of a key (always Released in headless mode).
    pub fn get_key(&self, _key: Key) -> Action {
        Action::Release
    }

    /// Gets the sample count for MSAA.
    pub fn sample_count(&self) -> u32 {
        self.canvas.sample_count()
    }

    // The following methods are window-specific and no-ops for headless rendering:

    /// Set the window title (no-op in headless mode).
    pub fn set_title(&mut self, _title: &str) {}

    /// Set the cursor grabbing behaviour (no-op in headless mode).
    pub fn set_cursor_grab(&self, _grab: bool) {}

    /// Set the cursor position (no-op in headless mode).
    pub fn set_cursor_position(&self, _x: f64, _y: f64) {}

    /// Toggle the cursor visibility (no-op in headless mode).
    pub fn hide_cursor(&self, _hide: bool) {}

    /// Hide the window (no-op in headless mode).
    pub fn hide(&mut self) {}

    /// Show the window (no-op in headless mode).
    pub fn show(&mut self) {}
}
