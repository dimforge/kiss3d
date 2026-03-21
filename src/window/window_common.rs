//! The `Window` struct and shared accessor methods, used by both the windowed
//! (`window::Window`) and DRM (`drm::Window`) backends.
//!
//! The struct is defined here once. The two canvas-specific fields (`canvas`
//! and the event storage) are gated with `#[cfg]`. Everything else — all ten
//! renderer/state fields — is declared exactly once and shared.

use std::cell::RefCell;
use std::rc::Rc;

use glamx::UVec2;

use crate::color::Color;
use crate::event::{Action, Key, MouseButton};
use crate::renderer::{PointRenderer2d, PointRenderer3d, PolylineRenderer2d, PolylineRenderer3d};
use crate::resource::{FramebufferManager, RenderTarget};
use crate::text::TextRenderer;

// Canvas and event types are backend-specific.
#[cfg(feature = "drm")]
use super::drm::drm_canvas::DrmCanvas;
#[cfg(feature = "drm")]
use super::drm::drm_events::DrmEventManager;
#[cfg(not(feature = "drm"))]
use crate::event::WindowEvent;
#[cfg(not(feature = "drm"))]
use crate::window::Canvas;
#[cfg(not(feature = "drm"))]
use std::sync::mpsc::Receiver;

#[cfg(feature = "egui")]
use super::egui_integration::EguiContext;
#[cfg(feature = "recording")]
use super::recording::RecordingState;

/// Structure representing a window and a 3D scene.
///
/// This is the main interface with the 3D engine. It works identically whether
/// backed by a native OS window (winit) or a DRM/KMS display (headless).
pub struct Window {
    // ── Canvas — backend-specific ─────────────────────────────────────────
    /// Winit/wgpu surface canvas (non-DRM builds).
    #[cfg(not(feature = "drm"))]
    pub(crate) canvas: Canvas,
    /// Offscreen wgpu canvas driven directly via DRM/KMS (DRM builds).
    #[cfg(feature = "drm")]
    pub(crate) canvas: DrmCanvas,

    // ── Event storage — backend-specific ──────────────────────────────────
    /// Winit event channel (non-DRM builds).
    #[cfg(not(feature = "drm"))]
    pub(crate) events: Rc<Receiver<WindowEvent>>,
    #[cfg(not(feature = "drm"))]
    pub(crate) unhandled_events: Rc<RefCell<Vec<WindowEvent>>>,
    /// DRM event manager (DRM builds).
    #[cfg(feature = "drm")]
    pub(crate) event_manager: Rc<RefCell<DrmEventManager>>,

    // ── Shared fields ─────────────────────────────────────────────────────
    pub(crate) ambient_intensity: f32,
    pub(crate) background: Color,
    pub(crate) polyline_renderer_2d: PolylineRenderer2d,
    pub(crate) point_renderer_2d: PointRenderer2d,
    pub(crate) point_renderer: PointRenderer3d,
    pub(crate) polyline_renderer: PolylineRenderer3d,
    pub(crate) text_renderer: TextRenderer,
    #[allow(dead_code)]
    pub(crate) framebuffer_manager: FramebufferManager,
    pub(crate) post_process_render_target: RenderTarget,
    pub(crate) should_close: bool,
    #[cfg(feature = "egui")]
    pub(crate) egui_context: EguiContext,
    #[cfg(feature = "recording")]
    pub(crate) recording: Option<RecordingState>,
}

impl Window {
    /// Indicates whether this window should be closed.
    ///
    /// Returns `true` after [`close()`](Self::close) has been called, or after
    /// an `Escape` key or window-close event has been received.
    #[inline]
    pub fn should_close(&self) -> bool {
        self.should_close
    }

    /// Closes the window.
    ///
    /// After calling this method, [`render()`](Self::render) will return `false`
    /// on the next frame, allowing the render loop to exit gracefully.
    #[inline]
    pub fn close(&mut self) {
        self.should_close = true;
    }

    /// Returns the width of the render target in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.canvas.size().0
    }

    /// Returns the height of the render target in pixels.
    #[inline]
    pub fn height(&self) -> u32 {
        self.canvas.size().1
    }

    /// Returns the dimensions of the render target.
    #[inline]
    pub fn size(&self) -> UVec2 {
        let (w, h) = self.canvas.size();
        UVec2::new(w, h)
    }

    /// Sets the background clear colour.
    ///
    /// # Arguments
    /// * `color` - The background colour to use
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// use kiss3d::color::DARK_BLUE;
    /// let mut window = Window::new("Example").await;
    /// window.set_background_color(DARK_BLUE);
    /// # }
    /// ```
    #[inline]
    pub fn set_background_color(&mut self, color: Color) {
        self.background = color;
    }

    /// Sets the ambient light intensity for the scene.
    ///
    /// # Arguments
    /// * `ambient` - The ambient light intensity (typically 0.0 to 1.0)
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// // Set global ambient lighting intensity
    /// window.set_ambient(0.3);
    /// # }
    /// ```
    ///
    /// Note: Individual lights should be added to the scene tree using
    /// `SceneNode3d::add_point_light()`, `add_directional_light()`, or `add_spot_light()`.
    #[inline]
    pub fn set_ambient(&mut self, ambient: f32) {
        self.ambient_intensity = ambient;
    }

    /// Returns the current ambient light intensity.
    #[inline]
    pub fn ambient(&self) -> f32 {
        self.ambient_intensity
    }

    /// Returns the DPI scale factor of the display.
    ///
    /// This is the ratio between physical pixels and logical pixels.
    /// On high-DPI displays (like Retina displays) this will be greater than 1.0.
    /// On the DRM/headless backend this always returns 1.0.
    ///
    /// # Returns
    /// The scale factor (e.g., 1.0 for standard displays, 2.0 for Retina displays)
    #[inline]
    pub fn scale_factor(&self) -> f64 {
        self.canvas.scale_factor()
    }

    /// Returns the last known position of the mouse cursor, or `None` if unknown.
    ///
    /// The position is automatically updated when the mouse moves over the window.
    /// Coordinates are in pixels, with (0, 0) at the top-left corner.
    /// On the DRM/headless backend this always returns `None`.
    ///
    /// # Returns
    /// `Some((x, y))` with the cursor position, or `None` if the cursor position is unknown
    #[inline]
    pub fn cursor_pos(&self) -> Option<(f64, f64)> {
        self.canvas.cursor_pos()
    }

    /// Returns the current state of a keyboard key.
    ///
    /// # Arguments
    /// * `key` - The key to check
    ///
    /// # Returns
    /// The current `Action` state (e.g., `Action::Press`, `Action::Release`)
    ///
    /// On the DRM/headless backend this always returns `Action::Release`.
    #[inline]
    pub fn get_key(&self, key: Key) -> Action {
        self.canvas.get_key(key)
    }

    /// Returns the current state of a mouse button.
    ///
    /// # Arguments
    /// * `button` - The mouse button to check
    ///
    /// # Returns
    /// The current `Action` state (e.g., `Action::Press`, `Action::Release`)
    ///
    /// On the DRM/headless backend this always returns `Action::Release`.
    #[inline]
    pub fn get_mouse_button(&self, button: MouseButton) -> Action {
        self.canvas.get_mouse_button(button)
    }
}
