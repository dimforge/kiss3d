use std::sync::mpsc::Sender;

use crate::event::{Action, Key, MouseButton, WindowEvent};
use crate::window::WgpuCanvas;
use image::{GenericImage, Pixel};
use winit::window::WindowAttributes;

/// The possible number of samples for multisample anti-aliasing.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NumSamples {
    /// One sample
    One = 1,
    /// Four samples
    Four = 4,
}

impl NumSamples {
    /// Create a `NumSamples` from a number.
    /// Returns `None` if `i` is invalid.
    pub fn from_u32(i: u32) -> Option<NumSamples> {
        match i {
            1 => Some(NumSamples::One),
            4 => Some(NumSamples::Four),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// Canvas options.
pub struct CanvasSetup {
    /// Is vsync enabled?
    pub vsync: bool,
    /// Number of AA samples.
    pub samples: NumSamples,
    /// The id of the canvas element to use (WASM only).
    /// Defaults to `"canvas"`. If an element with this id exists in the DOM,
    /// it will be used; otherwise a new canvas element with this id is created.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::{Window, CanvasSetup};
    /// let setup = CanvasSetup {
    ///     canvas_id: "my-3d-canvas".to_string(),
    ///     ..Default::default()
    /// };
    /// let mut window = Window::new_with_setup("Title", 800, 600, setup);
    /// ```
    pub canvas_id: String,
    /// Extra wgpu device features to request when creating the device, on top of
    /// the ones kiss3d enables by default.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub required_features: wgpu::Features,
}

impl Default for CanvasSetup {
    fn default() -> Self {
        CanvasSetup {
            vsync: true,
            samples: NumSamples::Four,
            canvas_id: "canvas".to_string(),
            required_features: wgpu::Features::empty(),
        }
    }
}

/// An abstract structure representing a window for native applications, and a canvas for web applications.
pub struct Canvas {
    canvas: WgpuCanvas,
}

impl Canvas {
    /// Open a new window, and initialize the wgpu context.
    pub async fn open(
        window_attrs: WindowAttributes,
        canvas_setup: Option<CanvasSetup>,
        out_events: Sender<WindowEvent>,
    ) -> Self {
        Canvas {
            canvas: WgpuCanvas::open(window_attrs, canvas_setup, out_events).await,
        }
    }

    /// Open a headless canvas (no window) for off-screen rendering.
    pub async fn open_headless(
        width: u32,
        height: u32,
        canvas_setup: Option<CanvasSetup>,
        out_events: Sender<WindowEvent>,
    ) -> Self {
        Canvas {
            canvas: WgpuCanvas::open_headless(width, height, canvas_setup, out_events).await,
        }
    }

    /// Poll all events that occurred since the last call to this method.
    pub fn poll_events(&mut self) {
        self.canvas.poll_events()
    }

    /// Wait for an event, a [`super::Waker`] or `timeout`, then poll; see
    /// [`WgpuCanvas::wait_events`].
    pub fn wait_events(&mut self, timeout: Option<std::time::Duration>) -> bool {
        self.canvas.wait_events(timeout)
    }

    /// What ends a [`Self::wait_events`] from another thread.
    pub fn waker(&self) -> Option<super::Waker> {
        self.canvas.waker()
    }

    /// Resizes the canvas render targets.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.canvas.resize(width, height)
    }

    /// Gets the current surface texture for rendering.
    pub fn get_current_texture(&self) -> Option<wgpu::SurfaceTexture> {
        self.canvas.get_current_texture()
    }

    /// Presents the current frame.
    pub fn present(&self, frame: wgpu::SurfaceTexture) {
        self.canvas.present(frame)
    }

    /// Gets the depth texture view for rendering.
    pub fn depth_view(&self) -> &wgpu::TextureView {
        self.canvas.depth_view()
    }

    /// Gets the MSAA texture view if MSAA is enabled.
    pub fn msaa_view(&self) -> Option<&wgpu::TextureView> {
        self.canvas.msaa_view()
    }

    /// Gets the sample count for MSAA.
    pub fn sample_count(&self) -> u32 {
        self.canvas.sample_count()
    }

    /// Sets the MSAA sample count, recreating the render targets to match. Takes
    /// effect on the next rendered frame.
    pub fn set_samples(&mut self, samples: NumSamples) {
        self.canvas.set_sample_count(samples as u32)
    }

    /// Whether vsync is currently enabled.
    pub fn vsync(&self) -> bool {
        self.canvas.vsync()
    }

    /// Enables/disables vsync at runtime (reconfigures the surface present mode).
    pub fn set_vsync(&mut self, enabled: bool) {
        self.canvas.set_vsync(enabled)
    }

    /// Gets the surface format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.canvas.surface_format()
    }

    /// The size of the window.
    pub fn size(&self) -> (u32, u32) {
        self.canvas.size()
    }

    /// The current position of the cursor, if known.
    ///
    /// This position may not be known if, e.g., the cursor has not been moved since the
    /// window was open.
    pub fn cursor_pos(&self) -> Option<(f64, f64)> {
        self.canvas.cursor_pos()
    }

    /// The scale factor.
    pub fn scale_factor(&self) -> f64 {
        self.canvas.scale_factor()
    }

    /// Set the window title.
    pub fn set_title(&mut self, title: &str) {
        self.canvas.set_title(title)
    }

    /// Set the window icon. See `Window::set_icon` for details.
    pub fn set_icon(&mut self, icon: impl GenericImage<Pixel = impl Pixel<Subpixel = u8>>) {
        self.canvas.set_icon(icon)
    }

    /// Set the cursor grabbing behaviour.
    pub fn set_cursor_grab(&self, grab: bool) {
        self.canvas.set_cursor_grab(grab);
    }

    /// Enter or leave borderless fullscreen on the current monitor.
    pub fn set_fullscreen(&self, fullscreen: bool) {
        self.canvas.set_fullscreen(fullscreen);
    }

    /// Enter or leave exclusive fullscreen on the current monitor.
    pub fn set_exclusive_fullscreen(&self, exclusive: bool) {
        self.canvas.set_exclusive_fullscreen(exclusive);
    }

    /// Maximize the window, or restore it.
    pub fn set_maximized(&self, maximized: bool) {
        self.canvas.set_maximized(maximized);
    }

    /// Whether the window is currently maximized.
    pub fn is_maximized(&self) -> bool {
        self.canvas.is_maximized()
    }

    /// Show or hide the platform's on-screen keyboard (mobile; no-op elsewhere).
    pub fn set_keyboard_visible(&self, visible: bool) {
        self.canvas.set_keyboard_visible(visible);
    }

    /// Whether the window is currently fullscreen.
    pub fn is_fullscreen(&self) -> bool {
        self.canvas.is_fullscreen()
    }

    /// Files dropped onto the window since the last call, in drop order.
    pub fn take_dropped_files(&self) -> Vec<std::path::PathBuf> {
        self.canvas.take_dropped_files()
    }

    /// Composed text the input method reported since the last call.
    pub fn take_ime_events(&self) -> Vec<crate::event::ImeEvent> {
        self.canvas.take_ime_events()
    }

    /// Let the platform compose text through its input method.
    pub fn set_ime_allowed(&self, allowed: bool) {
        self.canvas.set_ime_allowed(allowed);
    }

    /// The display's insets in pixels as `[left, top, right, bottom]`.
    pub fn safe_area(&self) -> [f32; 4] {
        self.canvas.safe_area()
    }

    /// The reader's own preferred text size, as a multiplier of standard.
    pub fn text_scale(&self) -> f32 {
        self.canvas.text_scale()
    }

    /// Pixels of the window the on-screen keyboard covers, from the bottom.
    pub fn keyboard_height(&self) -> f32 {
        self.canvas.keyboard_height()
    }

    /// Set the cursor position.
    pub fn set_cursor_position(&self, x: f64, y: f64) {
        self.canvas.set_cursor_position(x, y);
    }

    /// Toggle the cursor visibility.
    pub fn hide_cursor(&self, hide: bool) {
        self.canvas.hide_cursor(hide);
    }

    /// Set the shape the pointer takes over this window.
    pub fn set_cursor_icon(&self, icon: winit::window::CursorIcon) {
        self.canvas.set_cursor_icon(icon);
    }

    /// Hide the window.
    pub fn hide(&mut self) {
        self.canvas.hide()
    }

    /// Show the window.
    pub fn show(&mut self) {
        self.canvas.show()
    }

    /// The state of a mouse button.
    pub fn get_mouse_button(&self, button: MouseButton) -> Action {
        self.canvas.get_mouse_button(button)
    }

    /// The state of a key.
    pub fn get_key(&self, key: Key) -> Action {
        self.canvas.get_key(key)
    }

    /// Copies the frame texture to the readback texture for later reading.
    pub fn copy_frame_to_readback(&self, frame: &wgpu::SurfaceTexture) {
        self.canvas.copy_frame_to_readback(frame)
    }

    /// Copies an arbitrary texture into the readback texture used by `read_pixels`.
    ///
    /// The source texture must have `COPY_SRC` usage and the same size and
    /// format as the surface.
    pub fn copy_texture_to_readback(&self, src: &wgpu::Texture) {
        self.canvas.copy_texture_to_readback(src)
    }

    /// Reads pixels from the readback texture into the provided buffer.
    /// Returns RGB data (3 bytes per pixel).
    pub fn read_pixels(&self, out: &mut Vec<u8>, x: usize, y: usize, width: usize, height: usize) {
        self.canvas.read_pixels(out, x, y, width, height)
    }

    /// Starts a non-blocking readback of the readback texture; complete it
    /// with [`Self::finish_read_pixels`]. See `WgpuCanvas::begin_read_pixels`.
    pub fn begin_read_pixels(&self, x: usize, y: usize, width: usize, height: usize) {
        self.canvas.begin_read_pixels(x, y, width, height)
    }

    /// Completes a readback started by [`Self::begin_read_pixels`], returning
    /// the captured `(width, height)`, or `None` when none is in flight.
    pub fn finish_read_pixels(&self, out: &mut Vec<u8>) -> Option<(u32, u32)> {
        self.canvas.finish_read_pixels(out)
    }
}
