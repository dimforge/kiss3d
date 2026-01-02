use std::sync::mpsc::Sender;

use crate::event::{Action, Key, MouseButton, WindowEvent};
use crate::window::WgpuCanvas;
use image::{GenericImage, Pixel};

/// The possible number of samples for multisample anti-aliasing.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NumSamples {
    /// Multisampling disabled.
    Zero = 0,
    /// One sample
    One = 1,
    /// Two samples
    Two = 2,
    /// Four samples
    Four = 4,
    /// Eight samples
    Eight = 8,
    /// Sixteen samples
    Sixteen = 16,
}

impl NumSamples {
    /// Create a `NumSamples` from a number.
    /// Returns `None` if `i` is invalid.
    pub fn from_u32(i: u32) -> Option<NumSamples> {
        match i {
            0 => Some(NumSamples::Zero),
            1 => Some(NumSamples::One),
            2 => Some(NumSamples::Two),
            4 => Some(NumSamples::Four),
            8 => Some(NumSamples::Eight),
            16 => Some(NumSamples::Sixteen),
            _ => None,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// Canvas options.
pub struct CanvasSetup {
    /// Is vsync enabled?
    pub vsync: bool,
    /// Number of AA samples.
    pub samples: NumSamples,
}

/// An abstract structure representing a window for native applications, and a canvas for web applications.
pub struct Canvas {
    canvas: WgpuCanvas,
}

impl Canvas {
    /// Open a new window, and initialize the wgpu context.
    pub async fn open(
        title: &str,
        hide: bool,
        width: u32,
        height: u32,
        canvas_setup: Option<CanvasSetup>,
        out_events: Sender<WindowEvent>,
    ) -> Self {
        Canvas {
            canvas: WgpuCanvas::open(title, hide, width, height, canvas_setup, out_events).await,
        }
    }

    /// Poll all events that occurred since the last call to this method.
    pub fn poll_events(&mut self) {
        self.canvas.poll_events()
    }

    /// Gets the current surface texture for rendering.
    pub fn get_current_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
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

    /// Set the cursor position.
    pub fn set_cursor_position(&self, x: f64, y: f64) {
        self.canvas.set_cursor_position(x, y);
    }

    /// Toggle the cursor visibility.
    pub fn hide_cursor(&self, hide: bool) {
        self.canvas.hide_cursor(hide);
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

    /// Reads pixels from the readback texture into the provided buffer.
    /// Returns RGB data (3 bytes per pixel).
    pub fn read_pixels(&self, out: &mut Vec<u8>, x: usize, y: usize, width: usize, height: usize) {
        self.canvas.read_pixels(out, x, y, width, height)
    }
}
