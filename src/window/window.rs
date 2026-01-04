//! The kiss3d window.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

use crate::color::{Color, BLACK};
use crate::context::Context;
use crate::event::WindowEvent;
use crate::renderer::{PointRenderer2d, PointRenderer3d, PolylineRenderer2d, PolylineRenderer3d};
use crate::resource::{
    FramebufferManager, MaterialManager2d, MeshManager2d, RenderTarget, Texture, TextureManager,
};
use crate::text::TextRenderer;
use crate::window::canvas::CanvasSetup;
use crate::window::Canvas;
use glamx::UVec2;
use image::{GenericImage, Pixel};

#[cfg(feature = "egui")]
pub(super) use super::egui_integration::EguiContext;
#[cfg(feature = "recording")]
pub(super) use super::recording::RecordingState;
use super::window_cache::WindowCache;

pub(super) static DEFAULT_WIDTH: u32 = 800u32;
pub(super) static DEFAULT_HEIGHT: u32 = 600u32;

/// Structure representing a window and a 3D scene.
///
/// This is the main interface with the 3d engine.
pub struct Window {
    pub(super) events: Rc<Receiver<WindowEvent>>,
    pub(super) unhandled_events: Rc<RefCell<Vec<WindowEvent>>>,
    pub(super) ambient_intensity: f32,
    pub(super) background: Color,
    pub(super) polyline_renderer_2d: PolylineRenderer2d,
    pub(super) point_renderer_2d: PointRenderer2d,
    pub(super) point_renderer: PointRenderer3d,
    pub(super) polyline_renderer: PolylineRenderer3d,
    pub(super) text_renderer: TextRenderer,
    #[allow(dead_code)]
    pub(super) framebuffer_manager: FramebufferManager,
    pub(super) post_process_render_target: RenderTarget,
    pub(super) should_close: bool,
    #[cfg(feature = "egui")]
    pub(super) egui_context: EguiContext,
    pub(super) canvas: Canvas,
    #[cfg(feature = "recording")]
    pub(super) recording: Option<RecordingState>,
}

impl Window {
    /// Indicates whether this window should be closed.
    #[inline]
    pub fn should_close(&self) -> bool {
        self.should_close
    }

    /// The window width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.canvas.size().0
    }

    /// The window height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.canvas.size().1
    }

    /// The size of the window.
    #[inline]
    pub fn size(&self) -> UVec2 {
        let (w, h) = self.canvas.size();
        UVec2::new(w, h)
    }

    /// Gets a reference to the underlying canvas.
    ///
    /// This provides access to low-level rendering features like:
    /// - Getting the current surface texture for custom rendering
    /// - Getting the depth texture view
    /// - Presenting frames manually
    #[inline]
    pub fn canvas(&self) -> &Canvas {
        &self.canvas
    }

    /// Gets a mutable reference to the underlying canvas.
    #[inline]
    pub fn canvas_mut(&mut self) -> &mut Canvas {
        &mut self.canvas
    }

    /// Sets the window title.
    ///
    /// # Arguments
    /// * `title` - The new title for the window
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// let mut window = Window::new("Initial Title").await;
    /// window.set_title("New Title");
    /// # }
    /// ```
    pub fn set_title(&mut self, title: &str) {
        self.canvas.set_title(title)
    }

    /// Set the window icon. On wasm this does nothing.
    ///
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// window.set_icon(image::open("foo.ico").unwrap());
    /// # }
    /// ```
    pub fn set_icon(&mut self, icon: impl GenericImage<Pixel = impl Pixel<Subpixel = u8>>) {
        self.canvas.set_icon(icon)
    }

    /// Sets the cursor grabbing behaviour.
    ///
    /// If cursor grabbing is enabled, the cursor is prevented from leaving the window.
    ///
    /// # Arguments
    /// * `grab` - `true` to enable cursor grabbing, `false` to disable it
    ///
    /// # Platform-specific
    /// Does nothing on web platforms.
    pub fn set_cursor_grab(&self, grab: bool) {
        self.canvas.set_cursor_grab(grab);
    }

    /// Sets the cursor position in window coordinates.
    ///
    /// # Arguments
    /// * `x` - The x-coordinate in pixels from the left edge of the window
    /// * `y` - The y-coordinate in pixels from the top edge of the window
    #[inline]
    pub fn set_cursor_position(&self, x: f64, y: f64) {
        self.canvas.set_cursor_position(x, y);
    }

    /// Controls the cursor visibility.
    ///
    /// # Arguments
    /// * `hide` - `true` to hide the cursor, `false` to show it
    #[inline]
    pub fn hide_cursor(&self, hide: bool) {
        self.canvas.hide_cursor(hide);
    }

    /// Closes the window.
    ///
    /// After calling this method, [`render()`](Self::render) will return `false` on the next frame,
    /// allowing the render loop to exit gracefully.
    #[inline]
    pub fn close(&mut self) {
        self.should_close = true;
    }

    /// Hides the window without closing it.
    ///
    /// Use [`show()`](Self::show) to make it visible again.
    /// The window continues to exist and can be shown again later.
    #[inline]
    pub fn hide(&mut self) {
        self.canvas.hide()
    }

    /// Makes the window visible.
    ///
    /// Use [`hide()`](Self::hide) to hide it again.
    #[inline]
    pub fn show(&mut self) {
        self.canvas.show()
    }

    /// Sets the background color for the window.
    ///
    /// # Arguments
    /// * `r` - Red component (0.0 to 1.0)
    /// * `g` - Green component (0.0 to 1.0)
    /// * `b` - Blue component (0.0 to 1.0)
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

    /// Loads a texture from a file and returns a reference to it.
    ///
    /// The texture is managed by the global texture manager and will be reused
    /// if loaded again with the same name.
    ///
    /// # Arguments
    /// * `path` - Path to the texture file
    /// * `name` - A unique name to identify this texture
    ///
    /// # Returns
    /// A reference-counted texture that can be applied to scene objects
    pub fn add_texture(&mut self, path: &Path, name: &str) -> Arc<Texture> {
        TextureManager::get_global_manager(|tm| tm.add(path, name))
    }

    /// Returns the DPI scale factor of the screen.
    ///
    /// This is the ratio between physical pixels and logical pixels.
    /// On high-DPI displays (like Retina displays), this will be greater than 1.0.
    ///
    /// # Returns
    /// The scale factor (e.g., 1.0 for standard displays, 2.0 for Retina displays)
    pub fn scale_factor(&self) -> f64 {
        self.canvas.scale_factor()
    }

    /// Sets the ambient light intensity for the scene.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # use kiss3d::light::Light;
    /// # use glamx::Vec3;
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
    pub fn set_ambient(&mut self, ambient: f32) {
        self.ambient_intensity = ambient;
    }

    /// Returns the current ambient lighting intensity.
    pub fn ambient(&self) -> f32 {
        self.ambient_intensity
    }

    /// Creates a new hidden window.
    ///
    /// The window is created but not displayed. Use [`show()`](Self::show) to make it visible.
    /// The default size is 800x600 pixels.
    ///
    /// # Arguments
    /// * `title` - The window title
    ///
    /// # Returns
    /// A new `Window` instance
    pub async fn new_hidden(title: &str) -> Window {
        Window::do_new(title, true, DEFAULT_WIDTH, DEFAULT_HEIGHT, None).await
    }

    /// Creates a new visible window with default settings.
    ///
    /// The window is created and immediately visible with a default size of 800x600 pixels.
    /// Use this in combination with the `#[kiss3d::main]` macro for cross-platform rendering.
    ///
    /// # Arguments
    /// * `title` - The window title
    ///
    /// # Returns
    /// A new `Window` instance
    ///
    /// # Example
    /// ```no_run
    /// use kiss3d::prelude::*;
    ///
    /// #[kiss3d::main]
    /// async fn main() {
    ///     let mut window = Window::new("My Application").await;
    ///     let mut camera = OrbitCamera3d::default();
    ///     let mut scene = SceneNode3d::empty();
    ///
    ///     while window.render_3d(&mut scene, &mut camera).await {
    ///         // Your render loop code here
    ///     }
    /// }
    /// ```
    pub async fn new(title: &str) -> Window {
        Window::do_new(title, false, DEFAULT_WIDTH, DEFAULT_HEIGHT, None).await
    }

    /// Creates a new window with custom dimensions.
    ///
    /// # Arguments
    /// * `title` - The window title
    /// * `width` - The window width in pixels
    /// * `height` - The window height in pixels
    ///
    /// # Returns
    /// A new `Window` instance
    pub async fn new_with_size(title: &str, width: u32, height: u32) -> Window {
        Window::do_new(title, false, width, height, None).await
    }

    /// Creates a new window with custom setup options.
    ///
    /// This allows fine-grained control over window creation, including VSync and anti-aliasing settings.
    ///
    /// # Arguments
    /// * `title` - The window title
    /// * `width` - The window width in pixels
    /// * `height` - The window height in pixels
    /// * `setup` - A `CanvasSetup` struct containing the window configuration
    ///
    /// # Returns
    /// A new `Window` instance
    pub async fn new_with_setup(
        title: &str,
        width: u32,
        height: u32,
        setup: CanvasSetup,
    ) -> Window {
        Window::do_new(title, false, width, height, Some(setup)).await
    }

    // TODO: make this pub?
    async fn do_new(
        title: &str,
        hide: bool,
        width: u32,
        height: u32,
        setup: Option<CanvasSetup>,
    ) -> Window {
        let (event_send, event_receive) = mpsc::channel();
        let canvas = Canvas::open(title, hide, width, height, setup, event_send).await;

        // Track window count for proper cleanup
        Context::increment_window_count();

        WindowCache::populate();

        let framebuffer_manager = FramebufferManager::new();
        let mut usr_window = Window {
            should_close: false,
            canvas,
            events: Rc::new(event_receive),
            unhandled_events: Rc::new(RefCell::new(Vec::new())),
            ambient_intensity: 0.2,
            background: BLACK,
            polyline_renderer_2d: PolylineRenderer2d::new(),
            point_renderer_2d: PointRenderer2d::new(),
            point_renderer: PointRenderer3d::new(),
            polyline_renderer: PolylineRenderer3d::new(),
            text_renderer: TextRenderer::new(),
            #[cfg(feature = "egui")]
            egui_context: EguiContext::new(),
            post_process_render_target: framebuffer_manager.new_render_target(width, height, true),
            framebuffer_manager,
            #[cfg(feature = "recording")]
            recording: None,
        };

        if hide {
            usr_window.canvas.hide()
        }

        usr_window
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        // Only clean up GPU resources when the last window is dropped.
        // This prevents TLS access order issues with wgpu internals that can cause
        // panics during thread cleanup.
        let is_last_window = Context::decrement_window_count();

        if is_last_window {
            // The order matters: clear caches first (which hold references to GPU resources),
            // then clear the Context (which holds the wgpu Device/Queue/Instance).

            // Clear 3D resource managers
            WindowCache::reset();

            // Clear 2D resource managers
            MeshManager2d::reset_global_manager();
            MaterialManager2d::reset_global_manager();

            // Finally, clear the wgpu context itself
            Context::reset();
        }
    }
}
