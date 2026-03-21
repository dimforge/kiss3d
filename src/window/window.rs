//! The kiss3d window (winit-backed). Only compiled when the `drm` feature is off.

#![cfg(not(feature = "drm"))]

use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;

use crate::color::BLACK;
use crate::context::Context;
use crate::resource::{MaterialManager2d, MeshManager2d, Texture, TextureManager};
use crate::window::canvas::CanvasSetup;
use crate::window::Canvas;
use image::{GenericImage, Pixel};

use super::window_cache::WindowCache;

pub(crate) static DEFAULT_WIDTH: u32 = 800u32;
pub(crate) static DEFAULT_HEIGHT: u32 = 600u32;

// Window struct is defined in window_common.rs.
use super::window_common::Window;

impl Window {
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
    /// This allows fine-grained control over window creation, including VSync and
    /// anti-aliasing settings.
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

        let framebuffer_manager = crate::resource::FramebufferManager::new();
        let mut usr_window = Window {
            should_close: false,
            canvas,
            events: std::rc::Rc::new(event_receive),
            unhandled_events: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
            ambient_intensity: 0.2,
            background: BLACK,
            polyline_renderer_2d: crate::renderer::PolylineRenderer2d::new(),
            point_renderer_2d: crate::renderer::PointRenderer2d::new(),
            point_renderer: crate::renderer::PointRenderer3d::new(),
            polyline_renderer: crate::renderer::PolylineRenderer3d::new(),
            text_renderer: crate::text::TextRenderer::new(),
            #[cfg(feature = "egui")]
            egui_context: super::egui_integration::EguiContext::new(),
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
