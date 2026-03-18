//! DRM Window for headless 3D rendering without a window manager.

use super::drm_canvas::DrmCanvas;
use super::drm_events::DrmEventManager;
use crate::camera::{Camera2d, Camera3d};
use crate::color::{Color, BLACK};
use crate::context::Context;
use crate::renderer::{PointRenderer2d, PointRenderer3d, PolylineRenderer2d, PolylineRenderer3d};
use crate::resource::{FramebufferManager, RenderTarget};
use crate::text::TextRenderer;
#[cfg(feature = "egui")]
use crate::window::egui_integration::EguiContext;
#[cfg(feature = "recording")]
use crate::window::recording::RecordingState;
use crate::window::window_cache::WindowCache;
use glamx::UVec2;
use std::cell::RefCell;
use std::error::Error;
use std::rc::Rc;

/// A window for headless 3D rendering using DRM (Direct Rendering Manager).
///
/// This window type allows rendering without a window manager, suitable for
/// console-only systems like Raspberry Pi setups. It reuses kiss3d's rendering
/// infrastructure but replaces the windowing system with offscreen buffers.
pub struct Window {
    pub(crate) event_manager: Rc<RefCell<DrmEventManager>>,
    // pub(crate) events: Rc<Receiver<WindowEvent>>,
    // pub(crate) unhandled_events: Rc<RefCell<Vec<WindowEvent>>>,
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
    pub(crate) canvas: DrmCanvas,
    #[cfg(feature = "recording")]
    pub(crate) recording: Option<RecordingState>,
}

impl Window {
    /// Creates a new DRM window for rendering.
    ///
    /// # Arguments
    /// * `device_path` - Path to the DRM device (e.g., "/dev/dri/card0")
    /// * `width` - Width of the render target in pixels
    /// * `height` - Height of the render target in pixels
    ///
    /// # Returns
    /// A new Window ready for rendering, or an error if initialization fails
    ///
    /// # Example
    /// ```no_run
    /// use kiss3d::prelude::*;
    ///
    /// #[kiss3d::main]
    /// async fn main() {
    ///     let mut window = Window::new("/dev/dri/card0", 1920, 1080)
    ///         .await
    ///         .expect("Failed to create DRM window");
    ///
    ///     let mut camera = OrbitCamera3d::default();
    ///     let mut scene = SceneNode3d::empty();
    ///
    ///     while window.render_3d(&mut scene, &mut camera).await {
    ///         // Your render loop code here
    ///     }
    /// }
    /// ```
    pub async fn try_new(device_path: &str) -> Result<Self, Box<dyn Error>> {
        // Create DRM canvas with display output (initializes wgpu headless + DRM/KMS)
        let canvas = DrmCanvas::new_with_display(device_path).await?;

        Self::new_from_canvas(canvas).await
    }

    /// Creates a new DRM window for offscreen-only rendering (no display output).
    ///
    /// This mode is useful for:
    /// - Screenshot/recording without a connected display
    /// - Server-side rendering
    /// - Testing without display hardware
    ///
    /// # Arguments
    /// * `width` - Width of the render target in pixels
    /// * `height` - Height of the render target in pixels
    ///
    /// # Returns
    /// A new Window ready for offscreen rendering (no display output)
    pub async fn new_offscreen(width: u32, height: u32) -> Result<Self, Box<dyn Error>> {
        log::info!("Creating DRM window (offscreen only): {}x{}", width, height);

        // Create DRM canvas in offscreen mode (no display initialization, no DRM device needed)
        let canvas = DrmCanvas::new(width, height).await?;

        Self::new_from_canvas(canvas).await
    }

    /// Internal helper to initialize Window from a DrmCanvas
    async fn new_from_canvas(canvas: DrmCanvas) -> Result<Self, Box<dyn Error>> {
        // Initialize window cache (material manager, mesh manager, texture manager)
        WindowCache::populate();

        // Create framebuffer manager
        let framebuffer_manager = FramebufferManager::new();
        let (width, height) = canvas.size();
        let post_process_render_target = framebuffer_manager.new_render_target(width, height, true);

        log::info!("DRM window initialized successfully");

        Ok(Self {
            event_manager: Rc::new(RefCell::new(DrmEventManager::new_headless())),
            // events: Rc::new(event_receive),
            // unhandled_events: Rc::new(RefCell::new(Vec::new())),
            ambient_intensity: 0.2,
            background: BLACK,
            polyline_renderer_2d: PolylineRenderer2d::new(),
            point_renderer_2d: PointRenderer2d::new(),
            point_renderer: PointRenderer3d::new(),
            polyline_renderer: PolylineRenderer3d::new(),
            text_renderer: TextRenderer::new(),
            framebuffer_manager,
            post_process_render_target,
            should_close: false,
            #[cfg(feature = "egui")]
            egui_context: EguiContext::new(),
            canvas,
            #[cfg(feature = "recording")]
            recording: None,
        })
    }

    pub async fn new(_title: &str) -> Self {
        let device_paths = [
            "/dev/dri/card0",
            "/dev/dri/card1",
            "/dev/dri/card2",
            "/dev/dri/renderD128",
            "/dev/dri/renderD129",
        ];
        for dev in device_paths {
            match Self::try_new(dev).await {
                Ok(window) => {
                    log::debug!("Created render target for device {dev}");
                    return window;
                }
                Err(e) => {
                    log::trace!("Could not created render target for device {dev}: {e}");
                    continue;
                }
            }
        }

        log::error!("Could not create any render target!");
        panic!("Could not create any render target!");
    }

    pub fn should_close(&self) -> bool {
        self.should_close
    }

    pub fn close(&mut self) {
        self.should_close = true;
    }

    // pub fn check_external_close_signal(&mut self) {
    //     // Check for SIGTERM, SIGINT, or a control file
    //     // For example: check if /tmp/kiss3d_stop exists
    //     self.should_close = std::path::Path::new("/tmp/kiss3d_stop").exists();
    // }

    /// Returns the width of the render target.
    #[inline]
    pub fn width(&self) -> u32 {
        self.canvas.size().0
    }

    /// Returns the height of the render target.
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

    /// Sets the background color for rendering.
    ///
    /// # Arguments
    /// * `color` - The background color to use
    #[inline]
    pub fn set_background_color(&mut self, color: Color) {
        self.background = color;
    }

    /// Sets the ambient light intensity for the scene.
    ///
    /// # Arguments
    /// * `ambient` - The ambient light intensity (typically 0.0 to 1.0)
    #[inline]
    pub fn set_ambient(&mut self, ambient: f32) {
        self.ambient_intensity = ambient;
    }

    /// Returns the current ambient lighting intensity.
    #[inline]
    pub fn ambient(&self) -> f32 {
        self.ambient_intensity
    }

    /// Enable event input from evdev devices
    #[cfg(target_os = "linux")]
    pub fn enable_evdev_input(&mut self, devices: Vec<String>) -> Result<(), std::io::Error> {
        let manager = DrmEventManager::new_with_evdev(devices)?;
        self.event_manager = Rc::new(RefCell::new(manager));
        log::info!("Evdev input enabled");
        Ok(())
    }

    /// Set a custom event source (for network control, GPIO buttons, etc.)
    pub fn set_custom_event_source(
        &mut self,
        receiver: std::sync::mpsc::Receiver<crate::event::WindowEvent>,
    ) {
        let manager = DrmEventManager::new_with_custom(receiver);
        self.event_manager = Rc::new(RefCell::new(manager));
        log::info!("Custom event source enabled");
    }

    /// Handle window events (DRM version)
    pub(crate) fn handle_events(
        &mut self,
        camera: &mut dyn Camera3d,
        camera_2d: &mut dyn Camera2d,
    ) {
        // Poll for new events
        self.event_manager.borrow_mut().poll_events();

        // Process all accumulated events
        let events: Vec<_> = self.event_manager.borrow_mut().drain_events().collect();

        for event in events {
            self.handle_event(camera, camera_2d, &event);
        }
    }

    pub(crate) fn handle_event(
        &mut self,
        camera: &mut dyn Camera3d,
        camera_2d: &mut dyn Camera2d,
        event: &crate::event::WindowEvent,
    ) {
        use crate::event::{Action, Key, WindowEvent};

        match *event {
            WindowEvent::Key(Key::Escape, Action::Release, _) | WindowEvent::Close => {
                self.close();
            }
            _ => {}
        }

        // Feed events to egui if enabled
        #[cfg(feature = "egui")]
        {
            // TODO: implement feed_egui_event for DRM
        }

        // Create a wrapper for camera compatibility
        let wrapper = super::DrmCanvasWrapper::new(&self.canvas);
        let canvas_ref: &crate::window::Canvas = unsafe { std::mem::transmute(&wrapper) };

        camera.handle_event(canvas_ref, event);
        camera_2d.handle_event(canvas_ref, event);
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        log::info!("Dropping DRM window");

        // Only clean up GPU resources when the last window is dropped
        let is_last_window = Context::decrement_window_count();

        if is_last_window {
            log::info!("Last DRM window dropped, cleaning up resources");

            // Clear resource managers
            WindowCache::reset();

            // Clear the wgpu context
            Context::reset();
        }
    }
}
