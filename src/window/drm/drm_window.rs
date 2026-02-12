//! DRM Window for headless 3D rendering without a window manager.

use crate::camera::Camera3d;
use crate::color::{Color, BLACK};
use crate::context::Context;
use crate::renderer::{PointRenderer2d, PointRenderer3d, PolylineRenderer2d, PolylineRenderer3d};
use crate::resource::{FramebufferManager, RenderTarget};
use crate::scene::SceneNode3d;
use crate::text::TextRenderer;
use crate::window::window_cache::WindowCache;
use image::{imageops, ImageBuffer, Rgb};
use std::error::Error;

use super::drm_canvas::DrmCanvas;

/// A window for headless 3D rendering using DRM (Direct Rendering Manager).
///
/// This window type allows rendering without a window manager, suitable for
/// console-only systems like Raspberry Pi setups. It reuses kiss3d's rendering
/// infrastructure but replaces the windowing system with offscreen buffers.
pub struct DRMWindow {
    /// The DRM canvas backend
    drm_canvas: DrmCanvas,
    /// Renderer for 3D polylines (lines in 3D space)
    polyline_renderer: PolylineRenderer3d,
    /// Renderer for 3D points
    point_renderer: PointRenderer3d,
    /// Renderer for 2D polylines
    polyline_renderer_2d: PolylineRenderer2d,
    /// Renderer for 2D points
    point_renderer_2d: PointRenderer2d,
    /// Text renderer for 2D text overlays
    text_renderer: TextRenderer,
    /// Framebuffer manager for offscreen rendering
    framebuffer_manager: FramebufferManager,
    /// Render target for post-processing effects
    post_process_render_target: RenderTarget,
    /// Ambient light intensity (0.0 to 1.0)
    pub(crate) ambient_intensity: f32,
    /// Background clear color
    pub(crate) background: Color,
}

impl DRMWindow {
    /// Creates a new DRM window for headless rendering.
    ///
    /// # Arguments
    /// * `device_path` - Path to the DRM device (e.g., "/dev/dri/card0")
    /// * `width` - Width of the render target in pixels
    /// * `height` - Height of the render target in pixels
    ///
    /// # Returns
    /// A new DRMWindow ready for rendering, or an error if initialization fails
    ///
    /// # Example
    /// ```no_run
    /// use kiss3d::prelude::*;
    ///
    /// #[kiss3d::main]
    /// async fn main() {
    ///     let mut window = DRMWindow::new("/dev/dri/card0", 1920, 1080)
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
    pub async fn new(
        device_path: &str,
        width: u32,
        height: u32,
    ) -> Result<Self, Box<dyn Error>> {
        log::info!("Creating DRM window: {}x{}", width, height);

        // Create DRM canvas (initializes wgpu headless)
        let drm_canvas = DrmCanvas::new(device_path, width, height).await?;

        // Initialize window cache (material manager, mesh manager, texture manager)
        WindowCache::populate();

        // Create framebuffer manager
        let framebuffer_manager = FramebufferManager::new();

        // Create post-processing render target
        let post_process_render_target =
            framebuffer_manager.new_render_target(width, height, true);

        // Initialize all renderers
        let polyline_renderer_2d = PolylineRenderer2d::new();
        let point_renderer_2d = PointRenderer2d::new();
        let point_renderer = PointRenderer3d::new();
        let polyline_renderer = PolylineRenderer3d::new();
        let text_renderer = TextRenderer::new();

        log::info!("DRM window created successfully");

        Ok(Self {
            drm_canvas,
            polyline_renderer,
            point_renderer,
            polyline_renderer_2d,
            point_renderer_2d,
            text_renderer,
            framebuffer_manager,
            post_process_render_target,
            ambient_intensity: 0.2,
            background: BLACK,
        })
    }

    /// Returns the width of the render target.
    #[inline]
    pub fn width(&self) -> u32 {
        self.drm_canvas.size().0
    }

    /// Returns the height of the render target.
    #[inline]
    pub fn height(&self) -> u32 {
        self.drm_canvas.size().1
    }

    /// Returns the dimensions of the render target.
    #[inline]
    pub fn size(&self) -> (u32, u32) {
        self.drm_canvas.size()
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

    /// Renders a 3D scene to the offscreen buffer.
    ///
    /// This is the main rendering method for DRM windows.
    /// It should be called once per frame in your render loop.
    ///
    /// # Arguments
    /// * `scene` - The 3D scene graph to render
    /// * `camera` - The camera used for viewing the scene
    ///
    /// # Returns
    /// Always returns `true` (no window close events in headless mode)
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::prelude::*;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = DRMWindow::new("/dev/dri/card0", 1920, 1080).await.unwrap();
    /// # let mut camera = OrbitCamera3d::default();
    /// # let mut scene = SceneNode3d::empty();
    /// while window.render_3d(&mut scene, &mut camera).await {
    ///     // Per-frame updates here
    /// }
    /// # }
    /// ```
    pub async fn render_3d(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut impl Camera3d,
    ) -> bool {
        use crate::context::Context;
        use crate::event::WindowEvent;
        use crate::resource::RenderContext2dEncoder;
        use crate::window::Canvas;

        let w = self.width();
        let h = self.height();

        // Create a canvas wrapper for camera compatibility
        let canvas_wrapper = super::DrmCanvasWrapper::new(&self.drm_canvas);
        
        // SAFETY: Cameras need a &Canvas reference, but in headless mode most
        // don't actually use it. This transmute is safe because:
        // 1. The camera only reads from the canvas (no writes)
        // 2. The lifetime is constrained to this function scope
        // 3. DrmCanvasWrapper provides compatible methods for camera calls
        let canvas_ref: &Canvas = unsafe { std::mem::transmute(&canvas_wrapper) };

        // Update camera state  
        camera.handle_event(canvas_ref, &WindowEvent::FramebufferSize(w, h));
        camera.update(canvas_ref);

        // Get the surface texture
        let frame = match self.drm_canvas.get_current_texture() {
            Ok(frame) => frame,
            Err(e) => {
                eprintln!("Failed to acquire DRM surface texture: {:?}", e);
                return true; // Continue rendering in headless mode
            }
        };
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let ctxt = Context::get();
        let mut encoder = ctxt.create_command_encoder(Some("drm_frame_encoder"));

        // Resize post-process render target if needed
        let surface_format = self.drm_canvas.surface_format();
        self.post_process_render_target
            .resize(w, h, surface_format);

        // Render directly to the frame (no post-processing for now)
        let color_view = &frame_view;
        let depth_view = self.drm_canvas.depth_view();

        // Use shared rendering pipeline
        crate::window::rendering::render_frame_3d(
            &mut encoder,
            color_view,
            depth_view,
            surface_format,
            self.drm_canvas.sample_count(),
            w,
            h,
            self.background,
            self.ambient_intensity,
            scene,
            camera,
            canvas_ref,
            &mut self.point_renderer,
            &mut self.polyline_renderer,
            &mut None, // No custom renderer for now
        );

        // Render text overlay
        {
            let mut context_2d_encoder = RenderContext2dEncoder {
                encoder: &mut encoder,
                color_view,
                surface_format,
                sample_count: self.drm_canvas.sample_count(),
                viewport_width: w,
                viewport_height: h,
            };
            
            self.text_renderer.render(w as f32, h as f32, &mut context_2d_encoder);
        }

        // Submit commands
        ctxt.submit(std::iter::once(encoder.finish()));

        // Present the frame
        self.drm_canvas.present();

        true // Always continue in headless mode
    }

    /// Captures the current framebuffer as raw RGB pixel data.
    ///
    /// Reads all pixels currently rendered in the offscreen buffer into a buffer.
    /// The buffer is automatically resized to fit the frame dimensions.
    /// Pixels are stored in RGB format (3 bytes per pixel), row by row from bottom to top.
    ///
    /// # Arguments
    /// * `out` - The output buffer. It will be resized to width × height × 3 bytes.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::DRMWindow;
    /// # async fn example() {
    /// # let window = DRMWindow::new(800, 600).await.unwrap();
    /// let mut pixels = Vec::new();
    /// window.snap(&mut pixels);
    /// // pixels now contains RGB data
    /// # }
    /// ```
    pub fn snap(&self, out: &mut Vec<u8>) {
        let (width, height) = self.drm_canvas.size();
        self.snap_rect(out, 0, 0, width as usize, height as usize)
    }

    /// Captures a rectangular region of the framebuffer as raw RGB pixel data.
    ///
    /// Reads a specific rectangular region of pixels from the offscreen buffer.
    /// Pixels are stored in RGB format (3 bytes per pixel).
    ///
    /// # Arguments
    /// * `out` - The output buffer. It will be resized to width × height × 3 bytes.
    /// * `x` - The x-coordinate of the rectangle's bottom-left corner
    /// * `y` - The y-coordinate of the rectangle's bottom-left corner
    /// * `width` - The width of the rectangle in pixels
    /// * `height` - The height of the rectangle in pixels
    pub fn snap_rect(&self, out: &mut Vec<u8>, x: usize, y: usize, width: usize, height: usize) {
        self.drm_canvas.read_pixels(out, x, y, width, height);
    }

    /// Captures the current framebuffer as an image.
    ///
    /// Returns an `ImageBuffer` containing the current offscreen rendered content.
    /// The image is automatically flipped vertically to match the expected orientation
    /// (bottom-left origin is converted to top-left).
    ///
    /// # Returns
    /// An `ImageBuffer<Rgb<u8>, Vec<u8>>` containing the frame pixels
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::DRMWindow;
    /// # async fn example() {
    /// # let window = DRMWindow::new(800, 600).await.unwrap();
    /// let image = window.snap_image();
    /// image.save("frame.png").unwrap();
    /// # }
    /// ```
    pub fn snap_image(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (width, height) = self.drm_canvas.size();
        let mut buf = Vec::new();
        self.snap(&mut buf);
        let img_opt = ImageBuffer::from_vec(width, height, buf);
        let img = img_opt.expect("Buffer created from DRM window was not big enough for image.");
        imageops::flip_vertical(&img)
    }
}

impl Drop for DRMWindow {
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