//! The kiss3d window.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

use crate::builtin::ShadowMapper;
use crate::camera::{Camera3d, FixedView2d};
use crate::color::{Color, BLACK};
use crate::context::Context;
use crate::event::{Key, Modifiers, WindowEvent};
use crate::post_processing::{HdrPipeline, HdrSettings, Tonemap};
use crate::renderer::timings::GpuTimer;
use crate::renderer::{
    PointRenderer2d, PointRenderer3d, PolylineRenderer2d, PolylineRenderer3d, RayTracer,
    RenderTimings,
};
use crate::resource::{
    FramebufferManager, MaterialManager2d, MeshManager2d, RenderTarget, Texture, TextureManager,
};
use crate::scene::SceneNode3d;
use crate::text::TextRenderer;
use crate::window::canvas::CanvasSetup;
use crate::window::{Canvas, NumSamples};
use glamx::UVec2;
use image::{GenericImage, Pixel};
use winit::dpi::LogicalSize;
use winit::window::WindowAttributes;

#[cfg(feature = "egui")]
pub(super) use super::egui_integration::EguiContext;
#[cfg(feature = "recording")]
pub(super) use super::recording::RecordingState;
use super::window_cache::WindowCache;

pub(super) static DEFAULT_WIDTH: u32 = 800u32;
pub(super) static DEFAULT_HEIGHT: u32 = 600u32;

/// Default per-layer resolution of the rasterizer shadow atlas. 4096 keeps shadows
/// crisp across the cascades; lower it with [`Window::set_shadow_resolution`] to
/// trade sharpness for memory (the atlas is `resolution² × MAX_SHADOW_VIEWS`).
pub(super) static DEFAULT_SHADOW_RESOLUTION: u32 = 4096u32;

/// Structure representing a window and a 3D scene.
///
/// This is the main interface with the 3d engine.
pub struct Window {
    pub(super) events: Rc<Receiver<WindowEvent>>,
    pub(super) unhandled_events: Rc<RefCell<Vec<WindowEvent>>>,
    pub(super) ambient_intensity: f32,
    pub(super) ambient_color: Color,
    pub(super) fog: crate::light::Fog,
    pub(super) background: Color,
    pub(super) polyline_renderer_2d: PolylineRenderer2d,
    pub(super) point_renderer_2d: PointRenderer2d,
    pub(super) point_renderer: PointRenderer3d,
    pub(super) polyline_renderer: PolylineRenderer3d,
    pub(super) text_renderer: TextRenderer,
    pub(super) framebuffer_manager: FramebufferManager,
    /// Real-time shadow mapper for the rasterization pipeline.
    pub(super) shadow_mapper: ShadowMapper,
    /// HDR film + tonemap + bloom resolve stage for the rasterizer. The scene is
    /// rendered into its `Rgba16Float` target, then tonemapped into the LDR
    /// swapchain/offscreen output. See [`HdrPipeline`].
    pub(super) hdr: HdrPipeline,
    pub(super) post_process_render_target: RenderTarget,
    /// Offscreen render target used when the window is hidden, so `snap` and
    /// recording work without a presentable surface. Created on first use.
    pub(super) offscreen_output_target: Option<RenderTarget>,
    /// Renderer for auxiliary outputs (depth, normals, segmentation). Created
    /// on first use of an AOV-producing method.
    pub(super) aov_renderer: Option<crate::builtin::AovRenderer>,
    /// Whether the window is hidden. Hidden windows render offscreen.
    pub(super) hidden: bool,
    pub(super) should_close: bool,
    /// `true` until the first surface texture has been successfully acquired.
    /// While set, frame acquisition retries (pumping window events) instead of
    /// skipping, so a freshly created window reliably renders its first frame.
    pub(super) first_frame: bool,
    pub(super) close_key: Option<Key>,
    pub(super) close_modifiers: Option<Modifiers>,
    /// Per-step timings of the most recently rendered frame, for the active
    /// renderer. `None` until the first frame. See [`Window::render_timings`].
    pub(super) last_timings: Option<RenderTimings>,
    /// GPU timestamp-query timer (disabled if the device lacks `TIMESTAMP_QUERY`).
    pub(super) gpu_timer: GpuTimer,
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

    /// The current number of MSAA samples (`1` means multisampling is disabled).
    #[inline]
    pub fn samples(&self) -> u32 {
        self.canvas.sample_count()
    }

    /// Sets the number of MSAA samples used for rendering, recreating the render
    /// targets to match. The change takes effect on the next rendered frame.
    ///
    /// This is the runtime equivalent of [`CanvasSetup::samples`]; it lets you toggle
    /// or change anti-aliasing after the window has been created.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::{Window, NumSamples};
    /// # async fn f(window: &mut Window) {
    /// window.set_samples(NumSamples::Four);
    /// # }
    /// ```
    #[inline]
    pub fn set_samples(&mut self, samples: NumSamples) {
        self.canvas.set_samples(samples);
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

    /// Timings of the most recently rendered frame.
    ///
    /// Returns `None` until the first frame has been rendered. The timings are
    /// for whichever renderer ran last (rasterizer or path tracer): per-pass GPU
    /// execution times from timestamp queries (when the device supports them),
    /// plus CPU wall-clock for the submit/present calls and the whole frame. See
    /// [`RenderTimings`]. The built-in inspector displays them.
    #[inline]
    pub fn render_timings(&self) -> Option<&RenderTimings> {
        self.last_timings.as_ref()
    }

    /// Renders one frame of a 3D scene with the GPU path tracer.
    ///
    /// This is the ray-traced counterpart of [`render_3d`](Self::render_3d). It
    /// bypasses the rasterizer and instead path-traces the scene, progressively
    /// accumulating samples for a photorealistic image. Keep the same
    /// [`RayTracer`] across frames so accumulation can converge; it restarts
    /// automatically when the camera moves, the window is resized, or the scene
    /// changes.
    ///
    /// If the path tracer is disabled (see [`RayTracer::set_enabled`]), this
    /// renders the scene with the rasterizer instead, so the same render call
    /// can switch between the two renderers without restructuring the loop.
    ///
    /// # Example
    /// ```no_run
    /// use kiss3d::prelude::*;
    /// use kiss3d::renderer::RayTracer;
    ///
    /// #[kiss3d::main]
    /// async fn main() {
    ///     let mut window = Window::new("Ray tracing").await;
    ///     let mut camera = OrbitCamera3d::default();
    ///     let mut scene = SceneNode3d::empty();
    ///     let mut raytracer = RayTracer::new();
    ///
    ///     while window.raytrace_3d(&mut scene, &mut camera, &mut raytracer).await {}
    /// }
    /// ```
    pub async fn raytrace_3d(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut impl Camera3d,
        raytracer: &mut RayTracer,
    ) -> bool {
        // When the path tracer is disabled, render the scene with the rasterizer
        // instead (`render_3d` handles events on its own).
        if !raytracer.enabled() {
            return self.render_3d(scene, camera).await;
        }

        let mut default_cam2 = FixedView2d::default();
        self.handle_events(camera, &mut default_cam2);
        self.raytrace_3d_frame(scene, camera, raytracer).await
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
        self.hidden = true;
        self.canvas.hide()
    }

    /// Makes the window visible.
    ///
    /// Use [`hide()`](Self::hide) to hide it again.
    #[inline]
    pub fn show(&mut self) {
        self.hidden = false;
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

    /// Sets the global ambient light color.
    ///
    /// The ambient term added to every surface is `ambient_color * ambient *
    /// albedo * ao`, so the color tints the fill light while
    /// [`set_ambient`](Self::set_ambient) controls its brightness. Defaults to
    /// white.
    pub fn set_ambient_color(&mut self, color: Color) {
        self.ambient_color = color;
    }

    /// Returns the current ambient light color.
    pub fn ambient_color(&self) -> Color {
        self.ambient_color
    }

    /// Sets the distance fog applied to the rasterized scene.
    ///
    /// Pass a [`Fog`](crate::light::Fog) describing the falloff curve and color,
    /// or [`Fog::default()`] (mode [`FogMode::Off`](crate::light::FogMode::Off))
    /// to disable fog. Fog blends shaded fragments toward the fog color by their
    /// view-space distance from the camera.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::prelude::*;
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// window.set_fog(Fog::exponential(Color::new(0.6, 0.7, 0.8, 1.0), 0.02));
    /// # }
    /// ```
    pub fn set_fog(&mut self, fog: crate::light::Fog) {
        self.fog = fog;
    }

    /// Returns the current distance fog settings.
    pub fn fog(&self) -> crate::light::Fog {
        self.fog
    }

    /// Mutable access to the distance fog settings.
    pub fn fog_mut(&mut self) -> &mut crate::light::Fog {
        &mut self.fog
    }

    /// Enables or disables real-time shadow mapping for the rasterizer.
    ///
    /// Shadows are enabled by default. When disabled, no shadow pre-pass runs and
    /// every light illuminates surfaces as if unobstructed. This has no effect on
    /// the path tracer, which always computes ray-traced shadows.
    pub fn set_shadows_enabled(&mut self, enabled: bool) {
        self.shadow_mapper.set_enabled(enabled);
    }

    /// Returns whether real-time shadow mapping is enabled for the rasterizer.
    pub fn shadows_enabled(&self) -> bool {
        self.shadow_mapper.is_enabled()
    }

    /// Sets the per-layer resolution of the shadow atlas (square), reallocating it.
    ///
    /// Higher values yield crisper shadows at the cost of memory and fill rate.
    /// The default is 1024.
    pub fn set_shadow_resolution(&mut self, resolution: u32) {
        self.shadow_mapper.set_resolution(resolution);
    }

    /// Returns the current per-layer shadow atlas resolution.
    pub fn shadow_resolution(&self) -> u32 {
        self.shadow_mapper.resolution()
    }

    /// Sets the rasterizer shadow-edge softness (PCF blur).
    ///
    /// `1.0` (the default) is the standard penumbra; larger values blur the
    /// shadow edges more, `0.0` gives hard edges. Has no effect on the path
    /// tracer, whose shadow softness comes from each light's `radius`.
    pub fn set_shadow_softness(&mut self, softness: f32) {
        self.shadow_mapper.set_softness(softness);
    }

    /// Returns the current rasterizer shadow-edge softness (PCF blur).
    pub fn shadow_softness(&self) -> f32 {
        self.shadow_mapper.softness()
    }

    /// The current HDR finishing settings (exposure, tonemap operator, bloom).
    ///
    /// The rasterizer renders into an HDR film and resolves it with these
    /// settings; see [`HdrSettings`] and [`HdrPipeline`].
    pub fn hdr_settings(&self) -> &HdrSettings {
        self.hdr.settings()
    }

    /// Mutable access to the HDR finishing settings.
    ///
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # use kiss3d::post_processing::Tonemap;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// let s = window.hdr_settings_mut();
    /// s.exposure = 1.5;
    /// s.tonemap = Tonemap::Aces;
    /// s.bloom_enabled = true;
    /// # }
    /// ```
    pub fn hdr_settings_mut(&mut self) -> &mut HdrSettings {
        self.hdr.settings_mut()
    }

    /// Sets the exposure multiplier applied before tonemapping (`1.0` is neutral).
    pub fn set_exposure(&mut self, exposure: f32) {
        self.hdr.settings_mut().exposure = exposure;
    }

    /// Sets the exposure from a physically-based [`Exposure`](crate::camera::Exposure).
    ///
    /// Applies to both the rasterizer and the path tracer (they share the HDR
    /// resolve exposure).
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::prelude::*;
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// // f/8, 1/125 s, ISO 100
    /// window.set_exposure_value(Exposure::from_physical(8.0, 1.0 / 125.0, 100.0));
    /// # }
    /// ```
    pub fn set_exposure_value(&mut self, exposure: crate::camera::Exposure) {
        self.hdr.settings_mut().exposure = exposure.exposure();
    }

    /// Selects the tonemapping operator used by the HDR resolve pass.
    pub fn set_tonemap(&mut self, tonemap: Tonemap) {
        self.hdr.settings_mut().tonemap = tonemap;
    }

    /// Enables or disables bloom.
    pub fn set_bloom_enabled(&mut self, enabled: bool) {
        self.hdr.settings_mut().bloom_enabled = enabled;
    }

    /// Sets the bloom brightness threshold and additive intensity.
    pub fn set_bloom(&mut self, threshold: f32, intensity: f32) {
        let s = self.hdr.settings_mut();
        s.bloom_threshold = threshold;
        s.bloom_intensity = intensity;
    }

    /// Rebinds the key to close the window.
    /// Set to None to disable.
    pub fn rebind_close_key(&mut self, new_close_key: Option<Key>) {
        self.close_key = new_close_key;
    }

    /// Rebinds the modifiers to close the window.
    /// Set to None make it work with any modifiers.
    pub fn rebind_close_modifiers(&mut self, new_close_modifiers: Option<Modifiers>) {
        self.close_modifiers = new_close_modifiers;
    }

    /// Returns the current key to close the window.
    pub fn close_key(&self) -> Option<Key> {
        self.close_key
    }

    /// Returns the current modifiers to close the window.
    pub fn close_modifiers(&self) -> Option<Modifiers> {
        self.close_modifiers
    }

    /// Creates a new hidden window.
    ///
    /// The window is created but not displayed. Use [`show()`](Self::show) to make it visible.
    /// The default size is 800x600 pixels.
    ///
    /// While hidden, the window renders off-screen instead of to its surface,
    /// so [`snap`](Self::snap), [`snap_image`](Self::snap_image) and recording
    /// work without ever displaying anything — this is how kiss3d does
    /// offscreen rendering.
    ///
    /// # Arguments
    /// * `title` - The window title
    ///
    /// # Returns
    /// A new `Window` instance
    pub async fn new_hidden(title: &str) -> Window {
        Window::do_new(title, true, DEFAULT_WIDTH, DEFAULT_HEIGHT, None).await
    }

    /// Creates a new hidden window with custom dimensions.
    ///
    /// The window is created but not displayed. Use [`show()`](Self::show) to make it visible.
    ///
    /// While hidden, the window renders off-screen instead of to its surface,
    /// so [`snap`](Self::snap), [`snap_image`](Self::snap_image) and recording
    /// work without ever displaying anything — this is how kiss3d does
    /// offscreen rendering.
    ///
    /// # Arguments
    /// * `title` - The window title
    /// * `width` - The window width in pixels
    /// * `height` - The window height in pixels
    ///
    /// # Returns
    /// A new `Window` instance
    pub async fn new_hidden_with_size(title: &str, width: u32, height: u32) -> Window {
        Window::do_new(title, true, width, height, None).await
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

    /// Creates a new window with custom attributes.
    ///
    /// This allows fine-grained control over window creation.
    ///
    /// # Arguments
    /// * `window_attrs` - The window title
    ///
    /// # Returns
    /// A new `Window` instance
    pub async fn new_with_window_attributes(window_attrs: WindowAttributes) -> Window {
        Window::do_new_with_window_attributes(window_attrs, None).await
    }

    // TODO: make this pub?
    async fn do_new(
        title: &str,
        hide: bool,
        width: u32,
        height: u32,
        setup: Option<CanvasSetup>,
    ) -> Window {
        let window_attrs = WindowAttributes::default()
            .with_title(title)
            .with_inner_size(LogicalSize::new(width as f64, height as f64))
            .with_visible(!hide);
        Self::do_new_with_window_attributes(window_attrs, setup).await
    }
    async fn do_new_with_window_attributes(
        window_attrs: WindowAttributes,
        setup: Option<CanvasSetup>,
    ) -> Window {
        let (event_send, event_receive) = mpsc::channel();
        let hide = !window_attrs.visible;
        let canvas = Canvas::open(window_attrs, setup, event_send).await;
        let (width, height) = canvas.size();
        // The HDR resolve pass tonemaps into the LDR swapchain. The rasterizer's
        // material pipelines are single-sampled, so the HDR film is too (see the
        // note in `render_single_frame`).
        let canvas_surface_format = canvas.surface_format();

        // Track window count for proper cleanup
        Context::increment_window_count();

        WindowCache::populate();

        let framebuffer_manager = FramebufferManager::new();
        let mut usr_window = Window {
            should_close: false,
            first_frame: true,
            close_key: Some(Key::Escape),
            close_modifiers: None,
            last_timings: None,
            gpu_timer: GpuTimer::new(),
            canvas,
            events: Rc::new(event_receive),
            unhandled_events: Rc::new(RefCell::new(Vec::new())),
            ambient_intensity: 0.2,
            ambient_color: crate::color::WHITE,
            fog: crate::light::Fog::default(),
            background: BLACK,
            polyline_renderer_2d: PolylineRenderer2d::new(),
            point_renderer_2d: PointRenderer2d::new(),
            point_renderer: PointRenderer3d::new(),
            polyline_renderer: PolylineRenderer3d::new(),
            text_renderer: TextRenderer::new(),
            #[cfg(feature = "egui")]
            egui_context: EguiContext::new(),
            hdr: HdrPipeline::new(width, height, 1, canvas_surface_format),
            post_process_render_target: framebuffer_manager.new_render_target(width, height, true),
            offscreen_output_target: None,
            aov_renderer: None,
            hidden: hide,
            shadow_mapper: ShadowMapper::new(DEFAULT_SHADOW_RESOLUTION),
            framebuffer_manager,
            #[cfg(feature = "recording")]
            recording: None,
        };

        if hide {
            usr_window.canvas.hide()
        }

        usr_window
    }

    /// Creates a headless window: a render target backed by no actual window,
    /// for off-screen rendering. Powers [`OffscreenSurface`](crate::window::OffscreenSurface).
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) async fn do_new_headless(
        width: u32,
        height: u32,
        setup: Option<CanvasSetup>,
    ) -> Window {
        let (event_send, event_receive) = mpsc::channel();
        let canvas = Canvas::open_headless(width, height, setup, event_send).await;
        let (width, height) = canvas.size();
        // A headless surface is never multisampled.
        let canvas_surface_format = canvas.surface_format();

        Context::increment_window_count();
        WindowCache::populate();

        let framebuffer_manager = FramebufferManager::new();
        Window {
            should_close: false,
            first_frame: true,
            close_key: None,
            close_modifiers: None,
            last_timings: None,
            gpu_timer: GpuTimer::new(),
            canvas,
            events: Rc::new(event_receive),
            unhandled_events: Rc::new(RefCell::new(Vec::new())),
            ambient_intensity: 0.2,
            ambient_color: crate::color::WHITE,
            fog: crate::light::Fog::default(),
            background: BLACK,
            polyline_renderer_2d: PolylineRenderer2d::new(),
            point_renderer_2d: PointRenderer2d::new(),
            point_renderer: PointRenderer3d::new(),
            polyline_renderer: PolylineRenderer3d::new(),
            text_renderer: TextRenderer::new(),
            #[cfg(feature = "egui")]
            egui_context: EguiContext::new(),
            // Offscreen rendering is single-sampled (see `render_single_frame`).
            hdr: HdrPipeline::new(width, height, 1, canvas_surface_format),
            post_process_render_target: framebuffer_manager.new_render_target(width, height, true),
            offscreen_output_target: None,
            aov_renderer: None,
            // A headless window has no surface; always render off-screen.
            hidden: true,
            shadow_mapper: ShadowMapper::new(DEFAULT_SHADOW_RESOLUTION),
            framebuffer_manager,
            #[cfg(feature = "recording")]
            recording: None,
        }
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
