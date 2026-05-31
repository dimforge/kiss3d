//! Off-screen (headless) rendering surface.

use crate::camera::{Camera2d, Camera3d};
use crate::color::Color;
use crate::post_processing::{PostProcessingEffect, Tonemap};
use crate::renderer::{RayTracer, Renderer3d};
use crate::scene::{SceneNode2d, SceneNode3d};
use crate::window::{CanvasSetup, Window};
use glamx::UVec2;
use image::{ImageBuffer, Luma, Rgb};

/// A headless rendering surface.
///
/// Unlike [`Window`], an `OffscreenSurface` creates **no window and no event
/// loop**: it renders a scene straight into a texture. It therefore works in
/// environments with no display server (CI, servers, containers), produces no
/// on-screen flicker, and can render at any resolution — independent of the
/// display.
///
/// The scene graph, cameras, lights and materials are exactly the same as with
/// [`Window`]. Since there are no input events, interactive cameras stay put;
/// position the camera programmatically instead.
///
/// # Example
/// ```no_run
/// use kiss3d::prelude::*;
///
/// #[kiss3d::main]
/// async fn main() {
///     let mut surface = OffscreenSurface::new(1920, 1080).await;
///
///     let mut scene = SceneNode3d::empty();
///     scene.add_cube(1.0, 1.0, 1.0).set_color(RED);
///     let mut camera = OrbitCamera3d::default();
///
///     surface.render_3d(&mut scene, &mut camera).await;
///     surface.snap_image().save("out.png").unwrap();
/// }
/// ```
pub struct OffscreenSurface {
    window: Window,
}

impl OffscreenSurface {
    /// Creates a new off-screen surface of the given size, in pixels.
    pub async fn new(width: u32, height: u32) -> OffscreenSurface {
        OffscreenSurface {
            window: Window::do_new_headless(width, height, None).await,
        }
    }

    /// Creates a new off-screen surface with custom setup options (e.g. MSAA).
    pub async fn with_setup(width: u32, height: u32, setup: CanvasSetup) -> OffscreenSurface {
        OffscreenSurface {
            window: Window::do_new_headless(width, height, Some(setup)).await,
        }
    }

    /// Renders one frame of a 3D scene into the off-screen texture.
    pub async fn render_3d(&mut self, scene: &mut SceneNode3d, camera: &mut impl Camera3d) {
        let _ = self.window.render_3d(scene, camera).await;
    }

    /// Renders one frame of a 2D scene into the off-screen texture.
    pub async fn render_2d(&mut self, scene: &mut SceneNode2d, camera: &mut impl Camera2d) {
        let _ = self.window.render_2d(scene, camera).await;
    }

    /// Renders one frame with full control over the scenes, cameras, an
    /// optional custom renderer and post-processing effect. See [`Window::render`].
    #[allow(clippy::too_many_arguments)]
    pub async fn render(
        &mut self,
        scene: Option<&mut SceneNode3d>,
        scene_2d: Option<&mut SceneNode2d>,
        camera: Option<&mut dyn Camera3d>,
        camera_2d: Option<&mut dyn Camera2d>,
        renderer: Option<&mut dyn Renderer3d>,
        post_processing: Option<&mut dyn PostProcessingEffect>,
    ) {
        let _ = self
            .window
            .render(
                scene,
                scene_2d,
                camera,
                camera_2d,
                renderer,
                post_processing,
            )
            .await;
    }

    /// Renders one path-traced frame into the off-screen texture.
    ///
    /// Call repeatedly with the same [`RayTracer`] to accumulate samples (the
    /// camera is static off-screen, so accumulation only restarts on the first
    /// frame). See [`Window::render_raytraced`].
    pub async fn render_raytraced(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut impl Camera3d,
        raytracer: &mut RayTracer,
    ) {
        let _ = self.window.render_raytraced(scene, camera, raytracer).await;
    }

    /// Path-traces a 3D scene for `samples` accumulated frames and returns the
    /// resulting image, in one call.
    pub async fn render_image_raytraced(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut impl Camera3d,
        raytracer: &mut RayTracer,
        samples: u32,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        for _ in 0..samples.max(1) {
            self.render_raytraced(scene, camera, raytracer).await;
        }
        self.snap_image()
    }

    /// Renders a 3D scene and returns the resulting image, in one call.
    pub async fn render_image_3d(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut impl Camera3d,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        self.render_3d(scene, camera).await;
        self.snap_image()
    }

    /// Captures the last rendered frame as raw RGB pixel data.
    pub fn snap(&self, out: &mut Vec<u8>) {
        self.window.snap(out)
    }

    /// Captures a rectangular region of the last rendered frame as raw RGB data.
    pub fn snap_rect(&self, out: &mut Vec<u8>, x: usize, y: usize, width: usize, height: usize) {
        self.window.snap_rect(out, x, y, width, height)
    }

    /// Captures the last rendered frame as an image.
    pub fn snap_image(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        self.window.snap_image()
    }

    // === Auxiliary render outputs (AOVs) ===

    /// Renders the scene and returns per-pixel linear, eye-space depth (in world
    /// units), row-major with a top-left origin. Background pixels read back as
    /// `0.0`. See [`Window::snap_depth_raw`].
    pub fn snap_depth_raw(&mut self, scene: &mut SceneNode3d, camera: &mut impl Camera3d) -> Vec<f32> {
        self.window.snap_depth_raw(scene, camera)
    }

    /// Renders the scene and returns its depth as a normalized 8-bit grayscale
    /// image (nearest surface brightest, background black). See
    /// [`Window::snap_depth`].
    pub fn snap_depth(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut impl Camera3d,
    ) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        self.window.snap_depth(scene, camera)
    }

    /// Renders the scene and returns its world-space surface normals, encoded
    /// from `[-1, 1]` to `[0, 255]` per channel. See [`Window::snap_normals`].
    pub fn snap_normals(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut impl Camera3d,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        self.window.snap_normals(scene, camera)
    }

    /// Like [`snap_normals`](Self::snap_normals) but in camera (eye) space. See
    /// [`Window::snap_camera_normals`].
    pub fn snap_camera_normals(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut impl Camera3d,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        self.window.snap_camera_normals(scene, camera)
    }

    /// Renders the scene and returns the per-pixel segmentation/object id (`0`
    /// for background), row-major with a top-left origin. See
    /// [`Window::snap_segmentation`].
    pub fn snap_segmentation(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut impl Camera3d,
    ) -> Vec<u32> {
        self.window.snap_segmentation(scene, camera)
    }

    /// Renders the scene and returns a colorized segmentation image (each id
    /// mapped to a distinct color, background black). See
    /// [`Window::snap_segmentation_colored`].
    pub fn snap_segmentation_colored(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut impl Camera3d,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        self.window.snap_segmentation_colored(scene, camera)
    }

    /// Resizes the off-screen surface. The next render uses the new size.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.window.canvas_mut().resize(width, height);
    }

    /// The size of the surface, in pixels.
    pub fn size(&self) -> UVec2 {
        self.window.size()
    }

    /// The width of the surface, in pixels.
    pub fn width(&self) -> u32 {
        self.window.width()
    }

    /// The height of the surface, in pixels.
    pub fn height(&self) -> u32 {
        self.window.height()
    }

    /// Sets the background color.
    pub fn set_background_color(&mut self, color: Color) {
        self.window.set_background_color(color);
    }

    /// Sets the global ambient light intensity (also drives the path tracer's
    /// sky/environment term).
    pub fn set_ambient(&mut self, ambient: f32) {
        self.window.set_ambient(ambient);
    }

    /// Sets the exposure multiplier applied before tonemapping (`1.0` is neutral).
    /// See [`Window::set_exposure`].
    pub fn set_exposure(&mut self, exposure: f32) {
        self.window.set_exposure(exposure);
    }

    /// Sets the per-layer resolution of the shadow atlas (higher = sharper shadows,
    /// more memory). See [`Window::set_shadow_resolution`].
    pub fn set_shadow_resolution(&mut self, resolution: u32) {
        self.window.set_shadow_resolution(resolution);
    }

    /// Selects the tonemapping operator used by the HDR resolve pass.
    /// See [`Window::set_tonemap`].
    pub fn set_tonemap(&mut self, tonemap: Tonemap) {
        self.window.set_tonemap(tonemap);
    }

    /// Enables or disables bloom. See [`Window::set_bloom_enabled`].
    pub fn set_bloom_enabled(&mut self, enabled: bool) {
        self.window.set_bloom_enabled(enabled);
    }

    /// Sets the bloom brightness threshold and additive intensity.
    /// See [`Window::set_bloom`].
    pub fn set_bloom(&mut self, threshold: f32, intensity: f32) {
        self.window.set_bloom(threshold, intensity);
    }

    /// Queues an egui UI to be drawn over the next rendered frame. See
    /// [`Window::draw_ui`].
    #[cfg(feature = "egui")]
    pub fn draw_ui<F>(&mut self, ui_fn: F)
    where
        F: FnOnce(&egui::Context),
    {
        self.window.draw_ui(ui_fn);
    }
}
