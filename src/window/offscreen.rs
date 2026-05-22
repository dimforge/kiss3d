//! Off-screen (headless) rendering surface.

use crate::camera::{Camera2d, Camera3d};
use crate::color::Color;
use crate::post_processing::PostProcessingEffect;
use crate::renderer::Renderer3d;
use crate::scene::{SceneNode2d, SceneNode3d};
use crate::window::{CanvasSetup, Window};
use glamx::UVec2;
use image::{ImageBuffer, Rgb};

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
            .render(scene, scene_2d, camera, camera_2d, renderer, post_processing)
            .await;
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
}
