#![allow(clippy::await_holding_refcell_ref)]

//! The kiss3d window.
/*
 * FIXME: this file is too big. Some heavy refactoring need to be done here.
 */
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use na::{Point2, Point3, Vector2, Vector3};

use crate::camera::{ArcBall, Camera};
use crate::context::Context;
use crate::event::MouseButton;
use crate::event::{Action, EventManager, Key, WindowEvent};
use crate::light::Light;
use crate::planar_camera::{PlanarCamera, PlanarFixedView};
use crate::planar_point_renderer::PlanarPointRenderer;
use crate::planar_polyline_renderer::{PlanarPolyline, PlanarPolylineRenderer};
use crate::post_processing::{PostProcessingContext, PostProcessingEffect};
#[cfg(feature = "egui")]
use crate::renderer::EguiRenderer;
use crate::renderer::{PointRenderer, Polyline, PolylineRenderer, Renderer};
use crate::resource::{
    FramebufferManager, GpuMesh, PlanarMesh, PlanarRenderContext, RenderContext, RenderTarget,
    Texture, TextureManager,
};
use crate::scene::{PlanarSceneNode, SceneNode};
use crate::text::{Font, TextRenderer};
use crate::window::canvas::CanvasSetup;
use crate::window::Canvas;
use image::imageops;
use image::{GenericImage, Pixel};
use image::{ImageBuffer, Rgb};
use parry3d::shape::TriMesh;

use super::window_cache::WindowCache;
use crate::procedural::RenderMesh;
#[cfg(feature = "egui")]
use egui::RawInput;
use std::sync::Arc;

static DEFAULT_WIDTH: u32 = 800u32;
static DEFAULT_HEIGHT: u32 = 600u32;

#[cfg(feature = "egui")]
struct EguiContext {
    renderer: EguiRenderer,
    raw_input: RawInput,
    #[cfg(not(target_arch = "wasm32"))]
    start_time: std::time::Instant,
}

#[cfg(feature = "egui")]
impl EguiContext {
    fn new() -> Self {
        Self {
            renderer: EguiRenderer::new(),
            raw_input: RawInput::default(),
            #[cfg(not(target_arch = "wasm32"))]
            start_time: std::time::Instant::now(),
        }
    }
}

/// Configuration options for video recording.
///
/// Use this to customize recording behavior such as frame skipping.
#[cfg(feature = "recording")]
#[derive(Clone)]
pub struct RecordingConfig {
    /// Record every Nth frame. Set to 1 to record every frame,
    /// 2 to record every other frame, etc.
    /// Default: 1
    pub frame_skip: u32,
}

#[cfg(feature = "recording")]
impl Default for RecordingConfig {
    fn default() -> Self {
        Self { frame_skip: 1 }
    }
}

#[cfg(feature = "recording")]
impl RecordingConfig {
    /// Creates a new recording config with default settings (every frame).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets how many frames to skip between captures.
    /// 1 = every frame, 2 = every other frame, etc.
    pub fn with_frame_skip(mut self, skip: u32) -> Self {
        self.frame_skip = skip.max(1);
        self
    }
}

/// State for video recording.
#[cfg(feature = "recording")]
struct RecordingState {
    frames: Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>,
    width: u32,
    height: u32,
    config: RecordingConfig,
    paused: bool,
    frame_counter: u32,
}

/// Structure representing a window and a 3D scene.
///
/// This is the main interface with the 3d engine.
pub struct Window {
    events: Rc<Receiver<WindowEvent>>,
    unhandled_events: Rc<RefCell<Vec<WindowEvent>>>,
    min_dur_per_frame: Option<Duration>,
    scene: SceneNode,
    scene2: PlanarSceneNode,
    light_mode: Light, // FIXME: move that to the scene graph
    background: Vector3<f32>,
    planar_polyline_renderer: PlanarPolylineRenderer,
    planar_point_renderer: PlanarPointRenderer,
    point_renderer: PointRenderer,
    polyline_renderer: PolylineRenderer,
    text_renderer: TextRenderer,
    #[allow(dead_code)]
    framebuffer_manager: FramebufferManager,
    post_process_render_target: RenderTarget,
    #[cfg(not(target_arch = "wasm32"))]
    curr_time: std::time::Instant,
    planar_camera: Rc<RefCell<PlanarFixedView>>,
    camera: Rc<RefCell<ArcBall>>,
    should_close: bool,
    #[cfg(feature = "egui")]
    egui_context: EguiContext,
    canvas: Canvas,
    #[cfg(feature = "recording")]
    recording: Option<RecordingState>,
}

impl Drop for Window {
    fn drop(&mut self) {
        WindowCache::clear();
    }
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
    pub fn size(&self) -> Vector2<u32> {
        let (w, h) = self.canvas.size();
        Vector2::new(w, h)
    }

    /// Sets the maximum number of frames per second. Cannot be 0. `None` means there is no limit.
    #[inline]
    pub fn set_framerate_limit(&mut self, fps: Option<u64>) {
        self.min_dur_per_frame = fps.map(|f| {
            assert!(f != 0);
            Duration::from_millis(1000 / f)
        })
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
    /// let mut window = Window::new("Example").await;
    /// window.set_background_color(0.1, 0.2, 0.3); // Dark blue-gray
    /// # }
    /// ```
    #[inline]
    pub fn set_background_color(&mut self, r: f32, g: f32, b: f32) {
        self.background.x = r;
        self.background.y = g;
        self.background.z = b;
    }

    /// Draws a 3D line for the current frame.
    ///
    /// The line is only drawn during the next frame. To keep a line visible,
    /// call this method every frame from within your render loop.
    ///
    /// # Arguments
    /// * `a` - The starting point of the line in 3D space
    /// * `b` - The ending point of the line in 3D space
    /// * `color` - RGB color (each component from 0.0 to 1.0)
    /// * `width` - Line width in pixels
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # use nalgebra::{Point3};
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// let start = Point3::new(0.0, 0.0, 0.0);
    /// let end = Point3::new(1.0, 1.0, 1.0);
    /// let red = Point3::new(1.0, 0.0, 0.0);
    /// window.draw_line(&start, &end, &red, 2.0);
    /// # }
    /// ```
    #[inline]
    pub fn draw_line(&mut self, a: &Point3<f32>, b: &Point3<f32>, color: &Point3<f32>, width: f32) {
        self.polyline_renderer.draw_line(*a, *b, *color, width);
    }

    /// Draws a 2D line for the current frame.
    ///
    /// The line is only drawn during the next frame. To keep a line visible,
    /// call this method every frame from within your render loop.
    ///
    /// # Arguments
    /// * `a` - The starting point of the line in 2D space
    /// * `b` - The ending point of the line in 2D space
    /// * `color` - RGB color (each component from 0.0 to 1.0)
    /// * `width` - Line width in pixels
    #[inline]
    pub fn draw_planar_line(&mut self, a: &Point2<f32>, b: &Point2<f32>, color: &Point3<f32>, width: f32) {
        self.planar_polyline_renderer.draw_line(*a, *b, *color, width);
    }

    /// Draws a 2D polyline (connected line segments) with configurable width.
    ///
    /// The polyline is only drawn during the next frame. To keep it visible,
    /// call this method every frame from within your render loop.
    ///
    /// Takes a reference to avoid allocations - segments are built immediately.
    ///
    /// # Arguments
    /// * `polyline` - The 2D polyline to draw
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # use kiss3d::planar_polyline_renderer::PlanarPolyline;
    /// # use nalgebra::Point2;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// let polyline = PlanarPolyline::new(vec![
    ///     Point2::new(0.0, 0.0),
    ///     Point2::new(100.0, 100.0),
    ///     Point2::new(200.0, 0.0),
    /// ])
    /// .with_color(1.0, 0.0, 0.0)
    /// .with_width(5.0);
    /// window.draw_planar_polyline(&polyline);
    /// # }
    /// ```
    #[inline]
    pub fn draw_planar_polyline(&mut self, polyline: &PlanarPolyline) {
        self.planar_polyline_renderer.draw_polyline(polyline);
    }

    /// Draws a 2D point for the current frame.
    ///
    /// The point is only drawn during the next frame. To keep a point visible,
    /// call this method every frame from within your render loop.
    ///
    /// # Arguments
    /// * `pt` - The position of the point in 2D space
    /// * `color` - RGB color (each component from 0.0 to 1.0)
    /// * `size` - The point size in pixels
    #[inline]
    pub fn draw_planar_point(&mut self, pt: &Point2<f32>, color: &Point3<f32>, size: f32) {
        self.planar_point_renderer.draw_point(*pt, *color, size);
    }

    /// Draws a 3D point for the current frame.
    ///
    /// The point is only drawn during the next frame. To keep a point visible,
    /// call this method every frame from within your render loop.
    ///
    /// # Arguments
    /// * `pt` - The position of the point in 3D space
    /// * `color` - RGB color (each component from 0.0 to 1.0)
    /// * `size` - The point size in pixels
    #[inline]
    pub fn draw_point(&mut self, pt: &Point3<f32>, color: &Point3<f32>, size: f32) {
        self.point_renderer.draw_point(*pt, *color, size);
    }

    /// Draws a polyline (connected line segments) with configurable width.
    ///
    /// The polyline is only drawn during the next frame. To keep it visible,
    /// call this method every frame from within your render loop.
    ///
    /// # Arguments
    /// * `polyline` - The polyline to draw
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # use kiss3d::renderer::Polyline;
    /// # use nalgebra::Point3;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// let polyline = Polyline::new(vec![
    ///     Point3::new(0.0, 0.0, 0.0),
    ///     Point3::new(1.0, 1.0, 0.0),
    ///     Point3::new(2.0, 0.0, 0.0),
    /// ])
    /// .with_color(1.0, 0.0, 0.0)
    /// .with_width(5.0);
    /// window.draw_polyline(&polyline);
    /// # }
    /// ```
    #[inline]
    pub fn draw_polyline(&mut self, polyline: &Polyline) {
        self.polyline_renderer.draw_polyline(polyline);
    }

    /// Draws text for the current frame.
    ///
    /// The text is only drawn during the next frame. To keep text visible,
    /// call this method every frame from within your render loop.
    ///
    /// # Arguments
    /// * `text` - The string to display
    /// * `pos` - The position in 2D screen coordinates
    /// * `scale` - The text scale factor
    /// * `font` - A reference to the font to use
    /// * `color` - RGB color (each component from 0.0 to 1.0)
    #[inline]
    pub fn draw_text(
        &mut self,
        text: &str,
        pos: &Point2<f32>,
        scale: f32,
        font: &Arc<Font>,
        color: &Point3<f32>,
    ) {
        self.text_renderer.draw_text(text, pos, scale, font, color);
    }

    /// Removes a 3D object from the scene.
    #[deprecated(note = "Use `remove_node` instead.")]
    pub fn remove(&mut self, sn: &mut SceneNode) {
        self.remove_node(sn)
    }

    /// Removes a 3D object from the scene.
    ///
    /// # Arguments
    /// * `sn` - The scene node to remove. After this call, the node is unlinked from its parent.
    pub fn remove_node(&mut self, sn: &mut SceneNode) {
        sn.unlink()
    }

    /// Removes a 2D object from the scene.
    ///
    /// # Arguments
    /// * `sn` - The planar scene node to remove. After this call, the node is unlinked from its parent.
    pub fn remove_planar_node(&mut self, sn: &mut PlanarSceneNode) {
        sn.unlink()
    }

    /// Adds an empty group node to the 3D scene.
    ///
    /// A group is a node without any geometry. It's useful for organizing
    /// scene hierarchies and applying transformations to multiple objects at once.
    ///
    /// # Returns
    /// A new `SceneNode` representing the group, which can have children added to it.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// let mut group = window.add_group();
    /// let mut cube1 = group.add_cube(1.0, 1.0, 1.0);
    /// let mut cube2 = group.add_cube(1.0, 1.0, 1.0);
    /// // Now transforming 'group' will affect both cubes
    /// # }
    /// ```
    pub fn add_group(&mut self) -> SceneNode {
        self.scene.add_group()
    }

    /// Adds an empty group node to the 2D scene.
    ///
    /// A group is a node without any geometry. It's useful for organizing
    /// scene hierarchies and applying transformations to multiple objects at once.
    ///
    /// # Returns
    /// A new `PlanarSceneNode` representing the group.
    pub fn add_planar_group(&mut self) -> PlanarSceneNode {
        self.scene2.add_group()
    }

    /// Loads and adds an OBJ model to the scene.
    ///
    /// # Arguments
    /// * `path` - Path to the .obj file to load
    /// * `mtl_dir` - Directory path where material (.mtl) files are located
    /// * `scale` - Scale factor to apply to the model along each axis
    ///
    /// # Returns
    /// A `SceneNode` representing the loaded model
    pub fn add_obj(&mut self, path: &Path, mtl_dir: &Path, scale: Vector3<f32>) -> SceneNode {
        self.scene.add_obj(path, mtl_dir, scale)
    }

    /// Adds a 3D mesh to the scene.
    ///
    /// # Arguments
    /// * `mesh` - A reference-counted GPU mesh to add
    /// * `scale` - Scale factor to apply to the mesh along each axis
    ///
    /// # Returns
    /// A `SceneNode` representing the mesh in the scene
    pub fn add_mesh(&mut self, mesh: Rc<RefCell<GpuMesh>>, scale: Vector3<f32>) -> SceneNode {
        self.scene.add_mesh(mesh, scale)
    }

    /// Adds a 2D mesh to the scene.
    ///
    /// # Arguments
    /// * `mesh` - A reference-counted planar mesh to add
    /// * `scale` - Scale factor to apply to the mesh along each axis
    ///
    /// # Returns
    /// A `PlanarSceneNode` representing the 2D mesh in the scene
    pub fn add_planar_mesh(
        &mut self,
        mesh: Rc<RefCell<PlanarMesh>>,
        scale: Vector2<f32>,
    ) -> PlanarSceneNode {
        self.scene2.add_mesh(mesh, scale)
    }

    /// Creates and adds a new 3D object from a triangle mesh.
    ///
    /// # Arguments
    /// * `mesh` - A `parry3d::shape::TriMesh` containing the geometry
    /// * `scale` - Scale factor to apply to the mesh along each axis
    ///
    /// # Returns
    /// A `SceneNode` representing the mesh in the scene
    pub fn add_trimesh(&mut self, mesh: TriMesh, scale: Vector3<f32>) -> SceneNode {
        self.scene.add_trimesh(mesh, scale)
    }

    /// Creates and adds a new 3D object from procedurally generated geometry.
    ///
    /// # Arguments
    /// * `mesh` - A `RenderMesh` containing the procedurally generated geometry
    /// * `scale` - Scale factor to apply to the mesh along each axis
    ///
    /// # Returns
    /// A `SceneNode` representing the mesh in the scene
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # use kiss3d::procedural;
    /// # use nalgebra::Vector3;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// let mesh = procedural::sphere(1.0, 32, 32, true);
    /// let mut node = window.add_render_mesh(mesh, Vector3::new(1.0, 1.0, 1.0));
    /// # }
    /// ```
    pub fn add_render_mesh(&mut self, mesh: RenderMesh, scale: Vector3<f32>) -> SceneNode {
        self.scene.add_render_mesh(mesh, scale)
    }

    /// Creates and adds a new object using a geometry registered with a specific name.
    ///
    /// # Arguments
    /// * `geometry_name` - The name of the registered geometry
    /// * `scale` - Scale factor to apply to the geometry along each axis
    ///
    /// # Returns
    /// `Some(SceneNode)` if the geometry was found, `None` otherwise
    pub fn add_geom_with_name(
        &mut self,
        geometry_name: &str,
        scale: Vector3<f32>,
    ) -> Option<SceneNode> {
        self.scene.add_geom_with_name(geometry_name, scale)
    }

    /// Adds a cube to the scene. The cube is initially axis-aligned and centered at (0, 0, 0).
    ///
    /// # Arguments
    /// * `wx` - the cube extent along the x axis
    /// * `wy` - the cube extent along the y axis
    /// * `wz` - the cube extent along the z axis
    pub fn add_cube(&mut self, wx: f32, wy: f32, wz: f32) -> SceneNode {
        self.scene.add_cube(wx, wy, wz)
    }

    /// Adds a sphere to the scene. The sphere is initially centered at (0, 0, 0).
    ///
    /// # Arguments
    /// * `r` - the sphere radius
    pub fn add_sphere(&mut self, r: f32) -> SceneNode {
        self.scene.add_sphere(r)
    }

    /// Adds a cone to the scene. The cone is initially centered at (0, 0, 0) and points toward the
    /// positive `y` axis.
    ///
    /// # Arguments
    /// * `h` - the cone height
    /// * `r` - the cone base radius
    pub fn add_cone(&mut self, r: f32, h: f32) -> SceneNode {
        self.scene.add_cone(r, h)
    }

    /// Adds a cylinder to the scene. The cylinder is initially centered at (0, 0, 0) and has its
    /// principal axis aligned with the `y` axis.
    ///
    /// # Arguments
    /// * `h` - the cylinder height
    /// * `r` - the cylinder base radius
    pub fn add_cylinder(&mut self, r: f32, h: f32) -> SceneNode {
        self.scene.add_cylinder(r, h)
    }

    /// Adds a capsule to the scene. The capsule is initially centered at (0, 0, 0) and has its
    /// principal axis aligned with the `y` axis.
    ///
    /// # Arguments
    /// * `r` - the capsule caps radius
    /// * `h` - the capsule height
    pub fn add_capsule(&mut self, r: f32, h: f32) -> SceneNode {
        self.scene.add_capsule(r, h)
    }

    /// Adds a 2D capsule to the scene. The capsule is initially centered at (0, 0) and has its
    /// principal axis aligned with the `y` axis.
    ///
    /// # Arguments
    /// * `r` - the capsule caps radius
    /// * `h` - the capsule height
    pub fn add_planar_capsule(&mut self, r: f32, h: f32) -> PlanarSceneNode {
        self.scene2.add_capsule(r, h)
    }

    /// Adds a double-sided subdivided quad to the scene.
    ///
    /// The quad is initially centered at (0, 0, 0) and lies in the XY plane.
    /// It's composed of a user-defined number of triangles regularly spaced on a grid.
    /// This is useful for drawing height maps or terrain.
    ///
    /// # Arguments
    /// * `w` - The quad width
    /// * `h` - The quad height
    /// * `usubdivs` - Number of horizontal subdivisions (number of squares along width). Must not be `0`.
    /// * `vsubdivs` - Number of vertical subdivisions (number of squares along height). Must not be `0`.
    ///
    /// # Returns
    /// A `SceneNode` representing the quad
    ///
    /// # Panics
    /// Panics if `usubdivs` or `vsubdivs` is `0`.
    pub fn add_quad(&mut self, w: f32, h: f32, usubdivs: usize, vsubdivs: usize) -> SceneNode {
        self.scene.add_quad(w, h, usubdivs, vsubdivs)
    }

    /// Adds a double-sided quad with custom vertex positions.
    ///
    /// # Arguments
    /// * `vertices` - Array of vertex positions defining the quad surface
    /// * `nhpoints` - Number of points along the horizontal direction
    /// * `nvpoints` - Number of points along the vertical direction
    ///
    /// # Returns
    /// A `SceneNode` representing the quad
    pub fn add_quad_with_vertices(
        &mut self,
        vertices: &[Point3<f32>],
        nhpoints: usize,
        nvpoints: usize,
    ) -> SceneNode {
        self.scene
            .add_quad_with_vertices(vertices, nhpoints, nvpoints)
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

    /// Adds a 2D rectangle to the scene.
    ///
    /// The rectangle is initially axis-aligned and centered at (0, 0).
    ///
    /// # Arguments
    /// * `wx` - The rectangle width (extent along the x-axis)
    /// * `wy` - The rectangle height (extent along the y-axis)
    ///
    /// # Returns
    /// A `PlanarSceneNode` representing the rectangle
    pub fn add_rectangle(&mut self, wx: f32, wy: f32) -> PlanarSceneNode {
        self.scene2.add_rectangle(wx, wy)
    }

    /// Adds a 2D circle to the scene.
    ///
    /// The circle is initially centered at (0, 0).
    ///
    /// # Arguments
    /// * `r` - The circle radius
    ///
    /// # Returns
    /// A `PlanarSceneNode` representing the circle
    pub fn add_circle(&mut self, r: f32) -> PlanarSceneNode {
        self.scene2.add_circle(r)
    }

    /// Adds a 2D convex polygon to the scene.
    ///
    /// # Arguments
    /// * `polygon` - Vector of points defining the polygon vertices in counter-clockwise order
    /// * `scale` - Scale factor to apply to the polygon along each axis
    ///
    /// # Returns
    /// A `PlanarSceneNode` representing the polygon
    pub fn add_convex_polygon(
        &mut self,
        polygon: Vec<Point2<f32>>,
        scale: Vector2<f32>,
    ) -> PlanarSceneNode {
        self.scene2.add_convex_polygon(polygon, scale)
    }

    /// Checks whether this window is closed.
    ///
    /// # Returns
    /// Currently always returns `false`. Use [`should_close()`](Self::should_close) instead.
    pub fn is_closed(&self) -> bool {
        false // FIXME
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

    /// Sets the light mode for the scene.
    ///
    /// Currently, only one light is supported. The light affects how 3D objects are rendered.
    ///
    /// # Arguments
    /// * `pos` - The light configuration (see [`Light`] enum)
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # use kiss3d::light::Light;
    /// # use nalgebra::Point3;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// // Light attached to the camera
    /// window.set_light(Light::StickToCamera);
    ///
    /// // Fixed position light
    /// window.set_light(Light::Absolute(Point3::new(5.0, 5.0, 5.0)));
    /// # }
    /// ```
    pub fn set_light(&mut self, pos: Light) {
        self.light_mode = pos;
    }

    /// Retrieves a mutable reference to the egui context.
    ///
    /// Use this to access egui's full API for creating custom UI elements.
    ///
    /// # Returns
    /// A mutable reference to the egui Context
    ///
    /// # Note
    /// Only available when the `egui` feature is enabled.
    #[cfg(feature = "egui")]
    pub fn egui_context_mut(&mut self) -> &mut egui::Context {
        self.egui_context.renderer.context_mut()
    }

    /// Retrieves a reference to the egui context.
    ///
    /// Use this to access egui's API for reading UI state.
    ///
    /// # Returns
    /// A reference to the egui Context
    ///
    /// # Note
    /// Only available when the `egui` feature is enabled.
    #[cfg(feature = "egui")]
    pub fn egui_context(&self) -> &egui::Context {
        self.egui_context.renderer.context()
    }

    /// Checks if egui is currently capturing mouse input.
    ///
    /// Returns `true` if the mouse is hovering over or interacting with an egui widget.
    /// This is useful for preventing 3D camera controls from interfering with UI interaction.
    ///
    /// # Returns
    /// `true` if egui wants mouse input, `false` otherwise
    ///
    /// # Note
    /// Only available when the `egui` feature is enabled.
    #[cfg(feature = "egui")]
    pub fn is_egui_capturing_mouse(&self) -> bool {
        self.egui_context.renderer.wants_pointer_input()
    }

    /// Checks if egui is currently capturing keyboard input.
    ///
    /// Returns `true` if an egui text field or other widget has keyboard focus.
    /// This is useful for preventing keyboard shortcuts from triggering while typing in UI.
    ///
    /// # Returns
    /// `true` if egui wants keyboard input, `false` otherwise
    ///
    /// # Note
    /// Only available when the `egui` feature is enabled.
    #[cfg(feature = "egui")]
    pub fn is_egui_capturing_keyboard(&self) -> bool {
        self.egui_context.renderer.wants_keyboard_input()
    }

    /// Feed a window event to egui for processing.
    #[cfg(feature = "egui")]
    fn feed_egui_event(&mut self, event: &WindowEvent) {
        let scale_factor = self.scale_factor() as f32;

        match *event {
            WindowEvent::CursorPos(x, y, _) => {
                // Convert physical pixels to logical coordinates
                let pos = egui::Pos2::new((x as f32) / scale_factor, (y as f32) / scale_factor);
                self.egui_context
                    .raw_input
                    .events
                    .push(egui::Event::PointerMoved(pos));
            }
            WindowEvent::MouseButton(button, action, _) => {
                let button = match button {
                    crate::event::MouseButton::Button1 => egui::PointerButton::Primary,
                    crate::event::MouseButton::Button2 => egui::PointerButton::Secondary,
                    crate::event::MouseButton::Button3 => egui::PointerButton::Middle,
                    _ => return,
                };

                if let Some(pos) = self.cursor_pos() {
                    // Convert physical pixels to logical coordinates
                    let pos = egui::Pos2::new(
                        (pos.0 as f32) / scale_factor,
                        (pos.1 as f32) / scale_factor,
                    );
                    let pressed = action == Action::Press;

                    self.egui_context
                        .raw_input
                        .events
                        .push(egui::Event::PointerButton {
                            pos,
                            button,
                            pressed,
                            modifiers: self.get_egui_modifiers(),
                        });
                }
            }
            WindowEvent::Scroll(_x, y, _) => {
                self.egui_context
                    .raw_input
                    .events
                    .push(egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Line,
                        delta: egui::Vec2::new(0.0, y as f32),
                        modifiers: self.get_egui_modifiers(),
                    });
            }
            WindowEvent::Char(ch) => {
                if !ch.is_control() {
                    self.egui_context
                        .raw_input
                        .events
                        .push(egui::Event::Text(ch.to_string()));
                }
            }
            WindowEvent::Key(key, action, _modifiers) => {
                if let Some(egui_key) = self.translate_key_to_egui(key) {
                    self.egui_context.raw_input.events.push(egui::Event::Key {
                        key: egui_key,
                        physical_key: None,
                        pressed: action == Action::Press,
                        repeat: false,
                        modifiers: self.get_egui_modifiers(),
                    });
                }
            }
            _ => {}
        }
    }

    #[cfg(feature = "egui")]
    fn get_egui_modifiers(&self) -> egui::Modifiers {
        egui::Modifiers {
            alt: self.get_key(Key::LAlt) == Action::Press
                || self.get_key(Key::RAlt) == Action::Press,
            ctrl: self.get_key(Key::LControl) == Action::Press
                || self.get_key(Key::RControl) == Action::Press,
            shift: self.get_key(Key::LShift) == Action::Press
                || self.get_key(Key::RShift) == Action::Press,
            mac_cmd: false,
            command: self.get_key(Key::LControl) == Action::Press
                || self.get_key(Key::RControl) == Action::Press,
        }
    }

    #[cfg(feature = "egui")]
    fn translate_key_to_egui(&self, key: Key) -> Option<egui::Key> {
        Some(match key {
            Key::A => egui::Key::A,
            Key::B => egui::Key::B,
            Key::C => egui::Key::C,
            Key::D => egui::Key::D,
            Key::E => egui::Key::E,
            Key::F => egui::Key::F,
            Key::G => egui::Key::G,
            Key::H => egui::Key::H,
            Key::I => egui::Key::I,
            Key::J => egui::Key::J,
            Key::K => egui::Key::K,
            Key::L => egui::Key::L,
            Key::M => egui::Key::M,
            Key::N => egui::Key::N,
            Key::O => egui::Key::O,
            Key::P => egui::Key::P,
            Key::Q => egui::Key::Q,
            Key::R => egui::Key::R,
            Key::S => egui::Key::S,
            Key::T => egui::Key::T,
            Key::U => egui::Key::U,
            Key::V => egui::Key::V,
            Key::W => egui::Key::W,
            Key::X => egui::Key::X,
            Key::Y => egui::Key::Y,
            Key::Z => egui::Key::Z,
            Key::Escape => egui::Key::Escape,
            Key::Tab => egui::Key::Tab,
            Key::Back => egui::Key::Backspace,
            Key::Return => egui::Key::Enter,
            Key::Space => egui::Key::Space,
            Key::Insert => egui::Key::Insert,
            Key::Delete => egui::Key::Delete,
            Key::Home => egui::Key::Home,
            Key::End => egui::Key::End,
            Key::PageUp => egui::Key::PageUp,
            Key::PageDown => egui::Key::PageDown,
            Key::Left => egui::Key::ArrowLeft,
            Key::Up => egui::Key::ArrowUp,
            Key::Right => egui::Key::ArrowRight,
            Key::Down => egui::Key::ArrowDown,
            _ => return None,
        })
    }

    /// Draws an immediate mode UI using egui.
    ///
    /// Call this method from your render loop to create and display UI elements.
    /// The UI is drawn on top of the 3D scene.
    ///
    /// # Arguments
    /// * `ui_fn` - A closure that receives the egui Context and can create UI elements
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[cfg(feature = "egui")]
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// while window.render().await {
    ///     window.draw_ui(|ctx| {
    ///         egui::Window::new("My Window").show(ctx, |ui| {
    ///             ui.label("Hello, world!");
    ///             if ui.button("Click me").clicked() {
    ///                 println!("Button clicked!");
    ///             }
    ///         });
    ///     });
    /// }
    /// # }
    /// # #[cfg(not(feature = "egui"))]
    /// # fn main() {}
    /// ```
    ///
    /// # Note
    /// Only available when the `egui` feature is enabled.
    #[cfg(feature = "egui")]
    pub fn draw_ui<F>(&mut self, ui_fn: F)
    where
        F: FnOnce(&egui::Context),
    {
        // Get time for animations - use egui context's own start time
        #[cfg(not(target_arch = "wasm32"))]
        let time = Some(self.egui_context.start_time.elapsed().as_secs_f64());
        #[cfg(target_arch = "wasm32")]
        let time = {
            // On WASM, use instant which is already configured
            use instant::Instant;
            static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
            let start = START.get_or_init(Instant::now);
            Some(start.elapsed().as_secs_f64())
        };

        let scale_factor = self.canvas.scale_factor() as f32;

        // Set pixels_per_point on the context to match our DPI scale
        self.egui_context
            .renderer
            .context()
            .set_pixels_per_point(scale_factor);

        // Build raw input with accumulated events
        let mut raw_input = std::mem::take(&mut self.egui_context.raw_input);
        raw_input.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(
                self.width() as f32 / scale_factor,
                self.height() as f32 / scale_factor,
            ),
        ));
        raw_input.time = time;
        raw_input.predicted_dt = 1.0 / 60.0;

        self.egui_context.renderer.begin_frame(raw_input);
        ui_fn(self.egui_context.renderer.context());
        self.egui_context.renderer.end_frame();

        // Reset raw_input for next frame (but keep it properly initialized)
        self.egui_context.raw_input = RawInput::default();
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
    /// # use kiss3d::window::Window;
    /// #[kiss3d::main]
    /// async fn main() {
    ///     let mut window = Window::new("My Application").await;
    ///     while window.render().await {
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
    pub async fn new_with_setup(title: &str, width: u32, height: u32, setup: CanvasSetup) -> Window {
        Window::do_new(title, false, width, height, Some(setup)).await
    }

    // FIXME: make this pub?
    async fn do_new(
        title: &str,
        hide: bool,
        width: u32,
        height: u32,
        setup: Option<CanvasSetup>,
    ) -> Window {
        let (event_send, event_receive) = mpsc::channel();
        let canvas = Canvas::open(title, hide, width, height, setup, event_send).await;

        init_wgpu();
        WindowCache::populate();

        let framebuffer_manager = FramebufferManager::new();
        let mut usr_window = Window {
            should_close: false,
            min_dur_per_frame: None,
            canvas,
            events: Rc::new(event_receive),
            unhandled_events: Rc::new(RefCell::new(Vec::new())),
            scene: SceneNode::new_empty(),
            scene2: PlanarSceneNode::new_empty(),
            light_mode: Light::Absolute(Point3::new(0.0, 10.0, 0.0)),
            background: Vector3::new(0.0, 0.0, 0.0),
            planar_polyline_renderer: PlanarPolylineRenderer::new(),
            planar_point_renderer: PlanarPointRenderer::new(),
            point_renderer: PointRenderer::new(),
            polyline_renderer: PolylineRenderer::new(),
            text_renderer: TextRenderer::new(),
            #[cfg(feature = "egui")]
            egui_context: EguiContext::new(),
            post_process_render_target: framebuffer_manager.new_render_target(
                width,
                height,
                true,
            ),
            framebuffer_manager,
            #[cfg(not(target_arch = "wasm32"))]
            curr_time: std::time::Instant::now(),
            planar_camera: Rc::new(RefCell::new(PlanarFixedView::new())),
            camera: Rc::new(RefCell::new(ArcBall::new(
                Point3::new(0.0f32, 0.0, -1.0),
                Point3::origin(),
            ))),
            #[cfg(feature = "recording")]
            recording: None,
        };

        if hide {
            usr_window.canvas.hide()
        }

        // usr_window.framebuffer_size_callback(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        let light = usr_window.light_mode.clone();
        usr_window.set_light(light);

        usr_window
    }

    /// Returns an immutable reference to the root 3D scene node.
    ///
    /// The scene node forms the root of a hierarchical scene graph.
    /// You can traverse or query the scene through this node.
    ///
    /// # Returns
    /// An immutable reference to the root `SceneNode`
    #[inline]
    pub fn scene(&self) -> &SceneNode {
        &self.scene
    }

    /// Returns a mutable reference to the root 3D scene node.
    ///
    /// Use this to modify the scene graph structure or node properties.
    ///
    /// # Returns
    /// A mutable reference to the root `SceneNode`
    #[inline]
    pub fn scene_mut(&mut self) -> &mut SceneNode {
        &mut self.scene
    }

    /// Captures the current framebuffer as raw RGB pixel data.
    ///
    /// Reads all pixels currently displayed on the screen into a buffer.
    /// The buffer is automatically resized to fit the screen dimensions.
    /// Pixels are stored in RGB format (3 bytes per pixel), row by row from bottom to top.
    ///
    /// # Arguments
    /// * `out` - The output buffer. It will be resized to width × height × 3 bytes.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let window = Window::new("Example").await;
    /// let mut pixels = Vec::new();
    /// window.snap(&mut pixels);
    /// // pixels now contains RGB data
    /// # }
    /// ```
    pub fn snap(&self, out: &mut Vec<u8>) {
        let (width, height) = self.canvas.size();
        self.snap_rect(out, 0, 0, width as usize, height as usize)
    }

    /// Captures a rectangular region of the framebuffer as raw RGB pixel data.
    ///
    /// Reads a specific rectangular region of pixels from the screen.
    /// Pixels are stored in RGB format (3 bytes per pixel).
    ///
    /// # Arguments
    /// * `out` - The output buffer. It will be resized to width × height × 3 bytes.
    /// * `x` - The x-coordinate of the rectangle's bottom-left corner
    /// * `y` - The y-coordinate of the rectangle's bottom-left corner
    /// * `width` - The width of the rectangle in pixels
    /// * `height` - The height of the rectangle in pixels
    pub fn snap_rect(&self, out: &mut Vec<u8>, x: usize, y: usize, width: usize, height: usize) {
        self.canvas.read_pixels(out, x, y, width, height);
    }

    /// Captures the current framebuffer as an image.
    ///
    /// Returns an `ImageBuffer` containing the current screen content.
    /// The image is automatically flipped vertically to match the expected orientation
    /// (OpenGL's bottom-left origin is converted to top-left).
    ///
    /// # Returns
    /// An `ImageBuffer<Rgb<u8>, Vec<u8>>` containing the screen pixels
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let window = Window::new("Example").await;
    /// let image = window.snap_image();
    /// image.save("screenshot.png").unwrap();
    /// # }
    /// ```
    pub fn snap_image(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (width, height) = self.canvas.size();
        let mut buf = Vec::new();
        self.snap(&mut buf);
        let img_opt = ImageBuffer::from_vec(width, height, buf);
        let img = img_opt.expect("Buffer created from window was not big enough for image.");
        imageops::flip_vertical(&img)
    }

    /// Starts recording frames for a screencast with default settings.
    ///
    /// After calling this method, each frame rendered will be captured and stored.
    /// Call `end_recording` to stop recording and encode the frames to an MP4 video file.
    ///
    /// **Note:** This feature requires the `recording` feature to be enabled.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// window.begin_recording();
    /// // Render some frames...
    /// # for _ in 0..60 {
    /// #     window.render().await;
    /// # }
    /// window.end_recording("output.mp4", 30).unwrap();
    /// # }
    /// ```
    #[cfg(feature = "recording")]
    pub fn begin_recording(&mut self) {
        self.begin_recording_with_config(RecordingConfig::default());
    }

    /// Starts recording frames for a screencast with custom configuration.
    ///
    /// # Arguments
    /// * `config` - Recording configuration specifying frame skip, etc.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::{Window, RecordingConfig};
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// // Record every 2nd frame (reduces file size and encoding time)
    /// let config = RecordingConfig::new()
    ///     .with_frame_skip(2);
    /// window.begin_recording_with_config(config);
    /// # for _ in 0..60 {
    /// #     window.render().await;
    /// # }
    /// window.end_recording("output.mp4", 30).unwrap();
    /// # }
    /// ```
    #[cfg(feature = "recording")]
    pub fn begin_recording_with_config(&mut self, config: RecordingConfig) {
        let (width, height) = self.canvas.size();
        self.recording = Some(RecordingState {
            frames: Vec::new(),
            width,
            height,
            config,
            paused: false,
            frame_counter: 0,
        });
    }

    /// Returns whether recording is currently active.
    ///
    /// **Note:** This feature requires the `recording` feature to be enabled.
    #[cfg(feature = "recording")]
    pub fn is_recording(&self) -> bool {
        self.recording.is_some()
    }

    /// Returns whether recording is currently paused.
    ///
    /// **Note:** This feature requires the `recording` feature to be enabled.
    #[cfg(feature = "recording")]
    pub fn is_recording_paused(&self) -> bool {
        self.recording.as_ref().map_or(false, |r| r.paused)
    }

    /// Pauses the current recording.
    ///
    /// While paused, frames will not be captured. Call `resume_recording` to continue.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// window.begin_recording();
    /// // Record some frames...
    /// # for _ in 0..30 { window.render().await; }
    /// window.pause_recording();
    /// // These frames won't be recorded
    /// # for _ in 0..30 { window.render().await; }
    /// window.resume_recording();
    /// // Continue recording...
    /// # for _ in 0..30 { window.render().await; }
    /// window.end_recording("output.mp4", 30).unwrap();
    /// # }
    /// ```
    #[cfg(feature = "recording")]
    pub fn pause_recording(&mut self) {
        if let Some(ref mut recording) = self.recording {
            recording.paused = true;
        }
    }

    /// Resumes a paused recording.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// window.begin_recording();
    /// window.pause_recording();
    /// // ... do something without recording ...
    /// window.resume_recording();
    /// # window.end_recording("output.mp4", 30).unwrap();
    /// # }
    /// ```
    #[cfg(feature = "recording")]
    pub fn resume_recording(&mut self) {
        if let Some(ref mut recording) = self.recording {
            recording.paused = false;
        }
    }

    /// Stops recording and encodes the captured frames to an MP4 video file.
    ///
    /// This method consumes all recorded frames and encodes them using H.264 codec
    /// with proper compression via FFmpeg (through the `video-rs` crate).
    ///
    /// **Note:** This feature requires the `recording` feature to be enabled and
    /// FFmpeg libraries to be installed on the system.
    ///
    /// # Arguments
    /// * `path` - The output file path for the video (should end in `.mp4`)
    /// * `fps` - The frames per second for the output video
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(String)` with an error message if encoding fails
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// window.begin_recording();
    /// for _ in 0..120 {
    ///     // Animate your scene...
    ///     window.render().await;
    /// }
    /// // Save as 30fps video (120 frames = 4 seconds)
    /// window.end_recording("animation.mp4", 30).unwrap();
    /// # }
    /// ```
    #[cfg(feature = "recording")]
    pub fn end_recording<P: AsRef<Path>>(&mut self, path: P, fps: u32) -> Result<(), String> {
        use ffmpeg_the_third as ffmpeg;
        use ffmpeg::{
            codec, encoder, format, frame, software::scaling, Dictionary, Packet, Rational,
        };

        let recording = self
            .recording
            .take()
            .ok_or_else(|| "No recording in progress".to_string())?;

        if recording.frames.is_empty() {
            return Err("No frames were recorded".to_string());
        }

        let width = recording.width;
        let height = recording.height;

        // Initialize FFmpeg (safe to call multiple times)
        ffmpeg::init().map_err(|e| format!("Failed to initialize FFmpeg: {}", e))?;

        // Create output context
        let mut octx = format::output(&path)
            .map_err(|e| format!("Failed to create output context: {}", e))?;

        // Check if global header is required before borrowing octx mutably
        let global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);

        // Find H.264 encoder
        let codec = encoder::find(codec::Id::H264)
            .ok_or_else(|| "H.264 encoder not found. Install FFmpeg with libx264 support.".to_string())?;

        // Add video stream
        let mut ost = octx.add_stream(Some(codec))
            .map_err(|e| format!("Failed to add stream: {}", e))?;

        let ost_index = ost.index();

        // Configure encoder
        let mut encoder_ctx = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .map_err(|e| format!("Failed to create encoder context: {}", e))?;

        encoder_ctx.set_width(width);
        encoder_ctx.set_height(height);
        encoder_ctx.set_format(format::Pixel::YUV420P);
        encoder_ctx.set_time_base(Rational::new(1, fps as i32));
        encoder_ctx.set_frame_rate(Some(Rational::new(fps as i32, 1)));

        // Set global header flag if required by container format
        if global_header {
            encoder_ctx.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        // Open encoder with x264 preset
        let mut x264_opts = Dictionary::new();
        x264_opts.set("preset", "medium");
        x264_opts.set("crf", "23");
        let mut encoder = encoder_ctx
            .open_with(x264_opts)
            .map_err(|e| format!("Failed to open encoder: {}", e))?;

        // Set stream parameters from encoder
        ost.set_parameters(codec::Parameters::from(&encoder));

        // Write header
        octx.write_header()
            .map_err(|e| format!("Failed to write header: {}", e))?;

        // Create scaler to convert RGB24 to YUV420P
        let mut scaler = scaling::Context::get(
            format::Pixel::RGB24,
            width,
            height,
            format::Pixel::YUV420P,
            width,
            height,
            scaling::Flags::BILINEAR,
        ).map_err(|e| format!("Failed to create scaler: {}", e))?;

        let ost_time_base = octx.stream(ost_index).unwrap().time_base();

        // Encode each frame
        for (i, img_frame) in recording.frames.into_iter().enumerate() {
            // Create RGB frame from captured image
            let raw_data: Vec<u8> = img_frame.into_raw();

            let mut rgb_frame = frame::Video::new(format::Pixel::RGB24, width, height);
            rgb_frame.data_mut(0).copy_from_slice(&raw_data);

            // Scale to YUV420P
            let mut yuv_frame = frame::Video::empty();
            scaler.run(&rgb_frame, &mut yuv_frame)
                .map_err(|e| format!("Failed to scale frame: {}", e))?;

            // Set PTS (presentation timestamp)
            yuv_frame.set_pts(Some(i as i64));

            // Send frame to encoder
            encoder.send_frame(&yuv_frame)
                .map_err(|e| format!("Failed to send frame: {}", e))?;

            // Receive and write encoded packets
            let mut packet = Packet::empty();
            while encoder.receive_packet(&mut packet).is_ok() {
                packet.set_stream(ost_index);
                packet.rescale_ts(Rational::new(1, fps as i32), ost_time_base);
                packet.write_interleaved(&mut octx)
                    .map_err(|e| format!("Failed to write packet: {}", e))?;
            }
        }

        // Flush encoder
        encoder.send_eof()
            .map_err(|e| format!("Failed to send EOF: {}", e))?;

        let mut packet = Packet::empty();
        while encoder.receive_packet(&mut packet).is_ok() {
            packet.set_stream(ost_index);
            packet.rescale_ts(Rational::new(1, fps as i32), ost_time_base);
            packet.write_interleaved(&mut octx)
                .map_err(|e| format!("Failed to write packet: {}", e))?;
        }

        // Write trailer
        octx.write_trailer()
            .map_err(|e| format!("Failed to write trailer: {}", e))?;

        Ok(())
    }

    /// Captures the current frame if recording is active, not paused, and frame skip allows.
    ///
    /// This is called automatically during `render()` when recording is enabled.
    #[cfg(feature = "recording")]
    fn capture_frame_if_recording(&mut self) {
        // Check if we should capture this frame
        let should_capture = if let Some(ref mut recording) = self.recording {
            if recording.paused {
                false
            } else {
                recording.frame_counter += 1;
                // Capture if frame_counter matches the skip interval
                (recording.frame_counter - 1) % recording.config.frame_skip == 0
            }
        } else {
            false
        };

        if should_capture {
            let frame = self.snap_image();
            let (current_width, current_height) = self.canvas.size();

            // Now we can mutably borrow recording
            if let Some(ref mut recording) = self.recording {
                // Check if window was resized during recording
                if current_width != recording.width || current_height != recording.height {
                    // For now, we'll just capture at current size
                    // A more robust solution would resize frames or fail
                    recording.width = current_width;
                    recording.height = current_height;
                }
                recording.frames.push(frame);
            }
        }
    }

    /// Returns an event manager for accessing window events.
    ///
    /// The event manager provides an iterator over events that occurred since the last frame,
    /// such as keyboard input, mouse movement, and window resizing.
    ///
    /// # Returns
    /// An `EventManager` that can be iterated to process events
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # use kiss3d::event::{WindowEvent, Action, Key};
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// # while window.render().await {
    /// for event in window.events().iter() {
    ///     match event.value {
    ///         WindowEvent::Key(Key::Escape, Action::Release, _) => {
    ///             println!("Escape pressed!");
    ///         }
    ///         _ => {}
    ///     }
    /// }
    /// # }
    /// # }
    /// ```
    pub fn events(&self) -> EventManager {
        EventManager::new(self.events.clone(), self.unhandled_events.clone())
    }

    /// Gets the current state of a keyboard key.
    ///
    /// # Arguments
    /// * `key` - The key to check
    ///
    /// # Returns
    /// The current `Action` state (e.g., `Action::Press`, `Action::Release`)
    pub fn get_key(&self, key: Key) -> Action {
        self.canvas.get_key(key)
    }

    /// Gets the current state of a mouse button.
    ///
    /// # Arguments
    /// * `button` - The mouse button to check
    ///
    /// # Returns
    /// The current `Action` state (e.g., `Action::Press`, `Action::Release`)
    pub fn get_mouse_button(&self, button: MouseButton) -> Action {
        self.canvas.get_mouse_button(button)
    }

    /// Gets the last known position of the mouse cursor.
    ///
    /// The position is automatically updated when the mouse moves over the window.
    /// Coordinates are in pixels, with (0, 0) at the top-left corner.
    ///
    /// # Returns
    /// `Some((x, y))` with the cursor position, or `None` if the cursor position is unknown
    pub fn cursor_pos(&self) -> Option<(f64, f64)> {
        self.canvas.cursor_pos()
    }

    #[inline]
    fn handle_events(
        &mut self,
        camera: &mut Option<&mut dyn Camera>,
        planar_camera: &mut Option<&mut dyn PlanarCamera>,
    ) {
        let unhandled_events = self.unhandled_events.clone(); // FIXME: could we avoid the clone?
        let events = self.events.clone(); // FIXME: could we avoid the clone?

        for event in unhandled_events.borrow().iter() {
            self.handle_event(camera, planar_camera, event)
        }

        for event in events.try_iter() {
            self.handle_event(camera, planar_camera, &event)
        }

        unhandled_events.borrow_mut().clear();
        self.canvas.poll_events();
    }

    fn handle_event(
        &mut self,
        camera: &mut Option<&mut dyn Camera>,
        planar_camera: &mut Option<&mut dyn PlanarCamera>,
        event: &WindowEvent,
    ) {
        match *event {
            WindowEvent::Key(Key::Escape, Action::Release, _) | WindowEvent::Close => {
                self.close();
            }
            WindowEvent::FramebufferSize(w, h) => {
                self.update_viewport(w as f32, h as f32);
            }
            _ => {}
        }

        // Feed events to egui and check if it wants to capture input
        #[cfg(feature = "egui")]
        {
            self.feed_egui_event(event);

            if event.is_keyboard_event() && self.is_egui_capturing_keyboard() {
                return;
            }

            if event.is_mouse_event() && self.is_egui_capturing_mouse() {
                return;
            }
        }

        match *planar_camera {
            Some(ref mut cam) => cam.handle_event(&self.canvas, event),
            None => self.camera.borrow_mut().handle_event(&self.canvas, event),
        }

        match *camera {
            Some(ref mut cam) => cam.handle_event(&self.canvas, event),
            None => self.camera.borrow_mut().handle_event(&self.canvas, event),
        }
    }

    /// Renders one frame of the scene.
    ///
    /// This is the main rendering function that should be called in your render loop.
    /// It handles events, updates the scene, renders all objects, and swaps buffers.
    ///
    /// # Returns
    /// `true` if rendering should continue, `false` if the window should close
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// #[kiss3d::main]
    /// async fn main() {
    ///     let mut window = Window::new("My Application").await;
    ///     while window.render().await {
    ///         // Your per-frame code here
    ///     }
    /// }
    /// ```
    ///
    /// # Platform-specific
    /// - **Native**: Returns immediately after rendering one frame
    /// - **WASM**: Yields to the browser's event loop and returns when the next frame is ready
    pub async fn render(&mut self) -> bool {
        self.render_with(None, None, None, None).await
    }

    /// Renders one frame with a post-processing effect applied.
    ///
    /// # Arguments
    /// * `effect` - The post-processing effect to apply after rendering the scene
    ///
    /// # Returns
    /// `true` if rendering should continue, `false` if the window should close
    pub async fn render_with_effect(&mut self, effect: &mut dyn PostProcessingEffect) -> bool {
        self.render_with(None, None, Some(effect), None).await
    }

    /// Renders one frame using a custom 3D camera.
    ///
    /// # Arguments
    /// * `camera` - The camera to use for 3D rendering instead of the default camera
    ///
    /// # Returns
    /// `true` if rendering should continue, `false` if the window should close
    pub async fn render_with_camera(&mut self, camera: &mut dyn Camera) -> bool {
        self.render_with(Some(camera), None, None, None).await
    }

    /// Renders one frame using a custom 2D planar camera.
    ///
    /// # Arguments
    /// * `planar_camera` - The camera to use for 2D rendering
    ///
    /// # Returns
    /// `true` if rendering should continue, `false` if the window should close
    pub async fn render_with_planar_camera(
        &mut self,
        planar_camera: &mut dyn PlanarCamera,
    ) -> bool {
        self.render_with(None, Some(planar_camera), None, None)
            .await
    }

    /// Renders one frame using custom 2D and 3D cameras.
    ///
    /// # Arguments
    /// * `camera` - The camera to use for 3D rendering
    /// * `planar_camera` - The camera to use for 2D planar rendering
    ///
    /// # Returns
    /// `true` if rendering should continue, `false` if the window should close
    pub async fn render_with_cameras(
        &mut self,
        camera: &mut dyn Camera,
        planar_camera: &mut dyn PlanarCamera,
    ) -> bool {
        self.render_with(Some(camera), Some(planar_camera), None, None)
            .await
    }

    /// Renders one frame using a custom camera and post-processing effect.
    ///
    /// # Arguments
    /// * `camera` - The camera to use for 3D rendering
    /// * `effect` - The post-processing effect to apply
    ///
    /// # Returns
    /// `true` if rendering should continue, `false` if the window should close
    pub async fn render_with_camera_and_effect(
        &mut self,
        camera: &mut dyn Camera,
        effect: &mut dyn PostProcessingEffect,
    ) -> bool {
        self.render_with(Some(camera), None, Some(effect), None)
            .await
    }

    /// Renders one frame using custom 2D and 3D cameras with a post-processing effect.
    ///
    /// # Arguments
    /// * `camera` - The camera to use for 3D rendering
    /// * `planar_camera` - The camera to use for 2D planar rendering
    /// * `effect` - The post-processing effect to apply
    ///
    /// # Returns
    /// `true` if rendering should continue, `false` if the window should close
    pub async fn render_with_cameras_and_effect(
        &mut self,
        camera: &mut dyn Camera,
        planar_camera: &mut dyn PlanarCamera,
        effect: &mut dyn PostProcessingEffect,
    ) -> bool {
        self.render_with(Some(camera), Some(planar_camera), Some(effect), None)
            .await
    }

    /// Renders one frame with full customization options.
    ///
    /// This is the most flexible rendering method, allowing you to customize
    /// all aspects of the rendering pipeline.
    ///
    /// # Arguments
    /// * `camera` - Optional custom 3D camera
    /// * `planar_camera` - Optional custom 2D camera
    /// * `post_processing` - Optional post-processing effect
    /// * `renderer` - Optional custom renderer
    ///
    /// # Returns
    /// `true` if rendering should continue, `false` if the window should close
    pub async fn render_with(
        &mut self,
        camera: Option<&mut dyn Camera>,
        planar_camera: Option<&mut dyn PlanarCamera>,
        post_processing: Option<&mut dyn PostProcessingEffect>,
        renderer: Option<&mut dyn Renderer>,
    ) -> bool {
        // FIXME: for backward-compatibility, we don't accept any custom renderer here.
        self.do_render_with(camera, planar_camera, renderer, post_processing)
            .await
    }

    async fn do_render_with(
        &mut self,
        camera: Option<&mut dyn Camera>,
        planar_camera: Option<&mut dyn PlanarCamera>,
        renderer: Option<&mut dyn Renderer>,
        post_processing: Option<&mut dyn PostProcessingEffect>,
    ) -> bool {
        let mut camera = camera;
        let mut planar_camera = planar_camera;
        self.handle_events(&mut camera, &mut planar_camera);

        let self_cam2 = self.planar_camera.clone(); // FIXME: this is ugly.
        let self_cam = self.camera.clone(); // FIXME: this is ugly.

        match (camera, planar_camera) {
            (Some(cam), Some(cam2)) => {
                self.render_single_frame(cam, cam2, renderer, post_processing)
                    .await
            }
            (None, Some(cam2)) => {
                self.render_single_frame(
                    &mut *self_cam.borrow_mut(),
                    cam2,
                    renderer,
                    post_processing,
                )
                .await
            }
            (Some(cam), None) => {
                self.render_single_frame(
                    cam,
                    &mut *self_cam2.borrow_mut(),
                    renderer,
                    post_processing,
                )
                .await
            }
            (None, None) => {
                self.render_single_frame(
                    &mut *self_cam.borrow_mut(),
                    &mut *self_cam2.borrow_mut(),
                    renderer,
                    post_processing,
                )
                .await
            }
        }
    }

    async fn render_single_frame(
        &mut self,
        camera: &mut dyn Camera,
        planar_camera: &mut dyn PlanarCamera,
        mut renderer: Option<&mut dyn Renderer>,
        mut post_processing: Option<&mut dyn PostProcessingEffect>,
    ) -> bool {
        // XXX: too bad we have to do this at each frame…
        let w = self.width();
        let h = self.height();

        planar_camera.handle_event(&self.canvas, &WindowEvent::FramebufferSize(w, h));
        camera.handle_event(&self.canvas, &WindowEvent::FramebufferSize(w, h));
        planar_camera.update(&self.canvas);
        camera.update(&self.canvas);

        if let Light::StickToCamera = self.light_mode {
            self.set_light(Light::StickToCamera)
        }

        // Get the surface texture
        let frame = match self.canvas.get_current_texture() {
            Ok(frame) => frame,
            Err(e) => {
                eprintln!("Failed to acquire surface texture: {:?}", e);
                return !self.should_close();
            }
        };
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let ctxt = Context::get();
        let mut encoder = ctxt.create_command_encoder(Some("kiss3d_frame_encoder"));

        // Resize post-process render target if needed
        self.post_process_render_target
            .resize(w, h, self.canvas.surface_format());

        // Determine which views to render to
        let (color_view, depth_view) = if post_processing.is_some() {
            // Render to offscreen buffer for post-processing
            match &self.post_process_render_target {
                RenderTarget::Offscreen(o) => (&o.color_view, &o.depth_view),
                RenderTarget::Screen => {
                    // Shouldn't happen, but fallback to main view
                    (&frame_view, self.canvas.depth_view())
                }
            }
        } else {
            (&frame_view, self.canvas.depth_view())
        };
        let (color_view, depth_view) = (color_view.clone(), depth_view.clone());

        // Clear the render target at the start of the frame
        {
            let bg = &self.background;
            let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg.x as f64,
                            g: bg.y as f64,
                            b: bg.z as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // Render pass is dropped here, ending the clear pass
        }

        // Render the 3D scene
        for pass in 0usize..camera.num_passes() {
            camera.start_pass(pass, &self.canvas);

            {
                let mut render_context = RenderContext {
                    encoder: &mut encoder,
                    color_view: &color_view,
                    depth_view: &depth_view,
                    surface_format: self.canvas.surface_format(),
                    sample_count: self.canvas.sample_count(),
                    viewport_width: w,
                    viewport_height: h,
                };

                self.render_scene(camera, pass, &mut render_context);

                if let Some(ref mut renderer) = renderer {
                    renderer.render(pass, camera, &mut render_context);
                }
            }
        }

        camera.render_complete(&self.canvas);

        // Render the 2D planar scene
        {
            let mut planar_context = PlanarRenderContext {
                encoder: &mut encoder,
                color_view: &color_view,
                surface_format: self.canvas.surface_format(),
                sample_count: self.canvas.sample_count(),
                viewport_width: w,
                viewport_height: h,
            };

            self.render_planar_scene(planar_camera, &mut planar_context);
        }

        let (znear, zfar) = camera.clip_planes();

        // Apply post-processing if enabled
        if let Some(ref mut p) = post_processing {
            // FIXME: use the real time value instead of 0.016!
            p.update(0.016, w as f32, h as f32, znear, zfar);

            let mut pp_context = PostProcessingContext {
                encoder: &mut encoder,
                output_view: &frame_view,
            };

            p.draw(&self.post_process_render_target, &mut pp_context);
        }

        // Render text
        {
            let mut planar_context = PlanarRenderContext {
                encoder: &mut encoder,
                color_view: &frame_view,
                surface_format: self.canvas.surface_format(),
                sample_count: self.canvas.sample_count(),
                viewport_width: w,
                viewport_height: h,
            };
            self.text_renderer
                .render(w as f32, h as f32, &mut planar_context);
        }

        // Submit the main command buffer
        ctxt.submit(std::iter::once(encoder.finish()));

        // Render egui if enabled (uses its own command encoder and submits it)
        #[cfg(feature = "egui")]
        {
            self.egui_context.renderer.render(
                &frame_view,
                &depth_view,
                w,
                h,
                self.canvas.scale_factor() as f32,
            );
        }

        // Copy frame to readback texture for snap/snap_rect functionality
        self.canvas.copy_frame_to_readback(&frame);

        // Capture frame for video recording if enabled
        #[cfg(feature = "recording")]
        self.capture_frame_if_recording();

        // Present the frame
        self.canvas.present(frame);
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use web_sys::wasm_bindgen::closure::Closure;

            if let Some(window) = web_sys::window() {
                let (s, r) = oneshot::channel();

                let closure = Closure::once(move || s.send(()).unwrap());

                window
                    .request_animation_frame(closure.as_ref().unchecked_ref())
                    .unwrap();

                r.await.unwrap();
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Limit the fps if needed.
            if let Some(dur) = self.min_dur_per_frame {
                let elapsed = self.curr_time.elapsed();
                if elapsed < dur {
                    std::thread::sleep(dur - elapsed);
                }
            }

            self.curr_time = std::time::Instant::now();
        }

        // self.transparent_objects.clear();
        // self.opaque_objects.clear();

        !self.should_close()
    }

    fn render_scene(&mut self, camera: &mut dyn Camera, pass: usize, context: &mut RenderContext) {
        // Render points
        self.point_renderer.render(pass, camera, context);

        // Render polylines (lines with configurable width)
        self.polyline_renderer.render(pass, camera, context);

        // Render scene graph (surfaces and wireframes are handled by ObjectMaterial)
        self.scene
            .data_mut()
            .render(pass, camera, &self.light_mode, context);
    }

    fn render_planar_scene(
        &mut self,
        camera: &mut dyn PlanarCamera,
        context: &mut PlanarRenderContext,
    ) {
        if self.planar_polyline_renderer.needs_rendering() {
            self.planar_polyline_renderer.render(camera, context);
        }

        if self.planar_point_renderer.needs_rendering() {
            self.planar_point_renderer.render(camera, context);
        }

        self.scene2.data_mut().render(camera, context);
    }

    fn update_viewport(&mut self, _w: f32, _h: f32) {
        // In wgpu, viewport is set per render pass, not globally
        // The surface is automatically resized via canvas resize events
    }
}

fn init_wgpu() {
    // wgpu doesn't require explicit state initialization like OpenGL
    // All state (depth testing, culling, etc.) is specified per-pipeline
}
