use crate::event::WindowEvent;
use crate::window::Canvas;
use glamx::{Mat4, Pose3, Vec2, Vec3, Vec4, Vec4Swizzles};

/// Trait that all camera implementations must implement.
///
/// Cameras control the viewpoint from which the 3D scene is rendered.
/// This trait defines the interface for event handling, transformations,
/// and rendering pipeline integration.
///
/// # Implementations
/// kiss3d provides several built-in camera types:
/// - [`OrbitCamera3d`](crate::camera::OrbitCamera3d) - Orbital camera (default)
/// - [`FirstPersonCamera3d`](crate::camera::FirstPersonCamera3d) - FPS-style camera
/// - [`FixedView`](crate::camera::FixedView) - Static camera with fixed view
///
/// # Custom Cameras
/// You can implement this trait to create custom camera behaviors.
pub trait Camera3d {
    // ==================
    // Event handling
    // ==================

    /// Handles window events to update camera state.
    ///
    /// This is called for each window event (mouse, keyboard, etc.) and allows
    /// the camera to respond to user input.
    ///
    /// # Arguments
    /// * `canvas` - Reference to the rendering canvas
    /// * `event` - The window event to handle
    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent);

    // ==================
    // Transformation-related methods
    // ==================

    /// Returns the camera's position in world space.
    ///
    /// # Returns
    /// The 3D point representing the camera's eye position
    fn eye(&self) -> Vec3;

    /// Returns the camera's view transformation.
    ///
    /// This is the inverse of the camera's world transformation and is used
    /// to transform world coordinates into camera/view space.
    ///
    /// # Returns
    /// An isometry (rotation + translation) representing the view transform
    fn view_transform(&self) -> Pose3;

    /// Returns the combined projection and view transformation matrix.
    ///
    /// This matrix transforms points from world coordinates to normalized device coordinates (NDC).
    /// It combines both the view transformation (world → camera space) and
    /// the projection transformation (camera space → NDC).
    ///
    /// # Returns
    /// A 4x4 transformation matrix
    fn transformation(&self) -> Mat4;

    /// Returns the inverse of the combined transformation matrix.
    ///
    /// This matrix transforms points from normalized device coordinates back to world coordinates.
    /// It's the inverse of [`transformation()`](Self::transformation).
    ///
    /// # Returns
    /// A 4x4 inverse transformation matrix
    fn inverse_transformation(&self) -> Mat4;

    /// Returns the near and far clipping plane distances.
    ///
    /// Objects closer than `znear` or farther than `zfar` from the camera are not rendered.
    ///
    /// # Returns
    /// A tuple `(znear, zfar)` with the clipping plane distances
    fn clip_planes(&self) -> (f32, f32);

    // ==================
    // Update & upload
    // ==================

    /// Updates the camera state for the current frame.
    ///
    /// This is called once at the beginning of each frame, before rendering.
    /// Use this to update internal camera state based on the canvas size or other factors.
    ///
    /// # Arguments
    /// * `canvas` - Reference to the rendering canvas
    fn update(&mut self, canvas: &Canvas);

    /// Returns the view and projection matrices for a given rendering pass.
    ///
    /// This method provides the matrices that materials use to transform objects
    /// for rendering. For single-pass cameras, pass 0 is typically the only pass.
    /// For stereo cameras, pass 0 might be the left eye and pass 1 the right eye.
    ///
    /// # Arguments
    /// * `pass` - The current rendering pass index
    ///
    /// # Returns
    /// A tuple `(view_transform, projection_matrix)` where:
    /// - `view_transform` is the camera's view transformation (world → camera space)
    /// - `projection_matrix` is the projection matrix (camera space → NDC)
    fn view_transform_pair(&self, pass: usize) -> (Pose3, Mat4);

    /// Returns the number of rendering passes required by this camera.
    ///
    /// Most cameras require only a single pass. Stereo cameras might require two passes
    /// (one for each eye).
    ///
    /// # Returns
    /// The number of rendering passes (default: 1)
    #[inline]
    fn num_passes(&self) -> usize {
        1usize
    }

    /// Called at the start of each rendering pass.
    ///
    /// Override this to perform per-pass setup (e.g., setting viewport for stereo rendering).
    ///
    /// # Arguments
    /// * `pass` - The index of the pass being started
    /// * `canvas` - Reference to the rendering canvas
    #[inline]
    fn start_pass(&self, _pass: usize, _canvas: &Canvas) {}

    /// Called after the scene has been rendered, before post-processing.
    ///
    /// Override this to perform cleanup or additional rendering steps.
    ///
    /// # Arguments
    /// * `canvas` - Reference to the rendering canvas
    #[inline]
    fn render_complete(&self, _canvas: &Canvas) {}

    /// Projects a 3D point in world coordinates to 2D screen coordinates.
    ///
    /// # Arguments
    /// * `world_coord` - The 3D point in world space
    /// * `size` - The size of the screen/viewport in pixels
    ///
    /// # Returns
    /// A 2D vector with screen coordinates (in pixels, origin at top-left)
    fn project(&self, world_coord: Vec3, size: Vec2) -> Vec2 {
        let h_world_coord = world_coord.extend(1.0);
        let h_normalized_coord = self.transformation() * h_world_coord;

        let normalized_coord = h_normalized_coord.xyz() / h_normalized_coord.w;

        Vec2::new(
            (1.0 + normalized_coord.x) * size.x / 2.0,
            (1.0 + normalized_coord.y) * size.y / 2.0,
        )
    }

    /// Unprojects a 2D screen point to a 3D ray in world space.
    ///
    /// Converts a point on the screen (in pixels) to a ray starting at the camera
    /// and passing through that screen point. This is useful for mouse picking and
    /// ray casting.
    ///
    /// # Arguments
    /// * `window_coord` - The 2D point in screen coordinates (origin at top-left)
    /// * `size` - The size of the screen/viewport in pixels
    ///
    /// # Returns
    /// A tuple `(origin, direction)` where:
    /// - `origin` is the start point of the ray (typically the camera position)
    /// - `direction` is the normalized direction vector of the ray
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::prelude::*;
    /// # let camera = OrbitCamera3d::new(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO);
    /// let mouse_pos = Vec2::new(400.0, 300.0);
    /// let screen_size = Vec2::new(800.0, 600.0);
    /// let (ray_origin, ray_dir) = camera.unproject(mouse_pos, screen_size);
    /// // Now you can use the ray for picking objects in the scene
    /// ```
    fn unproject(&self, window_coord: Vec2, size: Vec2) -> (Vec3, Vec3) {
        let normalized_coord = Vec2::new(
            2.0 * window_coord.x / size.x - 1.0,
            2.0 * -window_coord.y / size.y + 1.0,
        );

        let normalized_begin = Vec4::new(normalized_coord.x, normalized_coord.y, -1.0, 1.0);
        let normalized_end = Vec4::new(normalized_coord.x, normalized_coord.y, 1.0, 1.0);

        let cam = self.inverse_transformation();

        let h_unprojected_begin = cam * normalized_begin;
        let h_unprojected_end = cam * normalized_end;

        let unprojected_begin = h_unprojected_begin.xyz() / h_unprojected_begin.w;
        let unprojected_end = h_unprojected_end.xyz() / h_unprojected_end.w;

        (
            unprojected_begin,
            (unprojected_end - unprojected_begin).normalize(),
        )
    }
}
