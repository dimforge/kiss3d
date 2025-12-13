use crate::event::WindowEvent;
use crate::window::Canvas;
use na::{Matrix3, Point2, Vector2};

/// Trait that all 2D camera implementations must implement.
///
/// Planar cameras control the view for 2D overlays and planar scene elements.
/// Unlike 3D cameras, planar cameras work with 2D transformations and projections.
///
/// # Implementations
/// kiss3d provides built-in 2D camera types:
/// - [`PlanarFixedView`](crate::planar_camera::PlanarFixedView) - Static 2D camera
/// - [`Sidescroll`](crate::planar_camera::Sidescroll) - Side-scrolling camera
pub trait PlanarCamera {
    /// Handles window events to update camera state.
    ///
    /// Called for each window event, allowing the camera to respond to user input.
    ///
    /// # Arguments
    /// * `canvas` - Reference to the rendering canvas
    /// * `event` - The window event to handle
    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent);

    /// Updates the camera state for the current frame.
    ///
    /// Called once at the beginning of each frame before rendering.
    ///
    /// # Arguments
    /// * `canvas` - Reference to the rendering canvas
    fn update(&mut self, canvas: &Canvas);

    /// Returns the view and projection matrices for 2D rendering.
    ///
    /// This method provides the matrices that materials use to transform 2D objects.
    ///
    /// # Returns
    /// A tuple `(view_matrix, projection_matrix)` where:
    /// - `view_matrix` is the camera's view transformation
    /// - `projection_matrix` is the projection matrix
    fn view_transform_pair(&self) -> (Matrix3<f32>, Matrix3<f32>);

    /// Converts screen coordinates to 2D world coordinates.
    ///
    /// # Arguments
    /// * `window_coord` - The point in screen space (pixels)
    /// * `window_size` - The size of the window in pixels
    ///
    /// # Returns
    /// The corresponding point in 2D world space
    fn unproject(&self, window_coord: &Point2<f32>, window_size: &Vector2<f32>) -> Point2<f32>;
}
