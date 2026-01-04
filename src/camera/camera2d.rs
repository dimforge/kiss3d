use crate::event::WindowEvent;
use crate::window::Canvas;
use glamx::{Mat3, Vec2};

/// Trait that all 2D camera implementations must implement.
///
/// 2D cameras control the view for 2D overlays and scene elements.
/// Unlike 3D cameras, 2D cameras work with 2D transformations and projections.
///
/// # Implementations
/// kiss3d provides built-in 2D camera types:
/// - [`FixedView2d`](crate::camera::FixedView2d) - Static 2D camera
/// - [`PanZoomCamera2d`](crate::camera::PanZoomCamera2d) - Side-scrolling camera
pub trait Camera2d {
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
    fn view_transform_pair(&self) -> (Mat3, Mat3);

    /// Converts screen coordinates to 2D world coordinates.
    ///
    /// # Arguments
    /// * `window_coord` - The point in screen space (pixels)
    /// * `window_size` - The size of the window in pixels
    ///
    /// # Returns
    /// The corresponding point in 2D world space
    fn unproject(&self, window_coord: Vec2, window_size: Vec2) -> Vec2;
}
