//! Lighting configuration for 3D scenes.
//!
//! kiss3d currently supports a single light source. The light can either be
//! fixed in world space or attached to the camera.

use na::Point3;

/// The light configuration for a scene.
///
/// Currently, kiss3d supports only one light source per scene.
/// The light affects how 3D objects are shaded and rendered.
///
/// # Examples
/// ```no_run
/// # use kiss3d::window::Window;
/// # use kiss3d::light::Light;
/// # use nalgebra::Point3;
/// # #[kiss3d::main]
/// # async fn main() {
/// # let mut window = Window::new("Example").await;
/// // Light that follows the camera
/// window.set_light(Light::StickToCamera);
///
/// // Light at a fixed position in the world
/// window.set_light(Light::Absolute(Point3::new(10.0, 10.0, 10.0)));
/// # }
/// ```
#[derive(Clone)]
pub enum Light {
    /// A light positioned at a fixed location in world space.
    ///
    /// The light remains at this position regardless of camera movement.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::light::Light;
    /// # use nalgebra::Point3;
    /// // Light above and to the right of the origin
    /// let light = Light::Absolute(Point3::new(5.0, 10.0, 5.0));
    /// ```
    Absolute(Point3<f32>),

    /// A light that follows the camera position.
    ///
    /// The light source moves with the camera, ensuring objects are always
    /// illuminated from the viewer's perspective. This is the default lighting mode.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # use kiss3d::light::Light;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// window.set_light(Light::StickToCamera);
    /// # }
    /// ```
    StickToCamera,
}
