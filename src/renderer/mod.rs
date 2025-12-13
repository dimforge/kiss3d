//! Structures responsible for rendering elements other than kiss3d's meshes.

#[cfg(feature = "egui")]
pub use self::egui_renderer::EguiRenderer;
pub use self::point_renderer::PointRenderer;
pub use self::polyline_renderer::{Polyline, PolylineRenderer};
pub use self::renderer::Renderer;

#[cfg(feature = "egui")]
mod egui_renderer;
pub mod point_renderer;
pub mod polyline_renderer;
mod renderer;
