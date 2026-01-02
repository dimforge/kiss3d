//! Structures responsible for rendering elements other than kiss3d's meshes.

#[cfg(feature = "egui")]
pub use self::egui_renderer::EguiRenderer;
pub use self::point_renderer3d::PointRenderer3d;
pub use self::polyline_renderer3d::{Polyline3d, PolylineRenderer3d};
pub use self::point_renderer2d::PointRenderer2d;
pub use self::polyline_renderer2d::{Polyline2d, PolylineRenderer2d};
pub use self::renderer::Renderer3d;

#[cfg(feature = "egui")]
mod egui_renderer;
pub mod point_renderer3d;
pub mod polyline_renderer3d;
pub mod point_renderer2d;
pub mod polyline_renderer2d;
mod renderer;
