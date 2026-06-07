//! Structures responsible for rendering elements other than kiss3d's meshes.

#[cfg(feature = "egui")]
pub use self::egui_renderer::EguiRenderer;
pub use self::ibl::EnvironmentMap;
pub use self::point_renderer2d::PointRenderer2d;
pub use self::point_renderer3d::PointRenderer3d;
pub use self::polyline_renderer2d::{Polyline2d, PolylineRenderer2d};
pub use self::reflector::{MirrorCamera, Reflector};
pub use self::polyline_renderer3d::{Polyline3d, PolylineRenderer3d};
pub use self::raytracer::{RayBackend, RayTracer};
pub use self::reflection_probe::{
    CubeFaceCamera, ProbeCapture, ReflectionProbe, ReflectionProbes, MAX_PROBES,
};
pub use self::renderer::Renderer3d;
pub use self::skybox::Skybox;
pub use self::ssao::{Ssao, SsaoSettings};
pub use self::ssr::{Ssr, SsrMaterial, SsrSettings};
pub use self::timings::RenderTimings;

#[cfg(feature = "egui")]
mod egui_renderer;
mod ibl;
pub mod point_renderer2d;
pub mod point_renderer3d;
pub mod polyline_renderer2d;
pub mod polyline_renderer3d;
pub mod reflector;
pub mod raytracer;
pub mod reflection_probe;
mod renderer;
mod skybox;
mod ssao;
mod ssr;
pub mod timings;
