//! The window, and things to handle the rendering loop and events.

mod canvas;
mod drawing;
#[cfg(feature = "drm")]
mod drm;
#[cfg(feature = "egui")]
mod egui_integration;
#[cfg(not(feature = "drm"))]
mod events;
#[cfg(feature = "recording")]
mod recording;
mod rendering;
mod screenshot;
mod wgpu_canvas;
#[cfg(not(feature = "drm"))]
mod window;
mod window_cache;

pub use canvas::{Canvas, CanvasInputState, CanvasSetup, NumSamples};
#[cfg(feature = "drm")]
pub use drm::Window;
#[cfg(feature = "recording")]
pub use recording::RecordingConfig;
pub use wgpu_canvas::WgpuCanvas;
#[cfg(not(feature = "drm"))]
pub use window::Window;
pub(crate) use window_cache::WINDOW_CACHE;
