//! The window, and things to handle the rendering loop and events.

mod canvas;
mod drawing;
#[cfg(feature = "egui")]
mod egui_integration;
mod events;
#[cfg(feature = "recording")]
mod recording;
mod rendering;
mod screenshot;
mod wgpu_canvas;
mod window;
mod window_cache;

pub use canvas::{Canvas, CanvasSetup, NumSamples};
#[cfg(feature = "recording")]
pub use recording::RecordingConfig;
pub use wgpu_canvas::WgpuCanvas;
pub use window::Window;
pub(crate) use window_cache::WINDOW_CACHE;
