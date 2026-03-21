//! The window, and things to handle the rendering loop and events.

mod canvas;
mod drawing;
#[cfg(feature = "drm")]
mod drm;
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
mod window_common;

pub use canvas::{Canvas, CanvasInputState, CanvasSetup, NumSamples};
#[cfg(feature = "recording")]
pub use recording::RecordingConfig;
pub use wgpu_canvas::WgpuCanvas;
pub(crate) use window_cache::WINDOW_CACHE;
pub use window_common::Window;
