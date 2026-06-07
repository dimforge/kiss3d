//! The window, and things to handle the rendering loop and events.

mod aov;
mod canvas;
mod drawing;
#[cfg(feature = "egui")]
mod egui_integration;
mod events;
#[cfg(feature = "egui")]
mod inspector;
#[cfg(not(target_arch = "wasm32"))]
mod offscreen;
#[cfg(feature = "recording")]
mod recording;
mod rendering;
mod screenshot;
mod wgpu_canvas;
mod window;
mod window_cache;

pub use canvas::{Canvas, CanvasSetup, NumSamples};
#[cfg(feature = "egui")]
pub use inspector::Inspector;
#[cfg(not(target_arch = "wasm32"))]
pub use offscreen::OffscreenSurface;
#[cfg(feature = "recording")]
pub use recording::RecordingConfig;
pub use wgpu_canvas::WgpuCanvas;
pub use window::Window;
pub(crate) use window_cache::WINDOW_CACHE;
