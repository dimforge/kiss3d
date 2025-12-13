//! The window, and things to handle the rendering loop and events.

mod canvas;
mod wgpu_canvas;
mod window;
mod window_cache;

pub use canvas::{Canvas, CanvasSetup, NumSamples};
pub use wgpu_canvas::WgpuCanvas;
pub use window::Window;
#[cfg(feature = "recording")]
pub use window::RecordingConfig;
pub(crate) use window_cache::WINDOW_CACHE;
