//! The window, and things to handle the rendering loop and events.

mod aov;
mod canvas;
mod drawing;
#[cfg(all(feature = "drm", target_os = "linux"))]
mod drm;
#[cfg(feature = "egui")]
mod egui_integration;
mod events;
#[cfg(feature = "egui")]
mod inspector;
#[cfg(target_os = "ios")]
mod ios;
mod offscreen;
#[cfg(feature = "recording")]
mod recording;
mod rendering;
mod screenshot;
mod wgpu_canvas;
mod window;
mod window_cache;

pub use canvas::{Canvas, CanvasSetup, NumSamples};
#[cfg(all(feature = "drm", target_os = "linux"))]
pub use drm::{DrmError, DrmWindow};
#[cfg(feature = "egui")]
pub use inspector::{Inspector, InspectorTab};
#[cfg(target_os = "ios")]
pub use ios::run_ios;
pub use offscreen::OffscreenSurface;
#[cfg(feature = "recording")]
pub use recording::RecordingConfig;
#[cfg(target_os = "android")]
pub use wgpu_canvas::init_android;
pub use wgpu_canvas::WgpuCanvas;
pub use window::Window;
pub(crate) use window_cache::WINDOW_CACHE;
