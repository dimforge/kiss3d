//! DRM/KMS display output for headless windows (Linux only).
//!
//! A headless [`Window`](crate::window::Window) renders off-screen; this module
//! shows its frames on a display without a window system, by writing them into
//! dumb buffers on a separate thread.

mod card;
mod device;
mod display_thread;
mod error;
mod kms;
mod window;

pub use error::DrmError;
pub use window::DrmWindow;
