//! Errors of the DRM/KMS display layer.

use std::error::Error;
use std::fmt;

/// Error type for DRM display operations.
#[derive(Debug)]
pub enum DrmError {
    /// The DRM device is unusable (e.g. no dumb-buffer support).
    Init(String),
    /// A DRM ioctl failed while querying resources.
    Drm(String),
    /// No usable CRTC or display mode could be configured.
    Modeset(String),
    /// Opening or reading the device node failed.
    Io(std::io::Error),
    /// No connector reports a connected display.
    NoConnectedDisplay,
    /// No `/dev/dri/card*` device could drive a display. Holds one `path: reason` entry
    /// per device tried (empty when there are no candidates).
    NoUsableDevice(Vec<String>),
    /// The display thread has stopped, so frames can no longer be shown.
    DisplayThreadStopped,
}

impl fmt::Display for DrmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrmError::Init(msg) => write!(f, "Initialization error: {}", msg),
            DrmError::Drm(msg) => write!(f, "DRM error: {}", msg),
            DrmError::Modeset(msg) => write!(f, "Display configuration error: {}", msg),
            DrmError::Io(e) => write!(f, "I/O error: {}", e),
            DrmError::NoConnectedDisplay => write!(f, "No connected display found"),
            DrmError::NoUsableDevice(failures) if failures.is_empty() => {
                write!(f, "No DRM device found in /dev/dri")
            }
            DrmError::NoUsableDevice(failures) => {
                write!(f, "No usable DRM device ({})", failures.join("; "))
            }
            DrmError::DisplayThreadStopped => write!(f, "The display thread has stopped"),
        }
    }
}

impl Error for DrmError {}

impl From<std::io::Error> for DrmError {
    fn from(err: std::io::Error) -> Self {
        DrmError::Io(err)
    }
}
