//! Camera trait with some common implementations.

pub use self::camera2d::Camera2d;
pub use self::camera3d::Camera3d;
pub use self::first_person3d::FirstPersonCamera3d;
pub use self::first_person_stereo3d::FirstPersonCamera3dStereo;
pub use self::fixed_view2d::{CoordinateSystem2d, FixedView2d};
pub use self::fixed_view3d::FixedView3d;
pub use self::orbit3d::OrbitCamera3d;
pub use self::sidescroll2d::PanZoomCamera2d;

/// The projection a 3D camera uses to map view space to clip space.
///
/// Provides the two standard projections: a
/// perspective frustum, or a parallel orthographic box useful for CAD-style
/// inspection where parallel lines must stay parallel.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Projection {
    /// Perspective projection driven by the camera's field of view.
    Perspective,
    /// Orthographic (parallel) projection. `scale` is reserved for future
    /// per-camera control; the built-in `OrbitCamera3d` derives the orthographic
    /// half-height from its orbit distance so zooming keeps working.
    Orthographic,
}

impl Default for Projection {
    fn default() -> Self {
        Projection::Perspective
    }
}

/// Physically-based camera exposure, expressed as an EV100 value.
///
/// The scene's linear HDR radiance is scaled by
/// [`exposure`](Self::exposure) before tonemapping. Build one from photographic
/// settings with [`from_physical`](Self::from_physical) (aperture in f-stops,
/// shutter speed in seconds, ISO sensitivity), or set [`ev100`](Self::ev100)
/// directly.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Exposure {
    /// Exposure value at ISO 100. Larger values darken the image (one stop per
    /// unit).
    pub ev100: f32,
}

impl Default for Exposure {
    fn default() -> Self {
        // Neutral: 2^0 * 1.2 ≈ 1.2 → exposure ≈ 0.833. A physically-based default
        // would darken kiss3d's existing look, so default to a unit multiplier.
        Exposure::from_exposure(1.0)
    }
}

impl Exposure {
    /// A bright outdoor "sunlight" exposure (EV100 = 15), as a reference point.
    pub const SUNLIGHT: f32 = 15.0;
    /// An overcast-daylight exposure (EV100 = 12).
    pub const OVERCAST: f32 = 12.0;
    /// A dim indoor exposure (EV100 = 7).
    pub const INDOOR: f32 = 7.0;

    /// Builds an exposure from photographic settings.
    ///
    /// * `aperture_f_stops` — lens aperture (e.g. `4.0` for f/4).
    /// * `shutter_speed_s` — shutter open time in seconds (e.g. `1.0/250.0`).
    /// * `sensitivity_iso` — sensor ISO sensitivity (e.g. `100.0`).
    pub fn from_physical(aperture_f_stops: f32, shutter_speed_s: f32, sensitivity_iso: f32) -> Self {
        let ev100 = ((aperture_f_stops * aperture_f_stops) / shutter_speed_s
            * 100.0
            / sensitivity_iso)
            .log2();
        Exposure { ev100 }
    }

    /// Builds an exposure directly from a linear multiplier (the inverse of
    /// [`exposure`](Self::exposure)).
    pub fn from_exposure(exposure: f32) -> Self {
        // exposure = 1 / (2^ev100 * 1.2)  ⇒  ev100 = log2(1 / (exposure * 1.2)).
        Exposure {
            ev100: (1.0 / (exposure.max(1e-6) * 1.2)).log2(),
        }
    }

    /// The linear multiplier applied to scene radiance before tonemapping.
    pub fn exposure(&self) -> f32 {
        1.0 / (2.0f32.powf(self.ev100) * 1.2)
    }
}

mod camera2d;
mod camera3d;
mod first_person3d;
mod first_person_stereo3d;
mod fixed_view2d;
mod fixed_view3d;
mod orbit3d;
mod sidescroll2d;
