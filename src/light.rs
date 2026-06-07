//! Lighting configuration for 3D scenes.
//!
//! kiss3d supports multiple lights in the scene tree. Lights can be point lights,
//! directional lights, or spot lights, and they inherit transforms from their
//! parent scene nodes.

use crate::color::Color;
use glamx::Vec3;

/// Maximum number of lights supported in a scene.
pub const MAX_LIGHTS: usize = 8;

/// The type of light source.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LightType {
    /// A point light that emits light equally in all directions from a point.
    ///
    /// The light position comes from the scene node's world transform.
    Point {
        /// Maximum distance the light affects. Beyond this distance, the light
        /// contribution is zero.
        attenuation_radius: f32,
    },

    /// A directional light with parallel rays (like the sun).
    ///
    /// The light direction comes from the scene node's forward vector (-Z in local space).
    /// Position is ignored for directional lights.
    Directional(Vec3),

    /// A spot light that emits a cone of light.
    ///
    /// The light position comes from the scene node's world transform, and the
    /// direction comes from the forward vector (-Z in local space).
    Spot {
        /// Inner cone angle in radians. Full intensity within this cone.
        inner_cone_angle: f32,
        /// Outer cone angle in radians. Light fades to zero at this angle.
        outer_cone_angle: f32,
        /// Maximum distance the light affects.
        attenuation_radius: f32,
    },
}

impl Default for LightType {
    fn default() -> Self {
        LightType::Point {
            attenuation_radius: 100.0,
        }
    }
}

/// A light source that can be attached to a scene node.
///
/// The light's position and direction are determined by the scene node's
/// world transform.
///
/// # Examples
/// ```no_run
/// # use kiss3d::prelude::*;
/// # use glamx::Vec3;
/// // Create a white point light
/// let point_light = Light::point(100.0)
///     .with_color(WHITE)
///     .with_intensity(5.0);
///
/// // Create a directional "sun" light
/// let sun = Light::directional(Vec3::new(-1.0, -1.0, 0.0))
///     .with_color(Color::new(1.0, 0.95, 0.8, 1.0))
///     .with_intensity(2.0);
///
/// // Create a spot light (flashlight)
/// let spot = Light::spot(0.3, 0.5, 50.0)
///     .with_color(WHITE)
///     .with_intensity(10.0);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Light {
    /// The type of light (point, directional, or spot).
    pub light_type: LightType,
    /// The color of the light (RGBA, each component 0.0-1.0).
    pub color: Color,
    /// The intensity multiplier for the light.
    pub intensity: f32,
    /// Emitter sphere radius for soft shadows in the path tracer (0 = a hard
    /// point/spot light). Ignored by the rasterizer and by directional lights.
    pub radius: f32,
    /// Whether the light is enabled.
    pub enabled: bool,
    /// Whether this light casts shadows in the rasterization pipeline.
    ///
    /// When `true` (the default) and shadows are globally enabled on the window,
    /// the rasterizer renders a shadow map from this light's point of view and
    /// attenuates the light contribution of occluded fragments. Has no effect on
    /// the path tracer, which always computes ray-traced shadows.
    pub casts_shadows: bool,
    /// Light-layer bitmask (lighting channels). This light only affects an object
    /// when their masks share at least one bit (`object_layers & light_layers != 0`),
    /// the same idea as a per-light culling mask. Defaults to
    /// `u32::MAX` (every bit set), so by default a light affects every object. See
    /// [`Object3d::set_light_layers`](crate::scene::Object3d::set_light_layers).
    #[cfg_attr(feature = "serde", serde(default = "all_layers"))]
    pub layers: u32,
}

/// Default light-layer mask (all channels) — used by serde so scenes serialized
/// before the `layers` field deserialize as "affects every object" rather than `0`.
#[cfg(feature = "serde")]
fn all_layers() -> u32 {
    u32::MAX
}

impl Default for Light {
    fn default() -> Self {
        Self {
            light_type: LightType::default(),
            color: crate::color::WHITE,
            intensity: 3.0,
            radius: 0.0,
            enabled: true,
            casts_shadows: true,
            layers: u32::MAX,
        }
    }
}

impl Light {
    /// Creates a point light with the given attenuation radius.
    ///
    /// # Arguments
    /// * `attenuation_radius` - Maximum distance the light affects
    pub fn point(attenuation_radius: f32) -> Self {
        Self {
            light_type: LightType::Point { attenuation_radius },
            ..Default::default()
        }
    }

    /// Creates a directional light (like the sun).
    ///
    /// The direction is determined by the scene node's rotation.
    pub fn directional(dir: Vec3) -> Self {
        Self {
            light_type: LightType::Directional(dir),
            ..Default::default()
        }
    }

    /// Creates a spot light with the given cone angles and attenuation radius.
    ///
    /// # Arguments
    /// * `inner_cone_angle` - Inner cone angle in radians (full intensity)
    /// * `outer_cone_angle` - Outer cone angle in radians (fades to zero)
    /// * `attenuation_radius` - Maximum distance the light affects
    pub fn spot(inner_cone_angle: f32, outer_cone_angle: f32, attenuation_radius: f32) -> Self {
        Self {
            light_type: LightType::Spot {
                inner_cone_angle,
                outer_cone_angle,
                attenuation_radius,
            },
            ..Default::default()
        }
    }

    /// Sets the light color.
    ///
    /// # Arguments
    /// * `color` - RGBA color (each component 0.0-1.0)
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Sets the light intensity.
    ///
    /// # Arguments
    /// * `intensity` - Intensity multiplier (default: 3.0)
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }

    /// Sets whether the light is enabled.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Sets the emitter sphere radius for soft shadows in the path tracer.
    ///
    /// A radius above zero turns a point/spot light into a sphere sampled by the
    /// path tracer, producing soft shadow penumbrae. The rasterizer ignores it.
    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = radius.max(0.0);
        self
    }

    /// Sets whether this light casts shadows in the rasterization pipeline.
    ///
    /// Defaults to `true`. Disabling shadow casting for a light skips its shadow
    /// map render and makes it light occluded surfaces as if unobstructed, which
    /// is cheaper and can be useful for fill lights.
    pub fn with_casts_shadows(mut self, casts_shadows: bool) -> Self {
        self.casts_shadows = casts_shadows;
        self
    }

    /// Sets the light-layer bitmask (lighting channels).
    ///
    /// The light only affects objects whose own layer mask (see
    /// [`Object3d::set_light_layers`](crate::scene::Object3d::set_light_layers))
    /// shares at least one bit with `layers`. Defaults to `u32::MAX` (affects every
    /// object). Use this to confine a light to a subset of the scene.
    pub fn with_layers(mut self, layers: u32) -> Self {
        self.layers = layers;
        self
    }
}

/// A light that has been collected from the scene tree with its world-space transform.
#[derive(Clone, Debug)]
pub struct CollectedLight {
    /// The type of light.
    pub light_type: LightType,
    /// The light color.
    pub color: Vec3,
    /// The light intensity.
    pub intensity: f32,
    /// World-space position of the light.
    pub world_position: Vec3,
    /// World-space direction of the light (forward vector, -Z in local space).
    pub world_direction: Vec3,
    /// Emitter sphere radius for soft shadows (path tracer only).
    pub radius: f32,
    /// Whether this light should cast shadows in the rasterization pipeline.
    pub casts_shadows: bool,
    /// Light-layer bitmask (lighting channels); see [`Light::layers`].
    pub layers: u32,
}

/// Distance-fog falloff curve.
///
/// The common distance-fog falloff modes:
/// a `Linear` ramp between two distances, or physically-motivated `Exponential`
/// / `ExponentialSquared` density curves.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FogMode {
    /// Fog disabled.
    #[default]
    Off,
    /// Linear ramp: no fog before `start`, full fog past `end` (both view-space
    /// distances in world units).
    Linear { start: f32, end: f32 },
    /// Exponential falloff `1 - exp(-density * distance)`.
    Exponential { density: f32 },
    /// Exponential-squared falloff `1 - exp(-(density * distance)^2)`; denser, with
    /// a sharper onset.
    ExponentialSquared { density: f32 },
}

/// Distance fog applied to the rendered scene during shading.
///
/// Fog blends each shaded fragment toward [`color`](Self::color) by an amount
/// determined by [`mode`](Self::mode) and the fragment's view-space distance,
/// optionally thinned with altitude by [`height_falloff`](Self::height_falloff).
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fog {
    /// The fog/ambient color fragments are blended toward.
    pub color: Color,
    /// The falloff curve (and whether fog is active at all).
    pub mode: FogMode,
    /// Optional exponential thinning of fog with world-space height `y`
    /// (`0` disables it). Larger values clear the fog faster as you go up.
    pub height_falloff: f32,
}

impl Default for Fog {
    fn default() -> Self {
        Self {
            color: crate::color::Color::new(0.6, 0.7, 0.8, 1.0),
            mode: FogMode::Off,
            height_falloff: 0.0,
        }
    }
}

impl Fog {
    /// Linear fog ramping from `start` to `end` (view-space distances).
    pub fn linear(color: Color, start: f32, end: f32) -> Self {
        Self {
            color,
            mode: FogMode::Linear { start, end },
            height_falloff: 0.0,
        }
    }

    /// Exponential fog of the given density.
    pub fn exponential(color: Color, density: f32) -> Self {
        Self {
            color,
            mode: FogMode::Exponential { density },
            height_falloff: 0.0,
        }
    }

    /// Exponential-squared fog of the given density.
    pub fn exponential_squared(color: Color, density: f32) -> Self {
        Self {
            color,
            mode: FogMode::ExponentialSquared { density },
            height_falloff: 0.0,
        }
    }

    /// Sets the height falloff (exponential thinning of fog with world height).
    pub fn with_height_falloff(mut self, height_falloff: f32) -> Self {
        self.height_falloff = height_falloff.max(0.0);
        self
    }

    /// GPU-friendly encoding: `(mode_code, param_a, param_b, height_falloff)`.
    /// `mode_code` is 0 off / 1 linear / 2 exp / 3 exp2; for linear `param_a/b`
    /// are start/end, otherwise `param_a` is the density.
    pub(crate) fn params(&self) -> [f32; 4] {
        match self.mode {
            FogMode::Off => [0.0, 0.0, 0.0, 0.0],
            FogMode::Linear { start, end } => [1.0, start, end, self.height_falloff],
            FogMode::Exponential { density } => [2.0, density, 0.0, self.height_falloff],
            FogMode::ExponentialSquared { density } => [3.0, density, 0.0, self.height_falloff],
        }
    }
}

/// A collection of lights gathered from the scene tree during the prepare phase.
#[derive(Clone, Debug)]
pub struct LightCollection {
    /// The collected lights with their world-space transforms.
    pub lights: Vec<CollectedLight>,
    /// Global ambient lighting intensity.
    pub ambient: f32,
    /// Global ambient light color (multiplied by [`ambient`](Self::ambient)).
    pub ambient_color: Color,
    /// Distance fog applied to the scene during shading.
    pub fog: Fog,
}

impl Default for LightCollection {
    fn default() -> Self {
        Self::new()
    }
}

impl LightCollection {
    /// Creates a new empty light collection with default ambient.
    pub fn new() -> Self {
        Self {
            lights: Vec::with_capacity(MAX_LIGHTS),
            ambient: 0.2,
            ambient_color: crate::color::WHITE,
            fog: Fog::default(),
        }
    }

    /// Creates a new light collection with the specified ambient intensity.
    pub fn with_ambient(ambient: f32) -> Self {
        Self {
            lights: Vec::with_capacity(MAX_LIGHTS),
            ambient,
            ambient_color: crate::color::WHITE,
            fog: Fog::default(),
        }
    }

    /// Adds a light to the collection.
    ///
    /// All lights are collected; the renderer later splits them into the fixed
    /// "primary" tier (rendered through the uniform array with shadows) and the
    /// "clustered" overflow tier via [`split_primary_clustered`](Self::split_primary_clustered).
    /// Always returns `true` (the bool is kept for backwards compatibility).
    pub fn add(&mut self, light: CollectedLight) -> bool {
        self.lights.push(light);
        true
    }

    /// Returns `true` if the collection has reached the [`MAX_LIGHTS`] primary budget.
    ///
    /// This no longer prevents further lights from being collected — extra lights
    /// spill into the clustered tier — but remains useful to detect when the cheap
    /// shadow-capable primary slots are exhausted.
    pub fn is_full(&self) -> bool {
        self.lights.len() >= MAX_LIGHTS
    }

    /// Splits the collected lights into the primary and clustered tiers.
    ///
    /// The **primary** tier (at most [`MAX_LIGHTS`] lights) is rendered through the
    /// fixed uniform array and keeps full shadow-mapping support. The **clustered**
    /// tier holds the overflow, shaded by the clustered forward+ path without shadows.
    ///
    /// Returns two lists of indices into [`lights`](Self::lights). Both the object
    /// material's uniform upload and the shadow atlas must consume the **same**
    /// primary ordering so that uniform slot `i` and shadow view `i` refer to the
    /// same light.
    ///
    /// Selection: when the scene has `<= MAX_LIGHTS` lights, every light is primary
    /// in collection order (byte-identical to the legacy fixed path). Otherwise the
    /// primary slots go to shadow-casting lights first, then directional lights, then
    /// the remaining lights by descending intensity (collection order breaks ties).
    pub fn split_primary_clustered(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.lights.len();
        if n <= MAX_LIGHTS {
            return ((0..n).collect(), Vec::new());
        }

        // Lower key sorts earlier (higher priority for a primary slot).
        fn tier(l: &CollectedLight) -> u8 {
            if l.casts_shadows {
                0
            } else if matches!(l.light_type, LightType::Directional(_)) {
                1
            } else {
                2
            }
        }

        let mut order: Vec<usize> = (0..n).collect();
        // `sort_by` is stable, so equal-priority lights keep their collection order.
        order.sort_by(|&a, &b| {
            let (la, lb) = (&self.lights[a], &self.lights[b]);
            tier(la).cmp(&tier(lb)).then_with(|| {
                lb.intensity
                    .partial_cmp(&la.intensity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });

        let clustered = order.split_off(MAX_LIGHTS);
        (order, clustered)
    }

    /// Returns the number of lights in the collection.
    pub fn len(&self) -> usize {
        self.lights.len()
    }

    /// Returns `true` if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.lights.is_empty()
    }

    /// Clears all lights from the collection.
    pub fn clear(&mut self) {
        self.lights.clear();
    }
}
