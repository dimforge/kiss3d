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
    /// Whether the light is enabled.
    pub enabled: bool,
}

impl Default for Light {
    fn default() -> Self {
        Self {
            light_type: LightType::default(),
            color: crate::color::WHITE,
            intensity: 3.0,
            enabled: true,
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
}

/// A collection of lights gathered from the scene tree during the prepare phase.
#[derive(Clone, Debug)]
pub struct LightCollection {
    /// The collected lights with their world-space transforms.
    pub lights: Vec<CollectedLight>,
    /// Global ambient lighting intensity.
    pub ambient: f32,
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
        }
    }

    /// Creates a new light collection with the specified ambient intensity.
    pub fn with_ambient(ambient: f32) -> Self {
        Self {
            lights: Vec::with_capacity(MAX_LIGHTS),
            ambient,
        }
    }

    /// Adds a light to the collection if there's room.
    ///
    /// Returns `true` if the light was added, `false` if the collection is full.
    pub fn add(&mut self, light: CollectedLight) -> bool {
        if self.lights.len() < MAX_LIGHTS {
            self.lights.push(light);
            true
        } else {
            false
        }
    }

    /// Returns `true` if the collection has reached the maximum number of lights.
    pub fn is_full(&self) -> bool {
        self.lights.len() >= MAX_LIGHTS
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
