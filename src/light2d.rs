//! Dynamic lights for 2D scenes.
//!
//! 2D lights illuminate objects drawn with
//! [`LitMaterial2d`](crate::builtin::LitMaterial2d). A light lives slightly *above*
//! the 2D plane (its [`height`](Light2d::height)), so a normal-mapped sprite reacts
//! to it with diffuse and specular shading just like a 3D surface would. Without a
//! normal map a sprite is treated as flat (facing the camera) and the light still
//! contributes a smooth radial falloff.
//!
//! The active lights and ambient term are stored in a thread-local
//! [`Light2dManager`]; populate it each frame before rendering a lit 2D scene.

use crate::color::Color;
use glamx::Vec2;
use std::cell::RefCell;

/// Maximum number of simultaneous 2D lights (the lit shader stores them in a
/// fixed-size uniform array, so this is a hard cap).
pub const MAX_LIGHTS_2D: usize = 16;

/// The kind of 2D light source.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub enum Light2dKind {
    /// Emits in all directions from its position.
    #[default]
    Point,
    /// Emits a cone of light along [`Light2d::direction`], fading between the inner
    /// and outer cone angles.
    Spot,
}

/// A dynamic 2D light. Build one with [`Light2d::point`] or [`Light2d::spot`] and add
/// it to the [`Light2dManager`].
#[derive(Copy, Clone, Debug)]
pub struct Light2d {
    /// World-space position in the 2D plane.
    pub position: Vec2,
    /// Height above the plane. Larger values flatten the incidence angle (softer
    /// normal-map shading); 0 puts the light in the plane.
    pub height: f32,
    /// Light color.
    pub color: Color,
    /// Luminous intensity multiplier.
    pub intensity: f32,
    /// Distance beyond which the light contributes nothing.
    pub radius: f32,
    /// Point vs. spot.
    pub kind: Light2dKind,
    /// Spot direction in the plane (normalized internally); ignored for point lights.
    pub direction: Vec2,
    /// Spot inner cone half-angle (radians): full intensity within it.
    pub inner_angle: f32,
    /// Spot outer cone half-angle (radians): intensity reaches zero at it.
    pub outer_angle: f32,
}

impl Default for Light2d {
    fn default() -> Self {
        Light2d {
            position: Vec2::ZERO,
            height: 60.0,
            color: Color::new(1.0, 1.0, 1.0, 1.0),
            intensity: 1.0,
            radius: 300.0,
            kind: Light2dKind::Point,
            direction: Vec2::new(0.0, -1.0),
            inner_angle: 0.3,
            outer_angle: 0.6,
        }
    }
}

impl Light2d {
    /// A point light at `position` with the given `color`, `intensity` and falloff `radius`.
    pub fn point(position: Vec2, color: Color, intensity: f32, radius: f32) -> Self {
        Light2d {
            position,
            color,
            intensity,
            radius,
            kind: Light2dKind::Point,
            ..Default::default()
        }
    }

    /// A spot light at `position` aimed along `direction`, fading between the
    /// `inner` and `outer` cone half-angles (radians).
    pub fn spot(
        position: Vec2,
        direction: Vec2,
        color: Color,
        intensity: f32,
        radius: f32,
        inner: f32,
        outer: f32,
    ) -> Self {
        Light2d {
            position,
            direction,
            color,
            intensity,
            radius,
            kind: Light2dKind::Spot,
            inner_angle: inner,
            outer_angle: outer,
            ..Default::default()
        }
    }

    /// Sets the light's height above the plane.
    pub fn with_height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }
}

/// Thread-local store of the active 2D lights and ambient term, consumed each frame
/// by [`LitMaterial2d`](crate::builtin::LitMaterial2d).
pub struct Light2dManager {
    lights: Vec<Light2d>,
    ambient: Color,
}

thread_local!(static KEY_LIGHT2D_MANAGER: RefCell<Light2dManager> = RefCell::new(Light2dManager::new()));

impl Default for Light2dManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Light2dManager {
    /// Creates an empty manager with a dim default ambient.
    pub fn new() -> Self {
        Light2dManager {
            lights: Vec::new(),
            ambient: Color::new(0.1, 0.1, 0.1, 1.0),
        }
    }

    /// Runs `f` with the global 2D-light manager.
    pub fn get_global_manager<T, F: FnMut(&mut Light2dManager) -> T>(mut f: F) -> T {
        KEY_LIGHT2D_MANAGER.with(|m| f(&mut m.borrow_mut()))
    }

    /// Replaces the active lights (truncated to [`MAX_LIGHTS_2D`]).
    pub fn set_lights(&mut self, lights: &[Light2d]) {
        self.lights.clear();
        self.lights
            .extend(lights.iter().take(MAX_LIGHTS_2D).copied());
    }

    /// Adds one light (ignored once [`MAX_LIGHTS_2D`] is reached).
    pub fn push(&mut self, light: Light2d) {
        if self.lights.len() < MAX_LIGHTS_2D {
            self.lights.push(light);
        }
    }

    /// Removes all lights.
    pub fn clear(&mut self) {
        self.lights.clear();
    }

    /// The active lights.
    pub fn lights(&self) -> &[Light2d] {
        &self.lights
    }

    /// Sets the scene-wide ambient color (applied to every lit object).
    pub fn set_ambient(&mut self, ambient: Color) {
        self.ambient = ambient;
    }

    /// The scene-wide ambient color.
    pub fn ambient(&self) -> Color {
        self.ambient
    }
}
