//! Skeletal / node animation sampled into scene-node transforms.
//!
//! A glTF animation is a set of [`AnimationChannel`]s, each driving one
//! transform component (translation, rotation, or scale) of one target
//! [`SceneNode3d`] over time from a list of keyframes. An [`AnimationClip`]
//! bundles the channels of a single named animation, and an [`AnimationPlayer`]
//! owns every clip of a loaded model plus the playback cursor.
//!
//! Playback is **user-driven**: call [`AnimationPlayer::update`] each frame with
//! the elapsed time *before* rendering. `update` samples the active clip and
//! writes the result into the target nodes' local transforms, which invalidates
//! them so the next `prepare()` re-propagates world matrices. This animates rigid
//! hierarchies directly; for skinned meshes the same node transforms also drive
//! the skinning palette (see [`crate::scene::Skin3d`]).

use crate::scene::SceneNode3d;
use glamx::{Quat, Vec3};

/// How keyframe values are interpolated between two times (mirrors glTF).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Interpolation {
    /// Linear interpolation (`lerp` for vectors, `slerp` for rotations).
    Linear,
    /// Stepwise: hold the previous keyframe's value until the next time.
    Step,
    /// Cubic Hermite spline; each keyframe stores `(in_tangent, value, out_tangent)`.
    CubicSpline,
}

/// The animated transform component and its keyframe outputs.
///
/// For [`Interpolation::CubicSpline`] each keyframe contributes three consecutive
/// entries `(in_tangent, value, out_tangent)`, so the output vectors hold
/// `3 * times.len()` elements; otherwise they hold one element per time.
enum ChannelOutput {
    /// Local translation keyframes.
    Translation(Vec<Vec3>),
    /// Local rotation keyframes (unit quaternions).
    Rotation(Vec<Quat>),
    /// Local scale keyframes.
    Scale(Vec<Vec3>),
    /// Morph-target weight keyframes: `num_targets` weights per keyframe, stored
    /// flat as `[key * num_targets + target]` (×3 per key for `CubicSpline`).
    MorphWeights {
        values: Vec<f32>,
        num_targets: usize,
    },
}

/// One animation channel: a keyframe track bound to a single node's single
/// transform component.
pub struct AnimationChannel {
    target: SceneNode3d,
    times: Vec<f32>,
    interpolation: Interpolation,
    output: ChannelOutput,
}

impl AnimationChannel {
    /// Builds a translation channel.
    pub fn translation(
        target: SceneNode3d,
        times: Vec<f32>,
        values: Vec<Vec3>,
        interpolation: Interpolation,
    ) -> Self {
        Self {
            target,
            times,
            interpolation,
            output: ChannelOutput::Translation(values),
        }
    }

    /// Builds a rotation channel.
    pub fn rotation(
        target: SceneNode3d,
        times: Vec<f32>,
        values: Vec<Quat>,
        interpolation: Interpolation,
    ) -> Self {
        Self {
            target,
            times,
            interpolation,
            output: ChannelOutput::Rotation(values),
        }
    }

    /// Builds a scale channel.
    pub fn scale(
        target: SceneNode3d,
        times: Vec<f32>,
        values: Vec<Vec3>,
        interpolation: Interpolation,
    ) -> Self {
        Self {
            target,
            times,
            interpolation,
            output: ChannelOutput::Scale(values),
        }
    }

    /// Builds a morph-target-weight channel. `values` holds `num_targets` weights per
    /// keyframe (×3 per key for [`Interpolation::CubicSpline`]); on each update the
    /// sampled weights are fanned out to the target node's morphable objects.
    pub fn morph_weights(
        target: SceneNode3d,
        times: Vec<f32>,
        values: Vec<f32>,
        num_targets: usize,
        interpolation: Interpolation,
    ) -> Self {
        Self {
            target,
            times,
            interpolation,
            output: ChannelOutput::MorphWeights {
                values,
                num_targets,
            },
        }
    }

    /// The last keyframe time of this channel (0 when empty).
    fn end_time(&self) -> f32 {
        self.times.last().copied().unwrap_or(0.0)
    }

    /// Locates the keyframe segment for `t`, returning `(i0, i1, u)` where the
    /// sampled time lies between keyframes `i0` and `i1` at fraction `u ∈ [0, 1]`.
    /// Times are assumed sorted ascending (glTF requirement).
    fn segment(&self, t: f32) -> Option<(usize, usize, f32)> {
        let n = self.times.len();
        if n == 0 {
            return None;
        }
        if t <= self.times[0] || n == 1 {
            return Some((0, 0, 0.0));
        }
        if t >= self.times[n - 1] {
            return Some((n - 1, n - 1, 0.0));
        }
        // Binary search for the first time strictly greater than `t`.
        let i1 = self.times.partition_point(|&k| k <= t);
        let i0 = i1 - 1;
        let dt = self.times[i1] - self.times[i0];
        let u = if dt > 0.0 {
            (t - self.times[i0]) / dt
        } else {
            0.0
        };
        Some((i0, i1, u))
    }

    /// Samples this channel at time `t` and writes the result into the target
    /// node's matching local transform component.
    fn apply(&mut self, t: f32) {
        let Some((i0, i1, u)) = self.segment(t) else {
            return;
        };
        let cubic = self.interpolation == Interpolation::CubicSpline;
        let dt = if i0 != i1 {
            self.times[i1] - self.times[i0]
        } else {
            0.0
        };

        match &self.output {
            ChannelOutput::Translation(v) => {
                let value = sample_vec3(v, self.interpolation, i0, i1, u, dt);
                self.target.set_position(value);
            }
            ChannelOutput::Scale(v) => {
                let value = sample_vec3(v, self.interpolation, i0, i1, u, dt);
                self.target.set_local_scale(value.x, value.y, value.z);
            }
            ChannelOutput::Rotation(v) => {
                let value = if cubic {
                    sample_quat_cubic(v, i0, i1, u, dt)
                } else if self.interpolation == Interpolation::Step || i0 == i1 {
                    v[i0]
                } else {
                    v[i0].slerp(v[i1], u)
                };
                self.target.set_rotation(value.normalize());
            }
            ChannelOutput::MorphWeights {
                values,
                num_targets,
            } => {
                let weights =
                    sample_weights(values, *num_targets, self.interpolation, i0, i1, u, dt);
                self.target.set_morph_weights(&weights);
            }
        }
    }
}

/// Samples a morph-target-weight track at the located segment, returning the
/// `num_targets` interpolated weights.
fn sample_weights(
    v: &[f32],
    num_targets: usize,
    interp: Interpolation,
    i0: usize,
    i1: usize,
    u: f32,
    dt: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; num_targets];
    match interp {
        Interpolation::Step => {
            let base = i0 * num_targets;
            out.copy_from_slice(&v[base..base + num_targets]);
        }
        Interpolation::Linear => {
            let (a, b) = (i0 * num_targets, i1 * num_targets);
            for k in 0..num_targets {
                out[k] = if i0 == i1 {
                    v[a + k]
                } else {
                    v[a + k] + (v[b + k] - v[a + k]) * u
                };
            }
        }
        Interpolation::CubicSpline => {
            // Each key stores three `num_targets`-sized blocks: (in, value, out).
            let (h00, h10, h01, h11) = hermite_basis(u);
            for k in 0..num_targets {
                let p0 = v[(3 * i0 + 1) * num_targets + k];
                let m0 = v[(3 * i0 + 2) * num_targets + k] * dt;
                let p1 = v[(3 * i1 + 1) * num_targets + k];
                let m1 = v[(3 * i1) * num_targets + k] * dt;
                out[k] = p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11;
            }
        }
    }
    out
}

/// Samples a `Vec3` keyframe track (translation or scale).
fn sample_vec3(v: &[Vec3], interp: Interpolation, i0: usize, i1: usize, u: f32, dt: f32) -> Vec3 {
    match interp {
        Interpolation::CubicSpline => {
            // Each key stores (in, value, out); the value is the middle entry.
            let p0 = v[3 * i0 + 1];
            let m0 = v[3 * i0 + 2] * dt; // out-tangent of key i0
            let p1 = v[3 * i1 + 1];
            let m1 = v[3 * i1] * dt; // in-tangent of key i1
            hermite_vec3(p0, m0, p1, m1, u)
        }
        Interpolation::Step => v[i0],
        Interpolation::Linear => {
            if i0 == i1 {
                v[i0]
            } else {
                v[i0].lerp(v[i1], u)
            }
        }
    }
}

/// Cubic-Hermite-samples a quaternion track and renormalizes.
fn sample_quat_cubic(v: &[Quat], i0: usize, i1: usize, u: f32, dt: f32) -> Quat {
    let p0 = v[3 * i0 + 1];
    let m0 = quat_scale(v[3 * i0 + 2], dt);
    let p1 = v[3 * i1 + 1];
    let m1 = quat_scale(v[3 * i1], dt);
    let (h00, h10, h01, h11) = hermite_basis(u);
    let q = quat_add(
        quat_add(quat_scale(p0, h00), quat_scale(m0, h10)),
        quat_add(quat_scale(p1, h01), quat_scale(m1, h11)),
    );
    q.normalize()
}

/// Cubic Hermite blend of two endpoints and tangents.
fn hermite_vec3(p0: Vec3, m0: Vec3, p1: Vec3, m1: Vec3, u: f32) -> Vec3 {
    let (h00, h10, h01, h11) = hermite_basis(u);
    p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11
}

/// The four cubic Hermite basis weights at `u`.
fn hermite_basis(u: f32) -> (f32, f32, f32, f32) {
    let u2 = u * u;
    let u3 = u2 * u;
    (
        2.0 * u3 - 3.0 * u2 + 1.0, // p0
        u3 - 2.0 * u2 + u,         // m0
        -2.0 * u3 + 3.0 * u2,      // p1
        u3 - u2,                   // m1
    )
}

// Component-wise quaternion helpers used only by the cubic-spline path, where the
// keyframes are treated as 4-vectors before renormalization (per the glTF spec).
fn quat_scale(q: Quat, s: f32) -> Quat {
    Quat::from_xyzw(q.x * s, q.y * s, q.z * s, q.w * s)
}
fn quat_add(a: Quat, b: Quat) -> Quat {
    Quat::from_xyzw(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w)
}

/// A single named animation: all the channels that play together.
pub struct AnimationClip {
    /// The animation's name (from glTF; may be empty).
    pub name: String,
    channels: Vec<AnimationChannel>,
    duration: f32,
}

impl AnimationClip {
    /// Builds a clip from its channels, computing the duration as the latest
    /// keyframe time across all channels.
    pub fn new(name: String, channels: Vec<AnimationChannel>) -> Self {
        let duration = channels
            .iter()
            .map(|c| c.end_time())
            .fold(0.0_f32, f32::max);
        Self {
            name,
            channels,
            duration,
        }
    }

    /// The clip length in seconds.
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// Samples every channel at `t` and writes the target nodes' transforms.
    fn apply(&mut self, t: f32) {
        for ch in &mut self.channels {
            ch.apply(t);
        }
    }
}

/// Owns all animation clips of a loaded model and the playback cursor.
///
/// Returned inside [`crate::scene::GltfModel`]. The caller drives it each frame:
///
/// ```no_run
/// # use kiss3d::scene::{SceneNode3d, GltfModel};
/// # use std::path::Path;
/// # let mut scene = SceneNode3d::empty();
/// let mut model = scene.add_gltf(Path::new("character.glb"), glamx::Vec3::ONE);
/// model.player.play("Walk");
/// model.player.set_looping(true);
/// // each frame, with `dt` seconds elapsed:
/// // model.player.update(dt);
/// ```
pub struct AnimationPlayer {
    clips: Vec<AnimationClip>,
    current: Option<usize>,
    time: f32,
    looping: bool,
    speed: f32,
    playing: bool,
}

impl AnimationPlayer {
    /// Creates a player owning `clips`, initially stopped.
    pub fn new(clips: Vec<AnimationClip>) -> Self {
        Self {
            clips,
            current: None,
            time: 0.0,
            looping: true,
            speed: 1.0,
            playing: false,
        }
    }

    /// The number of clips owned by this player.
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// The names of all clips, in index order.
    pub fn clip_names(&self) -> impl Iterator<Item = &str> {
        self.clips.iter().map(|c| c.name.as_str())
    }

    /// Starts (or restarts) the clip with the given name. Returns `false` if no
    /// clip matches; current playback is then left unchanged.
    pub fn play(&mut self, name: &str) -> bool {
        match self.clips.iter().position(|c| c.name == name) {
            Some(i) => {
                self.play_index(i);
                true
            }
            None => false,
        }
    }

    /// Starts (or restarts) the clip at `index`. Out-of-range indices are ignored.
    pub fn play_index(&mut self, index: usize) {
        if index < self.clips.len() {
            self.current = Some(index);
            self.time = 0.0;
            self.playing = true;
        }
    }

    /// Stops playback (the pose freezes at the last sampled frame).
    pub fn stop(&mut self) {
        self.playing = false;
    }

    /// Whether a clip is currently advancing.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Sets whether the active clip loops (default `true`).
    pub fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
    }

    /// Sets the playback speed multiplier (default `1.0`; negative plays backward).
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    /// The current playback time in seconds within the active clip.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Seeks the active clip to `time` seconds and re-applies the pose.
    pub fn seek(&mut self, time: f32) {
        self.time = time;
        self.apply_current();
    }

    /// Advances the active clip by `dt` seconds and writes the sampled pose into
    /// the target nodes. Call once per frame before rendering. No-op when stopped
    /// or when no clip is active.
    pub fn update(&mut self, dt: f32) {
        if !self.playing {
            return;
        }
        let Some(i) = self.current else {
            return;
        };
        let duration = self.clips[i].duration;
        self.time += dt * self.speed;

        if duration > 0.0 {
            if self.looping {
                // rem_euclid keeps the cursor in `[0, duration)` for both
                // forward and backward playback.
                self.time = self.time.rem_euclid(duration);
            } else if self.time >= duration {
                self.time = duration;
                self.playing = false;
            } else if self.time < 0.0 {
                self.time = 0.0;
                self.playing = false;
            }
        } else {
            self.time = 0.0;
        }

        self.apply_current();
    }

    fn apply_current(&mut self) {
        if let Some(i) = self.current {
            let t = self.time;
            self.clips[i].apply(t);
        }
    }
}
