use crate::camera::Camera3d;
use crate::event::{Action, Key, Modifiers, MouseButton, WindowEvent};
use crate::window::Canvas;
use glamx::{Mat4, Pose3, Vec2, Vec3};
use std::f32;
use crate::camera::first_person3d::CoordSystemRh;

/// Arc-ball camera mode.
///
/// An arc-ball camera is a camera rotating around a fixed point (the focus point) and always
/// looking at it. The following inputs are handled:
///
/// * Left button press + drag - rotates the camera around the focus point
/// * Right button press + drag - translates the focus point on the plane orthogonal to the view
///   direction
/// * Scroll in/out - zoom in/out
/// * Enter key - set the focus point to the origin
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OrbitCamera3d {
    /// The focus point.
    at: Vec3,
    /// Yaw of the camera (rotation along the y axis).
    yaw: f32,
    /// Pitch of the camera (rotation along the x axis).
    pitch: f32,
    /// Distance from the camera to the `at` focus point.
    dist: f32,
    /// Minimum distance from the camera to the `at` focus point.
    min_dist: f32,
    /// Maximum distance from the camera to the `at` focus point.
    max_dist: f32,

    /// Increment of the yaw per unit mouse movement. The default value is 0.005.
    yaw_step: f32,
    /// Increment of the pitch per unit mouse movement. The default value is 0.005.
    pitch_step: f32,
    /// Minimum pitch of the camera.
    min_pitch: f32,
    /// Maximum pitch of the camera.
    max_pitch: f32,
    /// Distance change factor per unit scrolling. The default value is 1.01.
    dist_step: f32,
    rotate_button: Option<MouseButton>,
    rotate_modifiers: Option<Modifiers>,
    drag_button: Option<MouseButton>,
    drag_modifiers: Option<Modifiers>,
    reset_key: Option<Key>,

    fov: f32,
    znear: f32,
    zfar: f32,
    view: Mat4,
    proj: Mat4,
    proj_view: Mat4,
    inverse_proj_view: Mat4,
    last_cursor_pos: Vec2,
    last_framebuffer_size: Vec2,
    coord_system: CoordSystemRh,
}

impl Default for OrbitCamera3d {
    fn default() -> Self {
        Self::new(Vec3::new(0.0f32, 0.0, -2.0), Vec3::ZERO)
    }
}

impl OrbitCamera3d {
    /// Creates a new arc-ball camera with default settings.
    ///
    /// The camera will orbit around the `at` point, starting at the `eye` position.
    /// Default frustum parameters are used: 45° field of view, near plane at 0.1, far plane at 1000.
    ///
    /// # Arguments
    /// * `eye` - Initial camera position
    /// * `at` - The focus point the camera will look at and orbit around
    ///
    /// # Returns
    /// A new `OrbitCamera3d` camera instance
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::prelude::*;
    /// // Camera looking at origin from 5 units away on the Z axis
    /// let camera = OrbitCamera3d::new(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO);
    /// ```
    pub fn new(eye: Vec3, at: Vec3) -> OrbitCamera3d {
        OrbitCamera3d::new_with_frustum(f32::consts::PI / 4.0, 0.1, 1000.0, eye, at)
    }

    /// Creates a new arc-ball camera with custom frustum parameters.
    ///
    /// # Arguments
    /// * `fov` - Field of view in radians (typical value: π/4 for 45°)
    /// * `znear` - Near clipping plane distance
    /// * `zfar` - Far clipping plane distance
    /// * `eye` - Initial camera position
    /// * `at` - The focus point the camera will look at and orbit around
    ///
    /// # Returns
    /// A new `OrbitCamera3d` camera instance with custom frustum
    pub fn new_with_frustum(fov: f32, znear: f32, zfar: f32, eye: Vec3, at: Vec3) -> OrbitCamera3d {
        let mut res = OrbitCamera3d {
            at: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            dist: 0.0,
            min_dist: 0.00001,
            max_dist: 1.0e4,
            yaw_step: 0.005,
            pitch_step: 0.005,
            min_pitch: 0.01,
            max_pitch: std::f32::consts::PI - 0.01,
            #[cfg(target_os = "macos")]
            dist_step: 1.0001,
            #[cfg(not(target_os = "macos"))]
            dist_step: 1.01,
            rotate_button: Some(MouseButton::Button1),
            rotate_modifiers: None,
            drag_button: Some(MouseButton::Button2),
            drag_modifiers: None,
            reset_key: Some(Key::Return),
            fov,
            znear,
            zfar,
            view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
            proj_view: Mat4::IDENTITY,
            inverse_proj_view: Mat4::IDENTITY,
            last_framebuffer_size: Vec2::new(800.0, 600.0),
            last_cursor_pos: Vec2::ZERO,
            coord_system: CoordSystemRh::from_up_axis(Vec3::Y),
        };

        res.look_at(eye, at);

        res
    }

    /// Returns the focus point the camera is looking at.
    ///
    /// # Returns
    /// The 3D point in world space that the camera orbits around
    pub fn at(&self) -> Vec3 {
        self.at
    }

    /// Sets the focus point the camera should look at.
    ///
    /// The camera will orbit around this new point.
    ///
    /// # Arguments
    /// * `at` - The new focus point in world space
    pub fn set_at(&mut self, at: Vec3) {
        self.at = at;
        self.update_projviews();
    }

    /// Returns the camera's yaw angle in radians.
    ///
    /// Yaw is the horizontal rotation around the up axis.
    ///
    /// # Returns
    /// The yaw angle in radians
    pub fn yaw(&self) -> f32 {
        self.yaw
    }

    /// Sets the camera's yaw angle.
    ///
    /// Yaw controls horizontal rotation around the up axis (typically Y).
    ///
    /// # Arguments
    /// * `yaw` - The new yaw angle in radians
    pub fn set_yaw(&mut self, yaw: f32) {
        self.yaw = yaw;

        self.update_restrictions();
        self.update_projviews();
    }

    /// Returns the camera's pitch angle in radians.
    ///
    /// Pitch is the vertical rotation, controlling how high or low the camera looks.
    ///
    /// # Returns
    /// The pitch angle in radians
    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    /// Sets the camera's pitch angle.
    ///
    /// Pitch controls vertical rotation (looking up/down).
    ///
    /// # Arguments
    /// * `pitch` - The new pitch angle in radians
    pub fn set_pitch(&mut self, pitch: f32) {
        self.pitch = pitch;

        self.update_restrictions();
        self.update_projviews();
    }

    /// Returns the minimum allowed pitch angle.
    ///
    /// # Returns
    /// The minimum pitch in radians (default: 0.01)
    pub fn min_pitch(&self) -> f32 {
        self.min_pitch
    }

    /// Sets the minimum allowed pitch angle.
    ///
    /// This prevents the camera from rotating too far downward.
    ///
    /// # Arguments
    /// * `min_pitch` - The minimum pitch in radians
    pub fn set_min_pitch(&mut self, min_pitch: f32) {
        self.min_pitch = min_pitch;
    }

    /// Returns the maximum allowed pitch angle.
    ///
    /// # Returns
    /// The maximum pitch in radians (default: π - 0.01)
    pub fn max_pitch(&self) -> f32 {
        self.max_pitch
    }

    /// Sets the maximum allowed pitch angle.
    ///
    /// This prevents the camera from rotating too far upward.
    ///
    /// # Arguments
    /// * `max_pitch` - The maximum pitch in radians
    pub fn set_max_pitch(&mut self, max_pitch: f32) {
        self.max_pitch = max_pitch;
    }

    /// Returns the current distance from the camera to the focus point.
    ///
    /// # Returns
    /// The distance in world units
    pub fn dist(&self) -> f32 {
        self.dist
    }

    /// Sets the distance from the camera to the focus point.
    ///
    /// The distance is automatically clamped between `min_dist` and `max_dist`.
    ///
    /// # Arguments
    /// * `dist` - The new distance in world units
    pub fn set_dist(&mut self, dist: f32) {
        self.dist = dist;

        self.update_restrictions();
        self.update_projviews();
    }

    /// Returns the minimum allowed distance from the camera to the focus point.
    ///
    /// # Returns
    /// The minimum distance in world units (default: 0.00001)
    pub fn min_dist(&self) -> f32 {
        self.min_dist
    }

    /// Sets the minimum allowed distance from the camera to the focus point.
    ///
    /// This prevents the camera from getting too close.
    ///
    /// # Arguments
    /// * `min_dist` - The minimum distance in world units
    pub fn set_min_dist(&mut self, min_dist: f32) {
        self.min_dist = min_dist;
    }

    /// Returns the maximum allowed distance from the camera to the focus point.
    ///
    /// # Returns
    /// The maximum distance in world units (default: 10000)
    pub fn max_dist(&self) -> f32 {
        self.max_dist
    }

    /// Sets the maximum allowed distance from the camera to the focus point.
    ///
    /// This prevents the camera from zooming out too far.
    ///
    /// # Arguments
    /// * `max_dist` - The maximum distance in world units
    pub fn set_max_dist(&mut self, max_dist: f32) {
        self.max_dist = max_dist;
    }

    /// Sets the zoom speed factor.
    ///
    /// Each mouse wheel scroll multiplies the distance by this factor.
    /// Values > 1.0 zoom out, values < 1.0 zoom in.
    ///
    /// # Arguments
    /// * `dist_step` - The multiplier per scroll unit (default: 1.01 on most platforms, 1.0001 on macOS)
    pub fn set_dist_step(&mut self, dist_step: f32) {
        self.dist_step = dist_step;
    }

    /// Positions and orients the camera to look at a specific point from a specific position.
    ///
    /// This is similar to gluLookAt. The camera will be positioned at `eye`,
    /// looking at `at`, and will orbit around `at` when the user interacts with it.
    ///
    /// # Arguments
    /// * `eye` - The position to place the camera
    /// * `at` - The point to look at (becomes the new focus point)
    pub fn look_at(&mut self, eye: Vec3, at: Vec3) {
        let dist = (eye - at).length();

        let view_eye = self.coord_system.rotation_to_y_up * eye;
        let view_at = self.coord_system.rotation_to_y_up * at;
        let pitch = ((view_eye.y - view_at.y) / dist).acos();
        let yaw = (view_eye.z - view_at.z).atan2(view_eye.x - view_at.x);

        self.at = at;
        self.dist = dist;
        self.yaw = yaw;
        self.pitch = pitch;

        self.update_restrictions();
        self.update_projviews();
    }

    /// Transformation applied by the camera without perspective.
    fn update_restrictions(&mut self) {
        if self.dist < self.min_dist {
            self.dist = self.min_dist
        }

        if self.dist > self.max_dist {
            self.dist = self.max_dist
        }

        if self.pitch <= self.min_pitch {
            self.pitch = self.min_pitch
        }

        if self.pitch > self.max_pitch {
            self.pitch = self.max_pitch
        }
    }

    /// The button used to rotate the OrbitCamera3d camera.
    pub fn rotate_button(&self) -> Option<MouseButton> {
        self.rotate_button
    }

    /// Set the button used to rotate the OrbitCamera3d camera.
    /// Use None to disable rotation.
    pub fn rebind_rotate_button(&mut self, new_button: Option<MouseButton>) {
        self.rotate_button = new_button;
    }

    /// Modifiers that must be pressed for the camera rotation to occur.
    pub fn rotate_modifiers(&self) -> Option<Modifiers> {
        self.rotate_modifiers
    }

    /// Sets the modifiers that must be pressed for the camera rotation to occur.
    ///
    /// If this is set to `None`, then pressing any modifier will not prevent rotation from occurring.
    /// If this is different from `None` then rotation will occur only if the exact specified set of modifiers is pressed.
    /// In particular, if this is set to `Some(Modifiers::empty())` then, rotation will occur only of no modifier is pressed.
    pub fn set_rotate_modifiers(&mut self, modifiers: Option<Modifiers>) {
        self.rotate_modifiers = modifiers
    }

    /// Modifiers that must be pressed for the camera drag to occur.
    pub fn drag_modifiers(&self) -> Option<Modifiers> {
        self.drag_modifiers
    }

    /// Sets the modifiers that must be pressed for the camera drag to occur.
    ///
    /// If this is set to `None`, then pressing any modifier will not prevent dragging from occurring.
    /// If this is different from `None` then drag will occur only if the exact specified set of modifiers is pressed.
    /// In particular, if this is set to `Some(Modifiers::empty())` then, drag will occur only of no modifier is pressed.
    pub fn set_drag_modifiers(&mut self, modifiers: Option<Modifiers>) {
        self.drag_modifiers = modifiers
    }

    /// The button used to drag the OrbitCamera3d camera.
    pub fn drag_button(&self) -> Option<MouseButton> {
        self.drag_button
    }

    /// Set the button used to drag the OrbitCamera3d camera.
    /// Use None to disable dragging.
    pub fn rebind_drag_button(&mut self, new_button: Option<MouseButton>) {
        self.drag_button = new_button;
    }

    /// The key used to reset the OrbitCamera3d camera.
    pub fn reset_key(&self) -> Option<Key> {
        self.reset_key
    }

    /// Set the key used to reset the OrbitCamera3d camera.
    /// Use None to disable reset.
    pub fn rebind_reset_key(&mut self, new_key: Option<Key>) {
        self.reset_key = new_key;
    }

    fn handle_left_button_displacement(&mut self, dpos: Vec2) {
        self.yaw += dpos.x * self.yaw_step;
        self.pitch -= dpos.y * self.pitch_step;

        self.update_restrictions();
        self.update_projviews();
    }

    /// Performs a translation of the camera eye and focus.
    /// The delta coordinates are expected to be normalized to the [-1, 1] range.
    fn handle_right_button_displacement(&mut self, dpos_norm: Vec2) {
        let eye = self.eye();
        let dir = (self.at - eye).normalize();
        let tangent = self.coord_system.up_axis.cross(dir).normalize();
        let bitangent = dir.cross(tangent);
        self.at =
            self.at + tangent * (dpos_norm.x * self.dist) + bitangent * (dpos_norm.y * self.dist);
        self.update_projviews();
    }

    fn handle_scroll(&mut self, off: f32) {
        // To "focus" the zoom towards the point under the cursor, first we
        // translate the camera to bring that point in the center of the view
        // and then undo the translation.
        let mut dpos = Vec2::new(
            0.5 - self.last_cursor_pos.x / self.last_framebuffer_size.x,
            0.5 - self.last_cursor_pos.y / self.last_framebuffer_size.y,
        );
        self.handle_right_button_displacement(dpos);

        self.dist *= self.dist_step.powf(off);
        self.update_restrictions();
        self.update_projviews();

        dpos = -dpos;
        self.handle_right_button_displacement(dpos);
    }

    fn update_projviews(&mut self) {
        let aspect = self.last_framebuffer_size.x / self.last_framebuffer_size.y;
        self.proj = Mat4::perspective_rh_gl(self.fov, aspect, self.znear, self.zfar);
        self.view = self.view_transform().to_mat4();
        self.proj_view = self.proj * self.view;
        self.inverse_proj_view = self.proj_view.inverse();
    }

    /// Sets the up vector of this camera. Prefer using [`set_up_axis_dir`](#method.set_up_axis_dir)
    /// if your up vector is already normalized.
    #[inline]
    pub fn set_up_axis(&mut self, up_axis: Vec3) {
        self.set_up_axis_dir(up_axis.normalize());
    }

    /// Sets the up-axis direction of this camera.
    #[inline]
    pub fn set_up_axis_dir(&mut self, up_axis: Vec3) {
        if self.coord_system.up_axis != up_axis {
            let new_coord_system = CoordSystemRh::from_up_axis(up_axis);
            // Since setting the up axis changes the meaning of pitch and yaw
            // angles, we need to recalculate them in order to preserve the eye
            // position.
            let old_eye = self.eye();
            self.coord_system = new_coord_system;
            self.look_at(old_eye, self.at);
        }
    }
}

impl Camera3d for OrbitCamera3d {
    fn clip_planes(&self) -> (f32, f32) {
        (self.znear, self.zfar)
    }

    fn view_transform(&self) -> Pose3 {
        Pose3::look_at_rh(self.eye(), self.at, self.coord_system.up_axis)
    }

    fn eye(&self) -> Vec3 {
        let view_at = self.coord_system.rotation_to_y_up * self.at;
        let px = view_at.x + self.dist * self.yaw.cos() * self.pitch.sin();
        let py = view_at.y + self.dist * self.pitch.cos();
        let pz = view_at.z + self.dist * self.yaw.sin() * self.pitch.sin();
        self.coord_system.rotation_to_y_up.conjugate() * Vec3::new(px, py, pz)
    }

    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent) {
        match *event {
            WindowEvent::CursorPos(x, y, modifiers) => {
                let curr_pos = Vec2::new(x as f32, y as f32);

                if let Some(rotate_button) = self.rotate_button {
                    if canvas.get_mouse_button(rotate_button) == Action::Press
                        && self
                            .rotate_modifiers
                            .map(|m| m == modifiers)
                            .unwrap_or(true)
                    {
                        let dpos = curr_pos - self.last_cursor_pos;
                        self.handle_left_button_displacement(dpos)
                    }
                }

                if let Some(drag_button) = self.drag_button {
                    if canvas.get_mouse_button(drag_button) == Action::Press
                        && self.drag_modifiers.map(|m| m == modifiers).unwrap_or(true)
                    {
                        let dpos = curr_pos - self.last_cursor_pos;
                        let dpos_norm = dpos / self.last_framebuffer_size;
                        self.handle_right_button_displacement(dpos_norm)
                    }
                }

                self.last_cursor_pos = curr_pos;
            }
            WindowEvent::Key(key, Action::Press, _) if Some(key) == self.reset_key => {
                self.at = Vec3::ZERO;
                self.update_projviews();
            }
            WindowEvent::Scroll(_, off, _) => self.handle_scroll(off as f32),
            WindowEvent::FramebufferSize(w, h) => {
                self.last_framebuffer_size = Vec2::new(w as f32, h as f32);
                self.update_projviews();
            }
            _ => {}
        }
    }

    #[inline]
    fn view_transform_pair(&self, _pass: usize) -> (Pose3, Mat4) {
        (self.view_transform(), self.proj)
    }

    fn transformation(&self) -> Mat4 {
        self.proj_view
    }

    fn inverse_transformation(&self) -> Mat4 {
        self.inverse_proj_view
    }

    fn update(&mut self, _: &Canvas) {}
}

