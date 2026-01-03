use crate::camera::Camera3d;
use crate::event::{Action, Key, MouseButton, WindowEvent};
use crate::window::Canvas;
use glamx::{Mat4, Pose3, Rot3, Vec2, Vec3};
use std::f32;

/// First-person (FPS-style) camera.
///
/// A camera that moves through the scene from a first-person perspective,
/// similar to controls in first-person shooter games.
///
/// # Default Controls
/// - **Left mouse + drag**: Look around (rotate view)
/// - **Right mouse + drag**: Strafe (move on plane perpendicular to view)
/// - **Arrow keys**: Move forward/backward/left/right
/// - **Mouse wheel**: Move forward/backward
///
/// All controls can be customized using the rebind methods.
///
/// # Example
/// ```no_run
/// # use kiss3d::prelude::*;
/// # #[kiss3d::main]
/// # async fn main() {
/// # let mut window = Window::new("Example");
/// let mut camera = FirstPersonCamera3d::new(
///     Vec3::new(0.0, 1.0, 5.0),  // Eye position
///     Vec3::ZERO                 // Looking at origin
/// );
/// // Use with window.render_with_camera(&mut camera).await
/// # }
/// ```
#[derive(Copy, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FirstPersonCamera3d {
    eye: Vec3,
    yaw: f32,
    pitch: f32,

    yaw_step: f32,
    pitch_step: f32,
    move_step: f32,
    rotate_button: Option<MouseButton>,
    drag_button: Option<MouseButton>,
    up_key: Option<Key>,
    down_key: Option<Key>,
    left_key: Option<Key>,
    right_key: Option<Key>,

    fov: f32,
    znear: f32,
    zfar: f32,
    proj: Mat4,
    view: Mat4,
    proj_view: Mat4,
    inverse_proj_view: Mat4,
    last_cursor_pos: Vec2,
    last_framebuffer_size: Vec2,
    coord_system: CoordSystemRh,
}

impl FirstPersonCamera3d {
    /// Creates a new first-person camera with default settings.
    ///
    /// Default frustum: 45° field of view, near plane at 0.1, far plane at 1024.
    ///
    /// # Arguments
    /// * `eye` - Initial camera position
    /// * `at` - Initial point to look at
    ///
    /// # Returns
    /// A new `FirstPersonCamera3d` camera instance
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::prelude::*;
    /// let camera = FirstPersonCamera3d::new(
    ///     Vec3::new(0.0, 5.0, 10.0),
    ///     Vec3::ZERO
    /// );
    /// ```
    pub fn new(eye: Vec3, at: Vec3) -> FirstPersonCamera3d {
        FirstPersonCamera3d::new_with_frustum(f32::consts::PI / 4.0, 0.1, 1024.0, eye, at)
    }

    /// Creates a new first-person camera with custom frustum parameters.
    ///
    /// # Arguments
    /// * `fov` - Field of view in radians
    /// * `znear` - Near clipping plane distance
    /// * `zfar` - Far clipping plane distance
    /// * `eye` - Initial camera position
    /// * `at` - Initial point to look at
    ///
    /// # Returns
    /// A new `FirstPersonCamera3d` camera instance
    pub fn new_with_frustum(
        fov: f32,
        znear: f32,
        zfar: f32,
        eye: Vec3,
        at: Vec3,
    ) -> FirstPersonCamera3d {
        let mut res = FirstPersonCamera3d {
            eye: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            yaw_step: 0.005,
            pitch_step: 0.005,
            move_step: 0.5,
            rotate_button: Some(MouseButton::Button1),
            drag_button: Some(MouseButton::Button2),
            up_key: Some(Key::Up),
            down_key: Some(Key::Down),
            left_key: Some(Key::Left),
            right_key: Some(Key::Right),
            fov,
            znear,
            zfar,
            proj: Mat4::IDENTITY,
            view: Mat4::IDENTITY,
            proj_view: Mat4::IDENTITY,
            inverse_proj_view: Mat4::IDENTITY,
            last_cursor_pos: Vec2::ZERO,
            last_framebuffer_size: Vec2::new(800.0, 600.0),
            coord_system: CoordSystemRh::from_up_axis(Vec3::Y),
        };

        res.look_at(eye, at);

        res
    }

    /// Sets the translational increment per arrow press.
    ///
    /// The default value is 0.5.
    #[inline]
    pub fn set_move_step(&mut self, step: f32) {
        self.move_step = step;
    }

    /// Sets the pitch increment per mouse movement.
    ///
    /// The default value is 0.005.
    #[inline]
    pub fn set_pitch_step(&mut self, step: f32) {
        self.pitch_step = step;
    }

    /// Sets the yaw increment per mouse movement.
    ///
    /// The default value is 0.005.
    #[inline]
    pub fn set_yaw_step(&mut self, step: f32) {
        self.yaw_step = step;
    }

    /// Gets the translational increment per arrow press.
    #[inline]
    pub fn move_step(&self) -> f32 {
        self.move_step
    }

    /// Gets the pitch increment per mouse movement.
    #[inline]
    pub fn pitch_step(&self) -> f32 {
        self.pitch_step
    }

    /// Gets the yaw  increment per mouse movement.
    #[inline]
    pub fn yaw_step(&self) -> f32 {
        self.yaw_step
    }

    /// Changes the orientation and position of the camera to look at the specified point.
    pub fn look_at(&mut self, eye: Vec3, at: Vec3) {
        let dist = (eye - at).length();

        let view_eye = self.coord_system.rotation_to_y_up * eye;
        let view_at = self.coord_system.rotation_to_y_up * at;
        let pitch = ((view_at.y - view_eye.y) / dist).acos();
        let yaw = (view_at.z - view_eye.z).atan2(view_at.x - view_eye.x);

        self.eye = eye;
        self.yaw = yaw;
        self.pitch = pitch;
        self.update_projviews();
    }

    /// The point the camera is looking at.
    pub fn at(&self) -> Vec3 {
        let view_eye = self.coord_system.rotation_to_y_up * self.eye;
        let ax = view_eye.x + self.yaw.cos() * self.pitch.sin();
        let ay = view_eye.y + self.pitch.cos();
        let az = view_eye.z + self.yaw.sin() * self.pitch.sin();
        self.coord_system.rotation_to_y_up.conjugate() * Vec3::new(ax, ay, az)
    }

    fn update_restrictions(&mut self) {
        if self.pitch <= 0.01 {
            self.pitch = 0.01
        }

        let _pi: f32 = f32::consts::PI;
        if self.pitch > _pi - 0.01 {
            self.pitch = _pi - 0.01
        }
    }

    /// The button used to rotate the FirstPersonCamera3d camera.
    pub fn rotate_button(&self) -> Option<MouseButton> {
        self.rotate_button
    }

    /// Set the button used to rotate the FirstPersonCamera3d camera.
    /// Use None to disable rotation.
    pub fn rebind_rotate_button(&mut self, new_button: Option<MouseButton>) {
        self.rotate_button = new_button;
    }

    /// The button used to drag the FirstPersonCamera3d camera.
    pub fn drag_button(&self) -> Option<MouseButton> {
        self.drag_button
    }

    /// Set the button used to drag the FirstPersonCamera3d camera.
    /// Use None to disable dragging.
    pub fn rebind_drag_button(&mut self, new_button: Option<MouseButton>) {
        self.drag_button = new_button;
    }

    /// The movement button for up.
    pub fn up_key(&self) -> Option<Key> {
        self.up_key
    }

    /// The movement button for down.
    pub fn down_key(&self) -> Option<Key> {
        self.down_key
    }

    /// The movement button for left.
    pub fn left_key(&self) -> Option<Key> {
        self.left_key
    }

    /// The movement button for right.
    pub fn right_key(&self) -> Option<Key> {
        self.right_key
    }

    /// Set the movement button for up.
    /// Use None to disable movement in this direction.
    pub fn rebind_up_key(&mut self, new_key: Option<Key>) {
        self.up_key = new_key;
    }

    /// Set the movement button for down.
    /// Use None to disable movement in this direction.
    pub fn rebind_down_key(&mut self, new_key: Option<Key>) {
        self.down_key = new_key;
    }

    /// Set the movement button for left.
    /// Use None to disable movement in this direction.
    pub fn rebind_left_key(&mut self, new_key: Option<Key>) {
        self.left_key = new_key;
    }

    /// Set the movement button for right.
    /// Use None to disable movement in this direction.
    pub fn rebind_right_key(&mut self, new_key: Option<Key>) {
        self.right_key = new_key;
    }

    /// Disable the movement buttons for up, down, left and right.
    pub fn unbind_movement_keys(&mut self) {
        self.up_key = None;
        self.down_key = None;
        self.left_key = None;
        self.right_key = None;
    }

    #[doc(hidden)]
    pub fn handle_left_button_displacement(&mut self, dpos: Vec2) {
        self.yaw += dpos.x * self.yaw_step;
        self.pitch += dpos.y * self.pitch_step;

        self.update_restrictions();
        self.update_projviews();
    }

    #[doc(hidden)]
    pub fn handle_right_button_displacement(&mut self, dpos: Vec2) {
        let at = self.at();
        let dir = (at - self.eye).normalize();
        let tangent = self.coord_system.up_axis.cross(dir).normalize();
        let bitangent = dir.cross(tangent);

        self.eye = self.eye + tangent * (0.01 * dpos.x / 10.0) + bitangent * (0.01 * dpos.y / 10.0);
        self.update_restrictions();
        self.update_projviews();
    }

    #[doc(hidden)]
    pub fn handle_scroll(&mut self, yoff: f32) {
        let front = self.observer_frame().rotation * Vec3::Z;

        self.eye += front * (self.move_step * yoff);

        self.update_restrictions();
        self.update_projviews();
    }

    fn update_projviews(&mut self) {
        self.view = self.view_transform().to_mat4();
        let aspect = self.last_framebuffer_size.x / self.last_framebuffer_size.y;
        self.proj = Mat4::perspective_rh_gl(self.fov, aspect, self.znear, self.zfar);
        self.proj_view = self.proj * self.view;
        self.inverse_proj_view = self.proj_view.inverse();
    }

    /// The direction this camera is looking at.
    pub fn eye_dir(&self) -> Vec3 {
        (self.at() - self.eye).normalize()
    }

    /// The direction this camera is being moved by the keyboard keys for a given set of key states.
    pub fn move_dir(&self, up: bool, down: bool, right: bool, left: bool) -> Vec3 {
        let t = self.observer_frame();
        let frontv = t.rotation * Vec3::Z;
        let rightv = t.rotation * Vec3::X;

        let mut movement = Vec3::ZERO;

        if up {
            movement += frontv
        }

        if down {
            movement -= frontv
        }

        if right {
            movement -= rightv
        }

        if left {
            movement += rightv
        }

        if movement == Vec3::ZERO {
            movement
        } else {
            movement.normalize()
        }
    }

    /// Translates in-place this camera by `t`.
    #[inline]
    pub fn translate_mut(&mut self, t: Vec3) {
        let new_eye = self.eye + t;

        self.set_eye(new_eye);
    }

    /// Translates this camera by `t`.
    #[inline]
    pub fn translate(&self, t: Vec3) -> FirstPersonCamera3d {
        let mut res = *self;
        res.translate_mut(t);
        res
    }

    /// Sets the eye of this camera to `eye`.
    #[inline]
    fn set_eye(&mut self, eye: Vec3) {
        self.eye = eye;
        self.update_restrictions();
        self.update_projviews();
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
            let old_at = self.at();
            self.coord_system = new_coord_system;
            self.look_at(self.eye, old_at);
        }
    }

    /// The camera observer local frame.
    fn observer_frame(&self) -> Pose3 {
        Pose3::face_towards(self.eye, self.at(), self.coord_system.up_axis)
    }
}

impl Camera3d for FirstPersonCamera3d {
    fn clip_planes(&self) -> (f32, f32) {
        (self.znear, self.zfar)
    }

    /// The camera view transformation (i-e transformation without projection).
    fn view_transform(&self) -> Pose3 {
        Pose3::look_at_rh(self.eye, self.at(), self.coord_system.up_axis)
    }

    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent) {
        match *event {
            WindowEvent::CursorPos(x, y, _) => {
                let curr_pos = Vec2::new(x as f32, y as f32);

                if let Some(rotate_button) = self.rotate_button {
                    if canvas.get_mouse_button(rotate_button) == Action::Press {
                        let dpos = curr_pos - self.last_cursor_pos;
                        self.handle_left_button_displacement(dpos)
                    }
                }

                if let Some(drag_button) = self.drag_button {
                    if canvas.get_mouse_button(drag_button) == Action::Press {
                        let dpos = curr_pos - self.last_cursor_pos;
                        self.handle_right_button_displacement(dpos)
                    }
                }

                self.last_cursor_pos = curr_pos;
            }
            WindowEvent::Scroll(_, off, _) => self.handle_scroll(off as f32),
            WindowEvent::FramebufferSize(w, h) => {
                self.last_framebuffer_size = Vec2::new(w as f32, h as f32);
                self.update_projviews();
            }
            _ => {}
        }
    }

    fn eye(&self) -> Vec3 {
        self.eye
    }

    fn transformation(&self) -> Mat4 {
        self.proj_view
    }

    fn inverse_transformation(&self) -> Mat4 {
        self.inverse_proj_view
    }

    #[inline]
    fn view_transform_pair(&self, _pass: usize) -> (Pose3, Mat4) {
        (self.view_transform(), self.proj)
    }

    fn update(&mut self, canvas: &Canvas) {
        let up = check_optional_key_state(canvas, self.up_key, Action::Press);
        let down = check_optional_key_state(canvas, self.down_key, Action::Press);
        let right = check_optional_key_state(canvas, self.right_key, Action::Press);
        let left = check_optional_key_state(canvas, self.left_key, Action::Press);
        let dir = self.move_dir(up, down, right, left);

        let move_amount = dir * self.move_step;
        self.translate_mut(move_amount);
    }
}

fn check_optional_key_state(canvas: &Canvas, key: Option<Key>, key_state: Action) -> bool {
    if let Some(actual_key) = key {
        canvas.get_key(actual_key) == key_state
    } else {
        false
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CoordSystemRh {
    pub up_axis: Vec3,
    pub rotation_to_y_up: Rot3,
}

impl CoordSystemRh {
    #[inline]
    pub fn from_up_axis(up_axis: Vec3) -> Self {
        let rotation_to_y_up = Rot3::from_rotation_arc(up_axis.normalize(), Vec3::Y);
        Self {
            up_axis,
            rotation_to_y_up,
        }
    }
}
