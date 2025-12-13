use crate::camera::Camera;
use crate::event::{Action, Key, MouseButton, WindowEvent};
use crate::window::Canvas;
use na::{
    self, Isometry3, Matrix4, Perspective3, Point3, Translation3, Unit, UnitQuaternion, Vector2,
    Vector3,
};
use num::Zero;
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
/// # use kiss3d::camera::FirstPerson;
/// # use kiss3d::window::Window;
/// # use nalgebra::Point3;
/// # #[kiss3d::main]
/// # async fn main() {
/// # let mut window = Window::new("Example");
/// let mut camera = FirstPerson::new(
///     Point3::new(0.0, 1.0, 5.0),  // Eye position
///     Point3::origin()              // Looking at origin
/// );
/// // Use with window.render_with_camera(&mut camera).await
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FirstPerson {
    eye: Point3<f32>,
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

    projection: Perspective3<f32>,
    proj: Matrix4<f32>,
    view: Matrix4<f32>,
    proj_view: Matrix4<f32>,
    inverse_proj_view: Matrix4<f32>,
    last_cursor_pos: Vector2<f32>,
    coord_system: CoordSystemRh,
}

impl FirstPerson {
    /// Creates a new first-person camera with default settings.
    ///
    /// Default frustum: 45° field of view, near plane at 0.1, far plane at 1024.
    ///
    /// # Arguments
    /// * `eye` - Initial camera position
    /// * `at` - Initial point to look at
    ///
    /// # Returns
    /// A new `FirstPerson` camera instance
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::camera::FirstPerson;
    /// # use nalgebra::Point3;
    /// let camera = FirstPerson::new(
    ///     Point3::new(0.0, 5.0, 10.0),
    ///     Point3::origin()
    /// );
    /// ```
    pub fn new(eye: Point3<f32>, at: Point3<f32>) -> FirstPerson {
        FirstPerson::new_with_frustum(f32::consts::PI / 4.0, 0.1, 1024.0, eye, at)
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
    /// A new `FirstPerson` camera instance
    pub fn new_with_frustum(
        fov: f32,
        znear: f32,
        zfar: f32,
        eye: Point3<f32>,
        at: Point3<f32>,
    ) -> FirstPerson {
        let mut res = FirstPerson {
            eye: Point3::new(0.0, 0.0, 0.0),
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
            projection: Perspective3::new(800.0 / 600.0, fov, znear, zfar),
            proj: na::zero(),
            view: na::zero(),
            proj_view: na::zero(),
            inverse_proj_view: na::zero(),
            last_cursor_pos: na::zero(),
            coord_system: CoordSystemRh::from_up_axis(Vector3::y_axis()),
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
    pub fn look_at(&mut self, eye: Point3<f32>, at: Point3<f32>) {
        let dist = (eye - at).norm();

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
    pub fn at(&self) -> Point3<f32> {
        let view_eye = self.coord_system.rotation_to_y_up * self.eye;
        let ax = view_eye.x + self.yaw.cos() * self.pitch.sin();
        let ay = view_eye.y + self.pitch.cos();
        let az = view_eye.z + self.yaw.sin() * self.pitch.sin();
        self.coord_system.rotation_to_y_up.inverse() * Point3::new(ax, ay, az)
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

    /// The button used to rotate the FirstPerson camera.
    pub fn rotate_button(&self) -> Option<MouseButton> {
        self.rotate_button
    }

    /// Set the button used to rotate the FirstPerson camera.
    /// Use None to disable rotation.
    pub fn rebind_rotate_button(&mut self, new_button: Option<MouseButton>) {
        self.rotate_button = new_button;
    }

    /// The button used to drag the FirstPerson camera.
    pub fn drag_button(&self) -> Option<MouseButton> {
        self.drag_button
    }

    /// Set the button used to drag the FirstPerson camera.
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
    pub fn handle_left_button_displacement(&mut self, dpos: &Vector2<f32>) {
        self.yaw += dpos.x * self.yaw_step;
        self.pitch += dpos.y * self.pitch_step;

        self.update_restrictions();
        self.update_projviews();
    }

    #[doc(hidden)]
    pub fn handle_right_button_displacement(&mut self, dpos: &Vector2<f32>) {
        let at = self.at();
        let dir = (at - self.eye).normalize();
        let tangent = self.coord_system.up_axis.cross(&dir).normalize();
        let bitangent = dir.cross(&tangent);

        self.eye = self.eye + tangent * (0.01 * dpos.x / 10.0) + bitangent * (0.01 * dpos.y / 10.0);
        self.update_restrictions();
        self.update_projviews();
    }

    #[doc(hidden)]
    pub fn handle_scroll(&mut self, yoff: f32) {
        let front = self.observer_frame() * Vector3::z();

        self.eye += front * (self.move_step * yoff);

        self.update_restrictions();
        self.update_projviews();
    }

    fn update_projviews(&mut self) {
        self.view = self.view_transform().to_homogeneous();
        self.proj = *self.projection.as_matrix();
        self.proj_view = self.proj * self.view;
        let _ = self
            .proj_view
            .try_inverse()
            .map(|inverse_proj| self.inverse_proj_view = inverse_proj);
    }

    /// The direction this camera is looking at.
    pub fn eye_dir(&self) -> Vector3<f32> {
        (self.at() - self.eye).normalize()
    }

    /// The direction this camera is being moved by the keyboard keys for a given set of key states.
    pub fn move_dir(&self, up: bool, down: bool, right: bool, left: bool) -> Vector3<f32> {
        let t = self.observer_frame();
        let frontv = t * Vector3::z();
        let rightv = t * Vector3::x();

        let mut movement = na::zero::<Vector3<f32>>();

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

        if movement.is_zero() {
            movement
        } else {
            movement.normalize()
        }
    }

    /// Translates in-place this camera by `t`.
    #[inline]
    pub fn translate_mut(&mut self, t: &Translation3<f32>) {
        let new_eye = t * self.eye;

        self.set_eye(new_eye);
    }

    /// Translates this camera by `t`.
    #[inline]
    pub fn translate(&self, t: &Translation3<f32>) -> FirstPerson {
        let mut res = self.clone();
        res.translate_mut(t);
        res
    }

    /// Sets the eye of this camera to `eye`.
    #[inline]
    fn set_eye(&mut self, eye: Point3<f32>) {
        self.eye = eye;
        self.update_restrictions();
        self.update_projviews();
    }

    /// Sets the up vector of this camera. Prefer using [`set_up_axis_dir`](#method.set_up_axis_dir)
    /// if your up vector is already normalized.
    #[inline]
    pub fn set_up_axis(&mut self, up_axis: Vector3<f32>) {
        self.set_up_axis_dir(Unit::new_normalize(up_axis));
    }

    /// Sets the up-axis direction of this camera.
    #[inline]
    pub fn set_up_axis_dir(&mut self, up_axis: Unit<Vector3<f32>>) {
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
    fn observer_frame(&self) -> Isometry3<f32> {
        Isometry3::face_towards(&self.eye, &self.at(), &self.coord_system.up_axis)
    }
}

impl Camera for FirstPerson {
    fn clip_planes(&self) -> (f32, f32) {
        (self.projection.znear(), self.projection.zfar())
    }

    /// The camera view transformation (i-e transformation without projection).
    fn view_transform(&self) -> Isometry3<f32> {
        Isometry3::look_at_rh(&self.eye, &self.at(), &self.coord_system.up_axis)
    }

    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent) {
        match *event {
            WindowEvent::CursorPos(x, y, _) => {
                let curr_pos = Vector2::new(x as f32, y as f32);

                if let Some(rotate_button) = self.rotate_button {
                    if canvas.get_mouse_button(rotate_button) == Action::Press {
                        let dpos = curr_pos - self.last_cursor_pos;
                        self.handle_left_button_displacement(&dpos)
                    }
                }

                if let Some(drag_button) = self.drag_button {
                    if canvas.get_mouse_button(drag_button) == Action::Press {
                        let dpos = curr_pos - self.last_cursor_pos;
                        self.handle_right_button_displacement(&dpos)
                    }
                }

                self.last_cursor_pos = curr_pos;
            }
            WindowEvent::Scroll(_, off, _) => self.handle_scroll(off as f32),
            WindowEvent::FramebufferSize(w, h) => {
                self.projection.set_aspect(w as f32 / h as f32);
                self.update_projviews();
            }
            _ => {}
        }
    }

    fn eye(&self) -> Point3<f32> {
        self.eye
    }

    fn transformation(&self) -> Matrix4<f32> {
        self.proj_view
    }

    fn inverse_transformation(&self) -> Matrix4<f32> {
        self.inverse_proj_view
    }

    #[inline]
    fn view_transform_pair(&self, _pass: usize) -> (Isometry3<f32>, Matrix4<f32>) {
        (self.view_transform(), self.proj)
    }

    fn update(&mut self, canvas: &Canvas) {
        let up = check_optional_key_state(canvas, self.up_key, Action::Press);
        let down = check_optional_key_state(canvas, self.down_key, Action::Press);
        let right = check_optional_key_state(canvas, self.right_key, Action::Press);
        let left = check_optional_key_state(canvas, self.left_key, Action::Press);
        let dir = self.move_dir(up, down, right, left);

        let move_amount = dir * self.move_step;
        self.translate_mut(&Translation3::from(move_amount));
    }
}

fn check_optional_key_state(canvas: &Canvas, key: Option<Key>, key_state: Action) -> bool {
    if let Some(actual_key) = key {
        canvas.get_key(actual_key) == key_state
    } else {
        false
    }
}

#[derive(Clone, Copy, Debug)]
struct CoordSystemRh {
    up_axis: Unit<Vector3<f32>>,
    rotation_to_y_up: UnitQuaternion<f32>,
}

impl CoordSystemRh {
    #[inline]
    fn from_up_axis(up_axis: Unit<Vector3<f32>>) -> Self {
        let rotation_to_y_up = UnitQuaternion::rotation_between_axis(&up_axis, &Vector3::y_axis())
            .unwrap_or_else(|| {
                UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f32::consts::PI)
            });
        Self {
            up_axis,
            rotation_to_y_up,
        }
    }
}
