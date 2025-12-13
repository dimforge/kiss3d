use std::f32;

use na::{self, Isometry3, Matrix4, Perspective3, Point2, Point3, Vector2, Vector3};

use crate::camera::Camera;
use crate::event::{Action, Key, MouseButton, WindowEvent};
use crate::window::Canvas;

/// First-person camera mode.
///
///   * Left button press + drag - look around
///   * Right button press + drag - translates the camera position on the plane orthogonal to the
///     view direction
///   * Scroll in/out - zoom in/out
#[derive(Debug)]
pub struct FirstPersonStereo {
    /// The camera position
    eye: Point3<f32>,
    eye_left: Point3<f32>,
    eye_right: Point3<f32>,

    /// Inter Pupilary Distance
    ipd: f32,

    /// Yaw of the camera (rotation along the y axis).
    yaw: f32,
    /// Pitch of the camera (rotation along the x axis).
    pitch: f32,

    /// Increment of the yaw per unit mouse movement. The default value is 0.005.
    yaw_step: f32,
    /// Increment of the pitch per unit mouse movement. The default value is 0.005.
    pitch_step: f32,
    /// Increment of the translation per arrow press. The default value is 0.1.
    move_step: f32,

    /// Low level data
    projection: Perspective3<f32>,
    view_left: Matrix4<f32>,
    view_right: Matrix4<f32>,
    proj: Matrix4<f32>,
    proj_view: Matrix4<f32>,
    inverse_proj_view: Matrix4<f32>,
    last_cursor_pos: Point2<f32>,
}

impl FirstPersonStereo {
    /// Creates a first person camera with default sensitivity values.
    pub fn new(eye: Point3<f32>, at: Point3<f32>, ipd: f32) -> FirstPersonStereo {
        FirstPersonStereo::new_with_frustum(f32::consts::PI / 4.0, 0.1, 1024.0, eye, at, ipd)
    }

    /// Creates a new first person camera with default sensitivity values.
    pub fn new_with_frustum(
        fov: f32,
        znear: f32,
        zfar: f32,
        eye: Point3<f32>,
        at: Point3<f32>,
        ipd: f32,
    ) -> FirstPersonStereo {
        let mut res = FirstPersonStereo {
            eye: Point3::new(0.0, 0.0, 0.0),
            // left & right are initially wrong, don't take ipd into account
            eye_left: Point3::new(0.0, 0.0, 0.0),
            eye_right: Point3::new(0.0, 0.0, 0.0),
            ipd,
            yaw: 0.0,
            pitch: 0.0,
            yaw_step: 0.005,
            pitch_step: 0.005,
            move_step: 0.5,
            projection: Perspective3::new(800.0 / 600.0, fov, znear, zfar),
            proj_view: na::zero(),
            inverse_proj_view: na::zero(),
            last_cursor_pos: Point2::origin(),
            proj: na::zero(),
            view_left: na::zero(),
            view_right: na::zero(),
        };

        res.look_at(eye, at);

        res
    }

    /// Changes the orientation and position of the camera to look at the specified point.
    pub fn look_at(&mut self, eye: Point3<f32>, at: Point3<f32>) {
        let dist = (eye - at).norm();

        let pitch = ((at.y - eye.y) / dist).acos();
        let yaw = (at.z - eye.z).atan2(at.x - eye.x);

        self.eye = eye;
        self.yaw = yaw;
        self.pitch = pitch;
        self.update_projviews();
    }

    /// The point the camera is looking at.
    pub fn at(&self) -> Point3<f32> {
        let ax = self.eye.x + self.yaw.cos() * self.pitch.sin();
        let ay = self.eye.y + self.pitch.cos();
        let az = self.eye.z + self.yaw.sin() * self.pitch.sin();

        Point3::new(ax, ay, az)
    }

    fn update_restrictions(&mut self) {
        if self.pitch <= 0.0001 {
            self.pitch = 0.0001
        }

        let _pi: f32 = f32::consts::PI;
        if self.pitch > _pi - 0.0001 {
            self.pitch = _pi - 0.0001
        }
    }

    #[doc(hidden)]
    pub fn handle_left_button_displacement(&mut self, dpos: &Vector2<f32>) {
        self.yaw += dpos.x * self.yaw_step;
        self.pitch += dpos.y * self.pitch_step;

        self.update_restrictions();
        self.update_projviews();
    }

    fn update_eyes_location(&mut self) {
        // left and right are on a line perpendicular to both up and the target
        // up is always y
        let dir = (self.at() - self.eye).normalize();
        let tangent = Vector3::y().cross(&dir).normalize();
        self.eye_left = self.eye - tangent * (self.ipd / 2.0);
        self.eye_right = self.eye + tangent * (self.ipd / 2.0);
        //println(fmt!("eye_left = %f,%f,%f", self.eye_left.x as float, self.eye_left.y as float, self.eye_left.z as float));
        //println(fmt!("eye_right = %f,%f,%f", self.eye_right.x as float, self.eye_right.y as float, self.eye_right.z as float));
        // TODO: verify with an assert or something that the distance between the eyes is ipd, just to make me feel good.
    }

    #[doc(hidden)]
    pub fn handle_right_button_displacement(&mut self, dpos: &Vector2<f32>) {
        let at = self.at();
        let dir = (at - self.eye).normalize();
        let tangent = Vector3::y().cross(&dir).normalize();
        let bitangent = dir.cross(&tangent);

        self.eye = self.eye + tangent * (0.01 * dpos.x / 10.0) + bitangent * (0.01 * dpos.y / 10.0);
        // TODO: ugly - should move eye update to where eye_left & eye_right are updated
        self.update_eyes_location();
        self.update_restrictions();
        self.update_projviews();
    }

    #[doc(hidden)]
    pub fn handle_scroll(&mut self, yoff: f32) {
        let front: Vector3<f32> = self.view_transform() * Vector3::z();

        self.eye += front * (self.move_step * yoff);

        self.update_eyes_location();
        self.update_restrictions();
        self.update_projviews();
    }

    fn update_projviews(&mut self) {
        self.proj_view = *self.projection.as_matrix() * self.view_transform().to_homogeneous();
        self.inverse_proj_view = self.proj_view.try_inverse().unwrap();
        self.proj = *self.projection.as_matrix();
        self.view_left = self.view_transform_left().to_homogeneous();
        self.view_right = self.view_transform_right().to_homogeneous();
    }

    #[allow(dead_code)]
    fn view_eye(&self, eye: usize) -> Matrix4<f32> {
        match eye {
            0usize => self.view_left,
            1usize => self.view_right,
            _ => panic!("bad eye index"),
        }
    }

    /// The left eye camera view transformation
    fn view_transform_left(&self) -> Isometry3<f32> {
        Isometry3::look_at_rh(&self.eye_left, &self.at(), &Vector3::y())
    }

    /// The right eye camera view transformation
    fn view_transform_right(&self) -> Isometry3<f32> {
        Isometry3::look_at_rh(&self.eye_right, &self.at(), &Vector3::y())
    }

    /// return Inter Pupilary Distance
    pub fn ipd(&self) -> f32 {
        self.ipd
    }

    /// change Inter Pupilary Distance
    pub fn set_ipd(&mut self, ipd: f32) {
        self.ipd = ipd;

        self.update_eyes_location();
        self.update_restrictions();
        self.update_projviews();
    }
}

impl Camera for FirstPersonStereo {
    fn clip_planes(&self) -> (f32, f32) {
        (self.projection.znear(), self.projection.zfar())
    }

    /// The imaginary middle eye camera view transformation (i-e transformation without projection).
    fn view_transform(&self) -> Isometry3<f32> {
        Isometry3::look_at_rh(&self.eye, &self.at(), &Vector3::y())
    }

    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent) {
        match *event {
            WindowEvent::CursorPos(x, y, _) => {
                let curr_pos = Point2::new(x as f32, y as f32);

                if canvas.get_mouse_button(MouseButton::Button1) == Action::Press {
                    let dpos = curr_pos - self.last_cursor_pos;
                    self.handle_left_button_displacement(&dpos)
                }

                if canvas.get_mouse_button(MouseButton::Button2) == Action::Press {
                    let dpos = curr_pos - self.last_cursor_pos;
                    self.handle_right_button_displacement(&dpos)
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

    fn update(&mut self, canvas: &Canvas) {
        let t = self.view_transform();
        let front = t * Vector3::z();
        let right = t * Vector3::x();

        if canvas.get_key(Key::Up) == Action::Press {
            self.eye += front * self.move_step
        }

        if canvas.get_key(Key::Down) == Action::Press {
            self.eye += front * (-self.move_step)
        }

        if canvas.get_key(Key::Right) == Action::Press {
            self.eye += right * (-self.move_step)
        }

        if canvas.get_key(Key::Left) == Action::Press {
            self.eye += right * self.move_step
        }

        self.update_eyes_location();
        self.update_restrictions();
        self.update_projviews();
    }

    fn view_transform_pair(&self, pass: usize) -> (Isometry3<f32>, Matrix4<f32>) {
        let view = match pass {
            0 => self.view_transform_left(),
            1 => self.view_transform_right(),
            _ => self.view_transform(),
        };
        (view, self.proj)
    }

    fn num_passes(&self) -> usize {
        2usize
    }

    // Note: In wgpu, viewport/scissor are set per render pass, not globally.
    // The stereo camera's start_pass and render_complete functionality would need
    // to be handled differently in wgpu (e.g., through separate render passes
    // or by storing viewport info for materials to use).
    fn start_pass(&self, _pass: usize, _canvas: &Canvas) {
        // TODO: Viewport handling needs to be done at render pass creation in wgpu
    }

    fn render_complete(&self, _canvas: &Canvas) {
        // TODO: Viewport reset handled differently in wgpu
    }
}
