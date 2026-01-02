use std::f32;

use glamx::{Mat4, Pose3, Vec2, Vec3};

use crate::camera::Camera3d;
use crate::event::{Action, Key, MouseButton, WindowEvent};
use crate::window::Canvas;

/// First-person camera mode.
///
///   * Left button press + drag - look around
///   * Right button press + drag - translates the camera position on the plane orthogonal to the
///     view direction
///   * Scroll in/out - zoom in/out
#[derive(Copy, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FirstPersonCamera3dStereo {
    /// The camera position
    eye: Vec3,
    eye_left: Vec3,
    eye_right: Vec3,

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
    fov: f32,
    znear: f32,
    zfar: f32,
    view_left: Mat4,
    view_right: Mat4,
    proj: Mat4,
    proj_view: Mat4,
    inverse_proj_view: Mat4,
    last_cursor_pos: Vec2,
    last_framebuffer_size: Vec2,
}

impl FirstPersonCamera3dStereo {
    /// Creates a first person camera with default sensitivity values.
    pub fn new(eye: Vec3, at: Vec3, ipd: f32) -> FirstPersonCamera3dStereo {
        FirstPersonCamera3dStereo::new_with_frustum(
            f32::consts::PI / 4.0,
            0.1,
            1024.0,
            eye,
            at,
            ipd,
        )
    }

    /// Creates a new first person camera with default sensitivity values.
    pub fn new_with_frustum(
        fov: f32,
        znear: f32,
        zfar: f32,
        eye: Vec3,
        at: Vec3,
        ipd: f32,
    ) -> FirstPersonCamera3dStereo {
        let mut res = FirstPersonCamera3dStereo {
            eye: Vec3::ZERO,
            // left & right are initially wrong, don't take ipd into account
            eye_left: Vec3::ZERO,
            eye_right: Vec3::ZERO,
            ipd,
            yaw: 0.0,
            pitch: 0.0,
            yaw_step: 0.005,
            pitch_step: 0.005,
            move_step: 0.5,
            fov,
            znear,
            zfar,
            proj_view: Mat4::IDENTITY,
            inverse_proj_view: Mat4::IDENTITY,
            last_cursor_pos: Vec2::ZERO,
            last_framebuffer_size: Vec2::new(800.0, 600.0),
            proj: Mat4::IDENTITY,
            view_left: Mat4::IDENTITY,
            view_right: Mat4::IDENTITY,
        };

        res.look_at(eye, at);

        res
    }

    /// Changes the orientation and position of the camera to look at the specified point.
    pub fn look_at(&mut self, eye: Vec3, at: Vec3) {
        let dist = (eye - at).length();

        let pitch = ((at.y - eye.y) / dist).acos();
        let yaw = (at.z - eye.z).atan2(at.x - eye.x);

        self.eye = eye;
        self.yaw = yaw;
        self.pitch = pitch;
        self.update_eyes_location();
        self.update_projviews();
    }

    /// The point the camera is looking at.
    pub fn at(&self) -> Vec3 {
        let ax = self.eye.x + self.yaw.cos() * self.pitch.sin();
        let ay = self.eye.y + self.pitch.cos();
        let az = self.eye.z + self.yaw.sin() * self.pitch.sin();

        Vec3::new(ax, ay, az)
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
    pub fn handle_left_button_displacement(&mut self, dpos: Vec2) {
        self.yaw += dpos.x * self.yaw_step;
        self.pitch += dpos.y * self.pitch_step;

        self.update_restrictions();
        self.update_projviews();
    }

    fn update_eyes_location(&mut self) {
        // left and right are on a line perpendicular to both up and the target
        // up is always y
        let dir = (self.at() - self.eye).normalize();
        let tangent = Vec3::Y.cross(dir).normalize();
        self.eye_left = self.eye - tangent * (self.ipd / 2.0);
        self.eye_right = self.eye + tangent * (self.ipd / 2.0);
        //println(fmt!("eye_left = %f,%f,%f", self.eye_left.x as float, self.eye_left.y as float, self.eye_left.z as float));
        //println(fmt!("eye_right = %f,%f,%f", self.eye_right.x as float, self.eye_right.y as float, self.eye_right.z as float));
        // TODO: verify with an assert or something that the distance between the eyes is ipd, just to make me feel good.
    }

    #[doc(hidden)]
    pub fn handle_right_button_displacement(&mut self, dpos: Vec2) {
        let at = self.at();
        let dir = (at - self.eye).normalize();
        let tangent = Vec3::Y.cross(dir).normalize();
        let bitangent = dir.cross(tangent);

        self.eye = self.eye + tangent * (0.01 * dpos.x / 10.0) + bitangent * (0.01 * dpos.y / 10.0);
        // TODO: ugly - should move eye update to where eye_left & eye_right are updated
        self.update_eyes_location();
        self.update_restrictions();
        self.update_projviews();
    }

    #[doc(hidden)]
    pub fn handle_scroll(&mut self, yoff: f32) {
        let front = self.view_transform().rotation * Vec3::Z;

        self.eye += front * (self.move_step * yoff);

        self.update_eyes_location();
        self.update_restrictions();
        self.update_projviews();
    }

    fn update_projviews(&mut self) {
        let aspect = self.last_framebuffer_size.x / self.last_framebuffer_size.y;
        self.proj = Mat4::perspective_rh_gl(self.fov, aspect, self.znear, self.zfar);
        self.proj_view = self.proj * self.view_transform().to_mat4();
        self.inverse_proj_view = self.proj_view.inverse();
        self.view_left = self.view_transform_left().to_mat4();
        self.view_right = self.view_transform_right().to_mat4();
    }

    #[allow(dead_code)]
    fn view_eye(&self, eye: usize) -> Mat4 {
        match eye {
            0usize => self.view_left,
            1usize => self.view_right,
            _ => panic!("bad eye index"),
        }
    }

    /// The left eye camera view transformation
    fn view_transform_left(&self) -> Pose3 {
        Pose3::look_at_rh(self.eye_left, self.at(), Vec3::Y)
    }

    /// The right eye camera view transformation
    fn view_transform_right(&self) -> Pose3 {
        Pose3::look_at_rh(self.eye_right, self.at(), Vec3::Y)
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

impl Camera3d for FirstPersonCamera3dStereo {
    fn clip_planes(&self) -> (f32, f32) {
        (self.znear, self.zfar)
    }

    /// The imaginary middle eye camera view transformation (i-e transformation without projection).
    fn view_transform(&self) -> Pose3 {
        Pose3::look_at_rh(self.eye, self.at(), Vec3::Y)
    }

    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent) {
        match *event {
            WindowEvent::CursorPos(x, y, _) => {
                let curr_pos = Vec2::new(x as f32, y as f32);

                if canvas.get_mouse_button(MouseButton::Button1) == Action::Press {
                    let dpos = curr_pos - self.last_cursor_pos;
                    self.handle_left_button_displacement(dpos)
                }

                if canvas.get_mouse_button(MouseButton::Button2) == Action::Press {
                    let dpos = curr_pos - self.last_cursor_pos;
                    self.handle_right_button_displacement(dpos)
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

    fn update(&mut self, canvas: &Canvas) {
        let t = self.view_transform();
        let front = t.rotation * Vec3::Z;
        let right = t.rotation * Vec3::X;

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

    fn view_transform_pair(&self, pass: usize) -> (Pose3, Mat4) {
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
