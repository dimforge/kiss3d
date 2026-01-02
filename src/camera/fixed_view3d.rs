use crate::camera::Camera3d;
use crate::event::WindowEvent;
use crate::window::Canvas;
use glamx::{Mat4, Pose3, Vec3};
use std::f32;

/// A camera that cannot move.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FixedView3d {
    fov: f32,
    znear: f32,
    zfar: f32,
    proj: Mat4,
    inv_proj: Mat4,
    last_framebuffer_size: (f32, f32),
}

impl Default for FixedView3d {
    fn default() -> Self {
        Self::new()
    }
}

impl FixedView3d {
    /// Create a new static camera.
    pub fn new() -> FixedView3d {
        FixedView3d::new_with_frustum(f32::consts::PI / 4.0, 0.1, 1024.0)
    }

    /// Creates a new arc ball camera with default sensitivity values.
    pub fn new_with_frustum(fov: f32, znear: f32, zfar: f32) -> FixedView3d {
        let mut res = FixedView3d {
            fov,
            znear,
            zfar,
            proj: Mat4::IDENTITY,
            inv_proj: Mat4::IDENTITY,
            last_framebuffer_size: (800.0, 600.0),
        };
        res.update_projviews();
        res
    }

    fn update_projviews(&mut self) {
        let aspect = self.last_framebuffer_size.0 / self.last_framebuffer_size.1;
        self.proj = Mat4::perspective_rh_gl(self.fov, aspect, self.znear, self.zfar);
        self.inv_proj = self.proj.inverse();
    }
}

impl Camera3d for FixedView3d {
    fn clip_planes(&self) -> (f32, f32) {
        (self.znear, self.zfar)
    }

    fn view_transform(&self) -> Pose3 {
        Pose3::IDENTITY
    }

    fn eye(&self) -> Vec3 {
        Vec3::ZERO
    }

    fn handle_event(&mut self, _: &Canvas, event: &WindowEvent) {
        if let WindowEvent::FramebufferSize(w, h) = *event {
            self.last_framebuffer_size = (w as f32, h as f32);
            self.update_projviews();
        }
    }

    #[inline]
    fn view_transform_pair(&self, _pass: usize) -> (Pose3, Mat4) {
        (Pose3::IDENTITY, self.proj)
    }

    fn transformation(&self) -> Mat4 {
        self.proj
    }

    fn inverse_transformation(&self) -> Mat4 {
        self.inv_proj
    }

    fn update(&mut self, _: &Canvas) {}
}
