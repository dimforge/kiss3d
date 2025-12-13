extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::light::Light;
use kiss3d::window::Window;
use na::{Translation3, UnitQuaternion, Vector3};
use std::f32;
use std::path::Path;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: obj").await;

    // Teapot
    let obj_path = Path::new("examples/media/teapot/teapot.obj");
    let mtl_path = Path::new("examples/media/teapot");
    let mut teapot = window.add_obj(obj_path, mtl_path, Vector3::new(0.001, 0.001, 0.001));
    teapot.append_translation(&Translation3::new(0.0, -0.05, -0.2));

    // Rust logo
    let obj_path = Path::new("examples/media/rust_logo/rust_logo.obj");
    let mtl_path = Path::new("examples/media/rust_logo");
    let mut rust = window.add_obj(obj_path, mtl_path, Vector3::new(0.05, 0.05, 0.05));
    rust.prepend_to_local_rotation(&UnitQuaternion::from_axis_angle(
        &Vector3::x_axis(),
        -f32::consts::FRAC_PI_2,
    ));
    rust.set_color(0.0, 0.0, 1.0);

    window.set_light(Light::StickToCamera);

    let rot_teapot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);
    let rot_rust = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -0.014);

    while window.render().await {
        teapot.prepend_to_local_rotation(&rot_teapot);
        rust.prepend_to_local_rotation(&rot_rust);
    }
}
