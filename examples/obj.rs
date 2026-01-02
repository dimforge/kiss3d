use kiss3d::prelude::*;
use std::f32;
use std::path::Path;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: obj").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 0.0, -0.7), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, -10.0));

    // Teapot
    let obj_path = Path::new("examples/media/teapot/teapot.obj");
    let mtl_path = Path::new("examples/media/teapot");
    let mut teapot = scene
        .add_obj(obj_path, mtl_path, Vec3::new(0.001, 0.001, 0.001))
        .set_position(Vec3::new(0.0, -0.05, -0.2));

    // Rust logo
    let obj_path = Path::new("examples/media/rust_logo/rust_logo.obj");
    let mtl_path = Path::new("examples/media/rust_logo");
    let mut rust = scene
        .add_obj(obj_path, mtl_path, Vec3::new(0.05, 0.05, 0.05))
        .set_rotation(Quat::from_axis_angle(Vec3::X, -f32::consts::FRAC_PI_2))
        .set_color(BLUE);

    let rot_teapot = Quat::from_axis_angle(Vec3::Y, 0.014);
    let rot_rust = Quat::from_axis_angle(Vec3::Y, -0.014);

    while window.render_3d(&mut scene, &mut camera).await {
        teapot.rotate(rot_teapot);
        rust.prepend_rotation(rot_rust);
    }
}
