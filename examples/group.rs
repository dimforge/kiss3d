extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::light::Light;
use kiss3d::window::Window;
use na::{Translation3, UnitQuaternion, Vector3};

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: cube").await;

    let mut g1 = window.add_group();
    let mut g2 = window.add_group();

    g1.append_translation(&Translation3::new(2.0f32, 0.0, 0.0));
    g2.append_translation(&Translation3::new(-2.0f32, 0.0, 0.0));

    g1.add_cube(1.0, 5.0, 1.0);
    g1.add_cube(5.0, 1.0, 1.0);

    g2.add_cube(1.0, 5.0, 1.0);
    g2.add_cube(1.0, 1.0, 5.0);

    g1.set_color(1.0, 0.0, 0.0);
    g2.set_color(0.0, 1.0, 0.0);

    window.set_light(Light::StickToCamera);

    let rot1 = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);
    let rot2 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.014);

    while window.render().await {
        g1.prepend_to_local_rotation(&rot1);
        g2.prepend_to_local_rotation(&rot2);
    }
}
