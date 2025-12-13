extern crate kiss3d;
extern crate nalgebra as na;

use std::path::Path;

use kiss3d::light::Light;
use kiss3d::window::Window;
use na::{UnitQuaternion, Vector3};

// Based on cube example.
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: screenshot").await;
    let mut c = window.add_cube(0.2, 0.2, 0.2);

    c.set_color(1.0, 0.0, 0.0);
    c.prepend_to_local_rotation(&UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.785));
    c.prepend_to_local_rotation(&UnitQuaternion::from_axis_angle(
        &Vector3::x_axis(),
        -0.6f32,
    ));

    window.set_light(Light::StickToCamera);

    while window.render().await {
        let img = window.snap_image();
        let img_path = Path::new("screenshot.png");
        img.save(img_path).unwrap();
        println!("Screenshot saved to `screenshot.png`");
        break;
    }
}
