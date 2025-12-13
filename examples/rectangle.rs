extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::light::Light;
use kiss3d::window::Window;
use na::UnitComplex;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: rectangle").await;
    let mut c = window.add_rectangle(100.0, 150.0);

    c.set_color(1.0, 0.0, 0.0);

    window.set_light(Light::StickToCamera);

    let rot = UnitComplex::new(0.014);

    while window.render().await {
        c.prepend_to_local_rotation(&rot);
    }
}
