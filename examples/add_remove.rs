extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::light::Light;
use kiss3d::window::Window;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: add_remove").await;
    let mut c = window.add_cube(1.0, 1.0, 1.0);
    let mut added = true;

    window.set_light(Light::StickToCamera);

    while window.render().await {
        if added {
            window.remove_node(&mut c);
        } else {
            c = window.add_cube(1.0, 1.0, 1.0);
            c.set_color(1.0, 0.0, 0.0);
        }

        added = !added;
    }
}
