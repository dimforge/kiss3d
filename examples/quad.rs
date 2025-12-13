extern crate kiss3d;
extern crate nalgebra as na;
extern crate rand;

use kiss3d::light::Light;
use kiss3d::window::Window;
use rand::random;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: quad").await;

    let mut c = window.add_quad(5.0, 4.0, 100, 100);

    c.set_color(random(), random(), random());

    let mut time = 0.016f32;

    window.set_light(Light::StickToCamera);

    while window.render().await {
        c.modify_vertices(&mut |coords| {
            for v in coords.iter_mut() {
                v.z = time.sin()
                    * (((v.x + time) * 4.0).cos() + time.sin() * ((v.y + time) * 4.0 + time).cos())
                    / 2.0
            }
        });
        c.recompute_normals();

        time = time + 0.016;
    }
}
