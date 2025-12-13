extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::light::Light;
use kiss3d::window::Window;
use na::{Point2, Point3};

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: planar lines").await;

    window.set_light(Light::StickToCamera);

    while window.render().await {
        let a = Point2::new(-200.0, -200.0);
        let b = Point2::new(0.0, 200.0);
        let c = Point2::new(200.0, -200.0);

        window.draw_planar_line(&a, &b, &Point3::new(1.0, 0.0, 0.0), 2.0);
        window.draw_planar_line(&b, &c, &Point3::new(0.0, 1.0, 0.0), 2.0);
        window.draw_planar_line(&c, &a, &Point3::new(0.0, 0.0, 1.0), 2.0);
    }
}
