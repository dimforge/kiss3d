extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::light::Light;
use kiss3d::window::Window;
use na::{Point2, Point3};

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: points").await;

    window.set_light(Light::StickToCamera);

    while window.render().await {
        let a = Point3::new(-0.1, -0.1, 0.0);
        let b = Point3::new(0.0, 0.1, 0.0);
        let c = Point3::new(0.1, -0.1, 0.0);
        let red = Point3::new(1.0, 0.0, 0.0);
        let blue = Point3::new(0.0, 1.0, 0.0);
        let green = Point3::new(0.0, 0.0, 1.0);

        window.draw_point(&a, &red, 5.0);
        window.draw_point(&b, &blue, 15.0);
        window.draw_point(&c, &green, 25.0);

        window.draw_planar_point(&Point2::new(-50.0, -200.0), &red, 5.0);
        window.draw_planar_point(&Point2::new(0.0, -200.0), &blue, 15.0);
        window.draw_planar_point(&Point2::new(50.0, -200.0), &green, 25.0);
    }
}
