extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::text::Font;
use kiss3d::window::Window;
use na::{Point2, Point3};

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: text").await;
    let font = Font::default();

    while window.render().await {
        window.draw_text(
            "Hello birds!",
            &Point2::origin(),
            120.0,
            &font,
            &Point3::new(0.0, 1.0, 1.0),
        );

        let ascii = " !\"#$%&'`()*+,-_./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^abcdefghijklmnopqrstuvwxyz{|}~";

        window.draw_text(
            ascii,
            &Point2::new(0.0, 120.0),
            60.0,
            &font,
            &Point3::new(1.0, 1.0, 0.0),
        );
    }
}
