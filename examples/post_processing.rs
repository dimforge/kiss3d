extern crate kiss3d;
extern crate nalgebra as na;
extern crate rand;

use kiss3d::light::Light;
#[cfg(not(target_arch = "wasm32"))]
use kiss3d::post_processing::SobelEdgeHighlight;
use kiss3d::post_processing::{Grayscales, Waves};
use kiss3d::window::Window;
use na::Translation3;
use rand::random;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: post_processing").await;

    let mut c = window.add_cube(1.0, 1.0, 1.0);
    let mut b = window.add_sphere(0.5);
    let mut p = window.add_cone(0.5, 1.0);
    let mut y = window.add_cylinder(0.5, 1.0);
    let mut a = window.add_capsule(0.5, 1.0);

    c.append_translation(&Translation3::new(2.0, 0.0, 0.0));
    b.append_translation(&Translation3::new(4.0, 0.0, 0.0));
    p.append_translation(&Translation3::new(-2.0, 0.0, 0.0));
    y.append_translation(&Translation3::new(-4.0, 0.0, 0.0));
    a.append_translation(&Translation3::new(0.0, 0.0, 0.0));

    c.set_color(random(), random(), random());
    b.set_color(random(), random(), random());
    p.set_color(random(), random(), random());
    y.set_color(random(), random(), random());
    a.set_color(random(), random(), random());

    #[cfg(not(target_arch = "wasm32"))]
    let mut sobel = SobelEdgeHighlight::new(4.0);
    let mut waves = Waves::new();
    let mut grays = Grayscales::new();

    window.set_background_color(1.0, 1.0, 1.0);
    window.set_light(Light::StickToCamera);
    window.set_framerate_limit(Some(60));

    let mut time = 0usize;
    let mut counter = 0usize;

    while !window.should_close() {
        if time % 200 == 0 {
            time = 0;
            counter = (counter + 1) % 4;
        }

        time = time + 1;

        let _ = match counter {
            0 => window.render().await,
            1 => window.render_with_effect(&mut grays).await,
            2 => window.render_with_effect(&mut waves).await,
            #[cfg(not(target_arch = "wasm32"))]
            3 => window.render_with_effect(&mut sobel).await,
            _ => true,
        };
    }
}
