#[cfg(not(target_arch = "wasm32"))]
use kiss3d::post_processing::SobelEdgeHighlight;
use kiss3d::post_processing::{Grayscales, Waves};
use kiss3d::prelude::*;
use rand::random;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: post_processing").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));

    scene
        .add_cube(1.0, 1.0, 1.0)
        .translate(Vec3::new(2.0, 0.0, 0.0))
        .set_color(Color::new(random(), random(), random(), 1.0));
    scene
        .add_sphere(0.5)
        .translate(Vec3::new(4.0, 0.0, 0.0))
        .set_color(Color::new(random(), random(), random(), 1.0));
    scene
        .add_cone(0.5, 1.0)
        .translate(Vec3::new(-2.0, 0.0, 0.0))
        .set_color(Color::new(random(), random(), random(), 1.0));
    scene
        .add_cylinder(0.5, 1.0)
        .translate(Vec3::new(-4.0, 0.0, 0.0))
        .set_color(Color::new(random(), random(), random(), 1.0));
    scene
        .add_capsule(0.5, 1.0)
        .set_color(Color::new(random(), random(), random(), 1.0));

    #[cfg(not(target_arch = "wasm32"))]
    let mut sobel = SobelEdgeHighlight::new(4.0);
    let mut waves = Waves::new();
    let mut grays = Grayscales::new();

    window.set_background_color(WHITE);

    let mut time = 0usize;
    let mut counter = 0usize;

    while !window.should_close() {
        if time % 200 == 0 {
            time = 0;
            counter = (counter + 1) % 4;
        }

        time = time + 1;

        let _ = match counter {
            0 => {
                window
                    .render(Some(&mut scene), None, Some(&mut camera), None, None, None)
                    .await
            }
            1 => {
                window
                    .render(
                        Some(&mut scene),
                        None,
                        Some(&mut camera),
                        None,
                        None,
                        Some(&mut grays),
                    )
                    .await
            }
            2 => {
                window
                    .render(
                        Some(&mut scene),
                        None,
                        Some(&mut camera),
                        None,
                        None,
                        Some(&mut waves),
                    )
                    .await
            }
            #[cfg(not(target_arch = "wasm32"))]
            3 => {
                window
                    .render(
                        Some(&mut scene),
                        None,
                        Some(&mut camera),
                        None,
                        None,
                        Some(&mut sobel),
                    )
                    .await
            }
            _ => true,
        };
    }
}
