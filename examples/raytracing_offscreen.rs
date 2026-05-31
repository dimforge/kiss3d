//! Headless progressive path tracing: renders a scene to `raytraced.png` with no
//! window, accumulating many samples for a converged image.

use kiss3d::prelude::*;
use kiss3d::renderer::RayTracer;
use std::path::Path;

#[kiss3d::main]
async fn main() {
    env_logger::init();

    let mut surface = OffscreenSurface::new(800, 600).await;
    surface.set_background_color(BLACK);

    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 1.0, 6.0), Vec3::new(0.0, 1.0, 0.0));
    let mut scene = SceneNode3d::empty();

    scene
        .add_cube(6.0, 0.1, 6.0)
        .set_position(Vec3::new(0.0, -0.05, 0.0))
        .set_color(WHITE)
        .set_roughness(0.9);
    scene
        .add_cube(6.0, 0.1, 6.0)
        .set_position(Vec3::new(0.0, 4.0, 0.0))
        .set_color(WHITE)
        .set_roughness(0.9);
    scene
        .add_cube(6.0, 4.0, 0.1)
        .set_position(Vec3::new(0.0, 2.0, -3.0))
        .set_color(WHITE)
        .set_roughness(0.9);
    scene
        .add_cube(0.1, 4.0, 6.0)
        .set_position(Vec3::new(-3.0, 2.0, 0.0))
        .set_color(Color::new(0.8, 0.1, 0.1, 1.0))
        .set_roughness(0.9);
    scene
        .add_cube(0.1, 4.0, 6.0)
        .set_position(Vec3::new(3.0, 2.0, 0.0))
        .set_color(Color::new(0.1, 0.7, 0.1, 1.0))
        .set_roughness(0.9);

    scene
        .add_sphere(0.8)
        .set_position(Vec3::new(-1.1, 0.8, 0.0))
        .set_color(WHITE)
        .set_roughness(0.4);
    scene
        .add_sphere(0.8)
        .set_position(Vec3::new(1.1, 0.8, 0.6))
        .set_color(Color::new(1.0, 0.85, 0.4, 1.0))
        .set_metallic(1.0)
        .set_roughness(0.08);

    scene
        .add_cube(1.2, 0.1, 1.2)
        .set_position(Vec3::new(0.0, 3.9, 0.0))
        .set_color(WHITE)
        .set_emissive(Color::new(6.0, 6.0, 6.0, 1.0));
    scene
        .add_light(Light::point(30.0).with_intensity(8.0))
        .set_position(Vec3::new(0.0, 3.5, 1.0));

    let mut raytracer = RayTracer::new();
    raytracer.set_max_bounces(8);

    let samples = 256;
    let img = surface
        .render_image_raytraced(&mut scene, &mut camera, &mut raytracer, samples)
        .await;
    img.save(Path::new("raytraced.png")).unwrap();
    println!(
        "Rendered {} samples to `raytraced.png` ({:?})",
        raytracer.samples_accumulated(),
        surface.size()
    );
}
