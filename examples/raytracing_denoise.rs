//! Headless low-sample path tracing with the edge-aware denoiser on vs off.
//!
//! Renders the same Cornell-box-style scene at a deliberately low sample count
//! (so the raw image is visibly noisy) twice: once with the à-trous denoiser
//! disabled (`raytraced_noisy.png`) and once with it enabled
//! (`raytraced_denoised.png`). Comparing the two shows the denoiser cleaning up
//! Monte-Carlo noise while keeping edges and texture detail sharp.

use kiss3d::prelude::*;
use kiss3d::renderer::RayTracer;

/// Builds the demo scene (a small Cornell box with two spheres and a ceiling
/// light) into `scene`.
fn build_scene(scene: &mut SceneNode3d) {
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
        .add_light(Light::point(30.0).with_intensity(1.0))
        .set_position(Vec3::new(0.0, 3.5, 1.0));
}

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: ray tracing denoise").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 1.0, 6.0), Vec3::new(0.0, 1.0, 0.0));
    let mut scene = SceneNode3d::empty();
    build_scene(&mut scene);

    let mut raytracer = RayTracer::new();
    raytracer.set_max_bounces(8);
    raytracer.set_denoise(true);
    raytracer.set_denoise_iterations(5);

    while window
        .raytrace_3d(&mut scene, &mut camera, &mut raytracer)
        .await
    {}
}
