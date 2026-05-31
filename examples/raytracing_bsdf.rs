//! Headless showcase of the unified path-tracer BSDF: glass and metal spheres on
//! a floor, lit by an optional HDRI environment, an emissive area light, and a
//! soft (sphere) point light, with a thin-lens depth-of-field camera.
//!
//! Renders to `raytraced_bsdf.png`. Pass the path to an equirectangular `.hdr`
//! as the first CLI argument to enable image-based lighting; otherwise the
//! built-in procedural sky is used.

use kiss3d::prelude::*;
use kiss3d::renderer::RayTracer;
use std::path::Path;
use std::time::Instant;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: ray tracing BSDF").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 1.2, 6.0), Vec3::new(0.0, 0.8, 0.0));
    let mut scene = SceneNode3d::empty();

    // Floor.
    scene
        .add_cube(8.0, 0.1, 8.0)
        .set_position(Vec3::new(0.0, -0.05, 0.0))
        .set_color(Color::new(0.7, 0.7, 0.75, 1.0))
        .set_roughness(0.6);

    // Clear glass sphere (dielectric, smooth, fully transmissive).
    scene
        .add_sphere(0.8)
        .set_position(Vec3::new(-1.5, 0.8, 0.0))
        .set_color(WHITE)
        .set_bsdf(Bsdf::Glass)
        .set_ior(1.5)
        .set_transmission(1.0)
        .set_roughness(0.0);

    // Polished gold metal sphere (conductor with a warm specular tint).
    scene
        .add_sphere(0.8)
        .set_position(Vec3::new(0.0, 0.8, -1.0))
        .set_color(Color::new(1.0, 0.85, 0.4, 1.0))
        .set_bsdf(Bsdf::Metal)
        .set_metallic(1.0)
        .set_specular_tint(Color::new(1.0, 0.82, 0.45, 1.0))
        .set_roughness(0.05);

    // Translucent (subsurface) sphere.
    scene
        .add_sphere(0.8)
        .set_position(Vec3::new(1.5, 0.8, 0.0))
        .set_color(Color::new(0.9, 0.4, 0.4, 1.0))
        .set_subsurface(0.8, 0.5)
        .set_roughness(0.5);

    // Emissive area light (ceiling panel).
    scene
        .add_cube(1.5, 0.1, 1.5)
        .set_position(Vec3::new(0.0, 4.0, 0.0))
        .set_color(WHITE)
        .set_emissive(Color::new(5.0, 5.0, 5.0, 1.0));

    // Soft (sphere) point light for penumbrae.
    scene
        .add_light(Light::point(40.0).with_intensity(10.0).with_radius(0.4))
        .set_position(Vec3::new(2.5, 3.0, 2.0));

    let mut raytracer = RayTracer::new();
    raytracer.set_max_bounces(12);

    // Optional HDRI environment from the command line.
    if let Some(hdri) = std::env::args().nth(1) {
        if raytracer.set_environment_from_file(Path::new(&hdri)) {
            raytracer.set_environment_orientation(0.0, 1.0);
            println!("Using environment map: {hdri}");
        } else {
            eprintln!("Could not load environment `{hdri}`; using procedural sky.");
        }
    }

    // Thin-lens depth of field focused on the front spheres.
    raytracer.set_aperture(0.06, 6.0);

    while window
        .render_raytraced(&mut scene, &mut camera, &mut raytracer)
        .await
    {}
}
