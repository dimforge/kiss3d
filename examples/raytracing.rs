//! Progressive GPU path tracing.
//!
//! Builds a small Cornell-box-like scene and renders it with the path tracer
//! instead of the rasterizer. Samples accumulate over time for a noise-free,
//! physically-based image; the accumulation restarts automatically whenever the
//! camera is moved with the mouse.
//!
//! The portable compute backend runs everywhere (including macOS/Metal and the
//! web). On a capable Vulkan GPU the hardware ray-query backend is selected
//! automatically; otherwise it falls back to the compute backend.

use kiss3d::prelude::*;
use kiss3d::renderer::{RayBackend, RayTracer};

#[kiss3d::main]
async fn main() {
    env_logger::init();

    let mut window = Window::new("Kiss3d: ray tracing").await;
    // Keep the ambient/sky fill low so shadows stay visible; a high ambient floods
    // the open box with uniform light and washes shadows out.
    window.set_ambient(0.08);

    // Raised and angled down so the floor — where the shadows land — is in view.
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 2.6, 5.5), Vec3::new(0.0, 0.8, 0.0));
    let mut scene = SceneNode3d::empty();

    // Room: floor, ceiling, back and two colored side walls.
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

    // A rough dielectric cylinder and a polished metal sphere.
    scene
        .add_cylinder(0.8, 1.6)
        .set_position(Vec3::new(-1.1, 0.8, 0.0))
        .set_color(WHITE)
        .set_roughness(0.4);
    scene
        .add_sphere(0.8)
        .set_position(Vec3::new(1.1, 0.8, 0.6))
        .set_color(Color::new(1.0, 0.85, 0.4, 1.0))
        .set_metallic(1.0)
        .set_roughness(0.08);

    // A glowing emissive box acts as an area light.
    scene
        .add_cube(1.2, 0.1, 1.2)
        .set_position(Vec3::new(0.0, 3.9, 0.0))
        .set_color(WHITE)
        .set_emissive(Color::new(4.0, 4.0, 4.0, 1.0));

    // A point light, off to one side and toward the back, so the spheres cast
    // shadows the camera can actually see (a centered overhead light would hide
    // them straight underneath). Kept below the ceiling so it isn't occluded.
    scene
        .add_light(Light::point(40.0).with_intensity(1.0))
        .set_position(Vec3::new(-1.8, 3.6, -1.5));

    let mut raytracer = RayTracer::new();
    raytracer.set_max_bounces(8);

    let font = Font::default();
    match raytracer.backend() {
        RayBackend::Software => println!("Path tracer backend: compute (BVH)"),
        RayBackend::Hardware => println!("Path tracer backend: hardware ray queries"),
    }

    while window
        .raytrace_3d(&mut scene, &mut camera, &mut raytracer)
        .await
    {
        window.draw_text(
            &format!("Samples: {}", raytracer.samples_accumulated()),
            Vec2::new(10.0, 10.0),
            40.0,
            &font,
            WHITE,
        );
    }
}
