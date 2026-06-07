//! Transparency in the GPU path tracer, in an interactive window.
//!
//! The path tracer supports two distinct kinds of transparency:
//!
//! * **Coverage / alpha transparency** — a material whose color alpha is `< 1`.
//!   At each hit the ray passes straight through with probability `1 - alpha`,
//!   which averages over samples to order-independent alpha blending (the same
//!   notion of "see-through" as the rasterizer's OIT). The object keeps its own
//!   tint but does not bend light.
//! * **Physical glass** ([`Bsdf::Glass`]) — a dielectric that refracts light
//!   through it according to its index of refraction; the background is warped,
//!   not merely tinted.
//!
//! The scene lines both up in front of an opaque backdrop so the difference is
//! obvious: the left sphere is a clear glass dielectric (background bent), the
//! middle sphere is 35%-opacity blue coverage-alpha (background tinted but
//! straight), and an opaque red sphere sits behind them.
//!
//! Drag to orbit; samples accumulate into a clean image while the camera is
//! still and restart when it moves.

use kiss3d::prelude::*;
use kiss3d::renderer::RayTracer;

#[kiss3d::main]
async fn main() {
    env_logger::init();

    let mut window = Window::new("Kiss3d: ray tracing transparency").await;
    window.set_background_color(Color::new(0.55, 0.6, 0.7, 1.0));

    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 1.0, 6.0), Vec3::new(0.0, 0.4, 0.0));
    let mut scene = SceneNode3d::empty();

    // Opaque checker-ish floor and a back wall to refract / see through.
    scene
        .add_cube(12.0, 0.1, 12.0)
        .set_position(Vec3::new(0.0, -0.55, 0.0))
        .set_color(Color::new(0.75, 0.75, 0.8, 1.0))
        .set_roughness(0.7);
    scene
        .add_cube(8.0, 5.0, 0.2)
        .set_position(Vec3::new(0.0, 1.5, -2.2))
        .set_color(Color::new(0.85, 0.5, 0.35, 1.0))
        .set_roughness(0.8);

    // Opaque red reference sphere, partly behind the transparent ones.
    scene
        .add_sphere(0.7)
        .set_position(Vec3::new(0.6, 0.2, -0.9))
        .set_color(Color::new(0.9, 0.2, 0.2, 1.0));

    // Left: physical glass (refracts the background).
    scene
        .add_sphere(0.8)
        .set_position(Vec3::new(-1.6, 0.3, 0.4))
        .set_color(WHITE)
        .set_bsdf(Bsdf::Glass)
        .set_ior(1.5)
        .set_transmission(1.0)
        .set_roughness(0.0);

    // Middle: coverage/alpha transparency (tinted, see-through, no refraction).
    scene
        .add_sphere(0.8)
        .set_position(Vec3::new(0.4, 0.3, 0.6))
        .set_color(Color::new(0.3, 0.5, 0.95, 0.35));

    // Right: a stack of three translucent colored panels (order-independent).
    for (i, color) in [
        Color::new(0.95, 0.3, 0.3, 0.4),
        Color::new(0.3, 0.95, 0.4, 0.4),
        Color::new(0.95, 0.85, 0.3, 0.4),
    ]
    .iter()
    .copied()
    .enumerate()
    {
        scene
            .add_cube(1.1, 1.6, 0.04)
            .set_position(Vec3::new(2.1, 0.5, -0.2 + i as f32 * 0.22))
            .set_color(color);
    }

    // Emissive ceiling panel + a soft key light.
    scene
        .add_cube(2.0, 0.1, 2.0)
        .set_position(Vec3::new(0.0, 4.0, 0.5))
        .set_color(WHITE)
        .set_emissive(Color::new(4.0, 4.0, 4.0, 1.0));
    scene
        .add_light(Light::point(40.0).with_intensity(8.0).with_radius(0.3))
        .set_position(Vec3::new(2.5, 3.0, 2.5));

    let mut raytracer = RayTracer::new();
    // Coverage pass-throughs and glass refraction both consume bounces, so give
    // the paths plenty of room to reach the background through stacked surfaces.
    raytracer.set_max_bounces(16);

    while window
        .raytrace_3d(&mut scene, &mut camera, &mut raytracer)
        .await
    {}
}
