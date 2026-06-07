//! Planar reflectors (mirrors).
//!
//! Several reflective surfaces with different orientations coexist in one scene:
//! a horizontal floor mirror, a vertical wall mirror, and an angled mirror. Each
//! is a [`SceneNode3d::add_reflector`] quad placed/rotated with the usual node
//! transforms; the window renders the scene from a mirror camera into each one's
//! own texture every frame. A row of colored, orbiting shapes floats above the
//! floor so the reflections are easy to read.
//!
//! Run: `cargo run --example mirror`.

use kiss3d::prelude::*;
use std::f32::consts::FRAC_PI_2;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: mirrors").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 3.5, 11.0), Vec3::new(0.0, 1.2, 0.0));
    let mut scene = SceneNode3d::empty();

    window.set_ambient(0.2);
    scene.add_light(Light::directional(Vec3::new(-0.4, -0.9, -0.3)).with_intensity(2.5));
    #[cfg(not(target_arch = "wasm32"))]
    window.set_skybox_from_file(Path::new("./examples/media/skybox.png"));
    #[cfg(target_arch = "wasm32")]
    window.set_skybox_from_memory(include_bytes!("media/skybox.png"));

    // Horizontal floor mirror at y = 0: rotate the local-XY quad so its +Z normal
    // points up (+Y). The reflection is integrated into the PBR material, so the
    // floor keeps a dark tinted base color + glossy roughness with the reflection
    // blended on top (a partial, glossy mirror rather than a perfect one).
    scene
        .add_reflector(16.0, 16.0)
        .set_rotation(Quat::from_axis_angle(Vec3::X, -FRAC_PI_2))
        .set_color(Color::new(0.06, 0.06, 0.08, 1.0))
        .set_metallic(1.0)
        .set_roughness(0.12)
        .set_reflector_intensity(0.9);

    // Vertical wall mirror across the back, facing +Z (no rotation needed: the quad
    // already lies in XY with its normal along +Z). A near-white smooth surface →
    // close to a perfect mirror.
    scene
        .add_reflector(10.0, 5.0)
        .set_position(Vec3::new(0.0, 2.5, -5.0))
        .set_color(Color::new(0.85, 0.88, 0.95, 1.0))
        .set_metallic(1.0)
        .set_roughness(0.05);

    // A reflective sphere on the right: a planar reflector with a strong
    // normal-falloff.
    scene
        .add_cylinder(1.5, 4.0)
        .set_position(Vec3::new(5.5, 2.5, 0.0))
        .set_metallic(0.7)
        .set_roughness(0.08)
        .set_reflector(Some(Reflector::new()))
        .set_reflector_intensity(0.9)
        .set_reflector_normal_falloff(1.0)
        .set_reflector_normal(Vec3::new(-1.0, 0.0, 0.0));

    // A ring of colored shapes floating above the floor (animated below).
    let palette = [
        Color::new(0.95, 0.4, 0.4, 1.0),
        Color::new(0.4, 0.9, 0.5, 1.0),
        Color::new(0.4, 0.6, 0.95, 1.0),
        Color::new(0.95, 0.85, 0.4, 1.0),
        Color::new(0.85, 0.5, 0.9, 1.0),
    ];
    let mut shapes = Vec::new();
    for (i, color) in palette.iter().enumerate() {
        let mut s = if i % 2 == 0 {
            scene.add_sphere(0.7)
        } else {
            scene.add_cube(1.1, 1.1, 1.1)
        };
        s.set_color(*color).set_metallic(0.2).set_roughness(0.4);
        let base_angle = i as f32 / palette.len() as f32 * std::f32::consts::TAU;
        shapes.push((s, base_angle));
    }

    let mut t = 0.0f32;
    while window.render_3d(&mut scene, &mut camera).await {
        t += 0.01;
        for (i, (shape, base_angle)) in shapes.iter_mut().enumerate() {
            let angle = *base_angle + t;
            let x = 2.6 * angle.cos();
            let z = 2.6 * angle.sin();
            let y = 1.6 + 0.5 * (t * 1.7 + i as f32).sin();
            shape.set_position(Vec3::new(x, y, z));
            shape.set_rotation(Quat::from_axis_angle(Vec3::Y, angle * 1.5));
        }
    }
}
