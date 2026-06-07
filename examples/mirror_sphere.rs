//! A chrome mirror sphere in the middle of an environment, reflecting the scene
//! around it.
//!
//! The reflections use a **runtime-captured reflection probe**: each frame the
//! probe is re-centered on the sphere and the surrounding scene is rendered into
//! a parallax-corrected environment map, which the sphere's PBR material then
//! samples as a mirror reflection. This is the classic dynamic cube-camera +
//! environment-map reflection trick.
//!
//! Two render layers keep the sphere from reflecting itself:
//! - layer 0: the environment (floor, columns, orbiting cubes) — captured by the
//!   probe, so it shows up in the reflection.
//! - layer 1: the mirror sphere — *excluded* from the capture, so the sphere is
//!   hidden while the probe renders and never reflects itself.
//!
//! The orbiting cubes move every frame (and the probe is re-captured), so the
//! reflection on the sphere is live, mirroring the moving geometry around it.
//!
//! Run with: `cargo run --release --example mirror_sphere`.

use std::path::Path;

#[kiss3d::main]
async fn main() {
    use kiss3d::prelude::*;

    let mut window = Window::new("Kiss3d: mirror sphere").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 3.0, 11.0), Vec3::new(0.0, 1.0, 0.0));
    let mut scene = SceneNode3d::empty();

    // Skybox IBL + a couple of tinted directional lights give the metal something
    // rich to reflect even before the local probe kicks in.
    window.set_ambient(0.12);
    window.set_skybox_from_file(Path::new("./examples/media/skybox.png"));
    scene.add_light(Light::directional(Vec3::new(-0.5, -0.8, -0.4)).with_intensity(2.2));
    scene.add_light(
        Light::directional(Vec3::new(0.6, -0.4, 0.5))
            .with_color(Color::new(0.5, 0.65, 1.0, 1.0))
            .with_intensity(1.2),
    );

    // Reflective floor (captured on layer 0).
    let mut floor = scene.add_cube(20.0, 0.2, 20.0);
    floor.set_position(Vec3::new(0.0, -1.0, 0.0));
    floor.set_color(Color::new(0.45, 0.46, 0.5, 1.0));
    floor.set_roughness(0.22);

    // A ring of colored columns surrounding the sphere — static environment that
    // wraps around the mirror so you can read the reflection as you orbit.
    let column_colors = [
        Color::new(0.92, 0.30, 0.32, 1.0),
        Color::new(0.95, 0.62, 0.25, 1.0),
        Color::new(0.92, 0.86, 0.30, 1.0),
        Color::new(0.35, 0.80, 0.45, 1.0),
        Color::new(0.30, 0.70, 0.90, 1.0),
        Color::new(0.45, 0.45, 0.95, 1.0),
        Color::new(0.70, 0.40, 0.90, 1.0),
        Color::new(0.95, 0.45, 0.70, 1.0),
    ];
    let column_count = column_colors.len();
    for (i, color) in column_colors.iter().enumerate() {
        let angle = i as f32 / column_count as f32 * std::f32::consts::TAU;
        let radius = 6.5;
        let height = 4.0;
        let mut col = scene.add_cylinder(0.45, height);
        col.set_position(Vec3::new(
            radius * angle.cos(),
            height * 0.5 - 1.0,
            radius * angle.sin(),
        ));
        col.set_color(*color);
        col.set_metallic(0.1);
        col.set_roughness(0.6);
    }

    // Orbiting cubes — the moving content in the reflection. Stored with a
    // per-cube phase so they ripple around the sphere rather than move in lockstep.
    let cube_colors = [
        Color::new(0.95, 0.55, 0.45, 1.0),
        Color::new(0.55, 0.90, 0.70, 1.0),
        Color::new(0.55, 0.72, 0.98, 1.0),
        Color::new(0.95, 0.85, 0.50, 1.0),
    ];
    let mut cubes = Vec::new();
    for (i, color) in cube_colors.iter().enumerate() {
        let mut c = scene.add_cube(0.8, 0.8, 0.8);
        c.set_color(*color);
        c.set_metallic(0.2);
        c.set_roughness(0.35);
        let phase = i as f32 / cube_colors.len() as f32 * std::f32::consts::TAU;
        cubes.push((c, phase));
    }

    // The chrome mirror sphere. Metallic + near-zero roughness makes the env-map
    // reflection dominate, so it reads as polished chrome. On render layer 1 so
    // the probe capture excludes it (it must not reflect itself).
    let mut sphere = scene.add_sphere(1.6);
    sphere.set_color(Color::new(1.0, 1.0, 1.0, 1.0));
    sphere.set_metallic(1.0);
    sphere.set_roughness(0.1);
    sphere.set_render_layers(0b10);

    // The runtime probe captures only layer 0 (everything except the sphere).
    window.set_reflection_capture_layers(0b01);
    let probe = window
        .add_reflection_probe(ReflectionProbe {
            center: Vec3::new(0.0, 1.0, 0.0),
            // Parallax box bounding the room, so reflected geometry lands at the
            // right place on the curved surface instead of looking infinitely far.
            half_extents: Vec3::new(8.5, 5.0, 8.5),
            falloff: 1.5,
            intensity: 1.0,
            rotation: 0.0,
        })
        .expect("probe slot 0");

    let mut t = 0.0f32;
    while window.render_3d(&mut scene, &mut camera).await {
        t += 0.015;

        // Bob the sphere up and down with a gentle horizontal sway — it stays in
        // the middle but clearly moves.
        let sphere_pos = Vec3::new(
            0.6 * (t * 0.7).cos(),
            1.0 + 0.5 * (t * 1.3).sin(),
            0.6 * (t * 0.7).sin(),
        );
        sphere.set_position(sphere_pos);

        // Orbit the cubes around the center so the reflection is visibly live.
        for (cube, phase) in cubes.iter_mut() {
            let angle = t * 0.9 + *phase;
            let r = 3.6;
            cube.set_position(Vec3::new(
                r * angle.cos(),
                1.0 + 0.6 * (t * 1.5 + *phase).sin(),
                r * angle.sin(),
            ));
            cube.append_rotation(Quat::from_axis_angle(Vec3::Y, 0.03));
        }

        // Re-center the probe on the sphere and re-capture the (moving) scene each
        // frame — the equivalent of `mirrorSphereCamera.update(renderer, scene)`.
        if let Some(p) = window.reflection_probe_mut(probe) {
            p.center = sphere_pos;
        }
        window.capture_reflection_probe(probe);
    }
}
