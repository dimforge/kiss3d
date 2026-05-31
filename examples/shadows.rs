//! Real-time shadow mapping in the rasterization pipeline.
//!
//! A directional "sun" and a spot light cast shadows onto a ground plane from a
//! few rotating objects. Shadows are on by default; press `S` to toggle them.

use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    env_logger::init();
    let mut window = Window::new("Kiss3d: shadows").await;
    window.set_background_color(Color::new(0.05, 0.06, 0.09, 1.0));
    window.set_ambient(0.15);

    // Shadow mapping is enabled by default; this is just to be explicit. A higher
    // atlas resolution yields crisper shadows.
    window.set_shadows_enabled(true);
    window.set_shadow_resolution(2048);

    let mut camera =
        OrbitCamera3d::new_with_frustum(0.9, 0.1, 100.0, Vec3::new(6.0, 6.0, 8.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();

    // Ground plane (shadow receiver).
    scene
        .add_cube(14.0, 0.4, 14.0)
        .set_position(Vec3::new(0.0, -1.2, 0.0))
        .set_color(Color::new(0.7, 0.7, 0.7, 1.0));

    // A few shadow casters.
    let mut cube = scene
        .add_cube(1.5, 1.5, 1.5)
        .set_position(Vec3::new(-2.0, 0.0, 0.0))
        .set_color(Color::new(0.8, 0.3, 0.3, 1.0));

    let mut sphere = scene
        .add_sphere(1.0)
        .set_position(Vec3::new(2.0, 0.2, 1.0))
        .set_color(Color::new(0.3, 0.5, 0.8, 1.0));

    scene
        .add_cone(0.8, 1.8)
        .set_position(Vec3::new(0.5, 0.0, -2.5))
        .set_color(Color::new(0.3, 0.8, 0.4, 1.0));

    // Directional "sun" casting shadows from above (shadow casting on by default).
    scene
        .add_light(
            Light::directional(Vec3::new(-0.6, -1.0, -0.3))
                .with_color(Color::new(1.0, 0.96, 0.85, 1.0))
                .with_intensity(2.5),
        )
        .set_position(Vec3::new(0.0, 6.0, 0.0));

    // Spot light pointing down at the scene, casting a sharper shadow. `reorient`
    // aims the node's forward (-Z) vector from the light position toward the floor.
    let mut spot = scene.add_light(
        Light::spot(0.35, 0.55, 30.0)
            .with_color(Color::new(0.8, 0.85, 1.0, 1.0))
            .with_intensity(12.0),
    );
    spot.reorient(Vec3::new(4.0, 6.0, 4.0), Vec3::ZERO, Vec3::Y);

    // A non-shadow-casting fill light to lift the ambient look.
    scene
        .add_light(
            Light::point(40.0)
                .with_color(Color::new(0.3, 0.3, 0.4, 1.0))
                .with_intensity(1.5)
                .with_casts_shadows(false),
        )
        .set_position(Vec3::new(-5.0, 3.0, -5.0));

    let cube_rot = Quat::from_axis_angle(Vec3::Y, 0.01);
    let mut t = 0.0f32;

    while window.render_3d(&mut scene, &mut camera).await {
        t += 0.02;
        cube.rotate(cube_rot);
        sphere.set_position(Vec3::new(2.0, 0.2 + t.sin().abs() * 1.5, 1.0));

        // Toggle shadows with the `S` key.
        let mut toggle_shadows = false;
        for event in window.events().iter() {
            if let WindowEvent::Key(Key::S, Action::Press, _) = event.value {
                toggle_shadows = true;
            }
        }
        if toggle_shadows {
            let enabled = window.shadows_enabled();
            window.set_shadows_enabled(!enabled);
        }
    }
}
