//! Order-independent transparency in the rasterizer.
//!
//! kiss3d's raster pipeline resolves translucency with Weighted-Blended OIT
//! (McGuire & Bavoil 2013): transparent surfaces are accumulated into separate
//! weighted color / revealage targets and composited in one pass, so the result
//! needs **no depth sorting** and stays correct for interpenetrating and
//! overlapping translucency. A surface is treated as transparent when its color's
//! alpha is `< 1`.
//!
//! Orbit the camera (drag) and notice the three interpenetrating colored sheets
//! and the overlapping translucent spheres blend consistently from every angle —
//! there is no popping or order-dependent darkening as the view changes.

use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: order-independent transparency").await;
    window.set_background_color(Color::new(0.1, 0.11, 0.14, 1.0));
    window.set_ambient(0.5);

    let mut camera =
        OrbitCamera3d::new_with_frustum(0.9, 0.1, 100.0, Vec3::new(3.2, 2.2, 4.5), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();

    // Opaque ground and a solid sphere the translucent geometry intersects.
    scene
        .add_cube(12.0, 0.3, 12.0)
        .set_position(Vec3::new(0.0, -1.6, 0.0))
        .set_color(Color::new(0.6, 0.6, 0.65, 1.0));
    scene
        .add_sphere(0.9)
        .set_position(Vec3::ZERO)
        .set_color(Color::new(0.85, 0.85, 0.9, 1.0));

    // Three interpenetrating translucent sheets (alpha < 1). With order-dependent
    // alpha blending their overlaps would depend on draw order and break where
    // they cross; with OIT they blend consistently.
    let sheet = |angle: f32, color: Color| -> (f32, Color) { (angle, color) };
    for (angle, color) in [
        sheet(0.0, Color::new(0.95, 0.25, 0.25, 0.45)),
        sheet(1.05, Color::new(0.25, 0.9, 0.35, 0.45)),
        sheet(-1.05, Color::new(0.3, 0.45, 0.95, 0.45)),
    ] {
        let mut s = scene.add_cube(2.6, 2.6, 0.06);
        s.set_color(color);
        s.rotate(Quat::from_axis_angle(Vec3::Y, angle));
    }

    // A cluster of overlapping translucent spheres floating above.
    for (i, color) in [
        Color::new(0.95, 0.7, 0.2, 0.5),
        Color::new(0.2, 0.8, 0.9, 0.5),
        Color::new(0.9, 0.3, 0.8, 0.5),
    ]
    .iter()
    .copied()
    .enumerate()
    {
        let a = i as f32 * std::f32::consts::TAU / 3.0;
        scene
            .add_sphere(0.55)
            .set_position(Vec3::new(a.cos() * 0.5, 1.8, a.sin() * 0.5))
            .set_color(color);
    }

    scene
        .add_light(
            Light::directional(Vec3::new(-0.5, -0.8, -0.4))
                .with_color(Color::new(1.0, 0.97, 0.9, 1.0))
                .with_intensity(2.2),
        )
        .set_position(Vec3::new(4.0, 6.0, 3.0));
    scene
        .add_light(
            Light::point(40.0)
                .with_color(Color::new(0.4, 0.45, 0.6, 1.0))
                .with_intensity(1.5),
        )
        .set_position(Vec3::new(-4.0, 3.0, -4.0));

    while window.render_3d(&mut scene, &mut camera).await {}
}
