//! Polyline strip example
//!
//! Demonstrates drawing a polyline around a cube with configurable width,
//! similar to bevy_polyline's linestrip example.

use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: polyline strip").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));

    window.set_background_color(Color::new(0.2, 0.2, 0.25, 1.0));

    // Add a cube to show the polyline around
    let mut cube = scene
        .add_cube(1.0, 1.0, 1.0)
        .set_color(Color::new(0.5, 0.55, 1.0, 1.0));

    // Create base polyline once (vertices in local space)
    let mut polyline = Polyline3d::new(vec![
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(-0.5, -0.5, 0.5),
    ])
    .with_color(RED)
    .with_width(5.0)
    .with_depth_bias(0.0002); // Slight depth bias to render in front of the cube

    let mut angle = 0.0f32;

    while window.render_3d(&mut scene, &mut camera).await {
        // Rotate the cube
        angle += 0.01;
        let rotation = Quat::from_axis_angle(Vec3::Y, angle);
        cube.set_rotation(rotation);

        // Apply same rotation to polyline via transform
        polyline.transform.rotation = rotation;

        window.draw_polyline(&polyline);
    }
}
