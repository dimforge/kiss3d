//! Polyline strip example
//!
//! Demonstrates drawing a polyline around a cube with configurable width,
//! similar to bevy_polyline's linestrip example.

extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::light::Light;
use kiss3d::renderer::Polyline;
use kiss3d::window::Window;
use na::{Isometry3, Point3, UnitQuaternion, Vector3};

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: polyline strip").await;

    window.set_light(Light::StickToCamera);
    window.set_background_color(0.2, 0.2, 0.25);

    // Add a cube to show the polyline around
    let mut cube = window.add_cube(1.0, 1.0, 1.0);
    cube.set_color(0.5, 0.55, 1.0);

    // Create base polyline once (vertices in local space)
    let mut polyline = Polyline::new(vec![
        Point3::new(-0.5, -0.5, -0.5),
        Point3::new(0.5, -0.5, -0.5),
        Point3::new(0.5, 0.5, -0.5),
        Point3::new(-0.5, 0.5, -0.5),
        Point3::new(-0.5, 0.5, 0.5),
        Point3::new(0.5, 0.5, 0.5),
        Point3::new(0.5, -0.5, 0.5),
        Point3::new(-0.5, -0.5, 0.5),
    ])
    .with_color(1.0, 0.0, 0.0)
    .with_width(5.0)
    .with_depth_bias(0.0002); // Slight depth bias to render in front of the cube

    let mut angle = 0.0f32;

    while window.render().await {
        // Rotate the cube
        angle += 0.01;
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), angle);
        cube.set_local_rotation(rotation);

        // Apply same rotation to polyline via transform
        polyline.transform = Isometry3::from_parts(Default::default(), rotation);

        window.draw_polyline(&polyline);
    }
}
