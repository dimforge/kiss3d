use kiss3d::prelude::*;
use std::path::Path;

// Based on cube example.
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: screenshot").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));
    scene
        .add_cube(0.2, 0.2, 0.2)
        .set_color(RED)
        .rotate(Quat::from_axis_angle(Vec3::Y, 0.785))
        .rotate(Quat::from_axis_angle(Vec3::X, -0.6f32));

    window.render_3d(&mut scene, &mut camera).await; // Render one frame.
    let img = window.snap_image();
    let img_path = Path::new("screenshot.png");
    img.save(img_path).unwrap();
    println!("Screenshot saved to `screenshot.png`");
}
