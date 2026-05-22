use kiss3d::prelude::*;
use std::path::Path;

// Renders a scene to an image with no window at all.
#[kiss3d::main]
async fn main() {
    let mut surface = OffscreenSurface::new(1024, 768).await;
    surface.set_background_color(DARK_BLUE);

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

    let img = surface.render_image_3d(&mut scene, &mut camera).await;
    img.save(Path::new("offscreen.png")).unwrap();
    println!("Rendered to `offscreen.png` ({:?})", surface.size());
}
