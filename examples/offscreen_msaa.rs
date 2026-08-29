use kiss3d::prelude::*;
use kiss3d::window::NumSamples;
use std::path::Path;

// Renders a scene to an image with 4x MSAA and hard shadow edges, no window.
#[kiss3d::main]
async fn main() {
    let mut surface = OffscreenSurface::new(1024, 768).await;
    surface.set_background_color(DARK_BLUE);
    surface.set_samples(NumSamples::Four);
    surface.set_shadow_softness(0.0);

    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(2.0, 10.0, 4.0));
    scene
        .add_cube(4.0, 0.05, 4.0)
        .set_color(Color::new(0.3, 0.33, 0.38, 1.0))
        .set_position(Vec3::new(0.0, -0.3, 0.0));
    scene
        .add_cube(0.2, 0.2, 0.2)
        .set_color(RED)
        .rotate(Quat::from_axis_angle(Vec3::Y, 0.785))
        .rotate(Quat::from_axis_angle(Vec3::X, -0.6f32));

    let img = surface.render_image_3d(&mut scene, &mut camera).await;
    img.save(Path::new("offscreen_msaa.png")).unwrap();
    println!(
        "Rendered to `offscreen_msaa.png` ({:?}, {} samples)",
        surface.size(),
        surface.samples()
    );
}
