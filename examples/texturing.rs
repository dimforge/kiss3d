use kiss3d::prelude::*;
use std::path::Path;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: texturing").await;
    let mut camera_3d = OrbitCamera3d::default();
    let mut scene_3d = SceneNode3d::empty();
    let mut camera_2d = PanZoomCamera2d::default();
    let mut scene_2d = SceneNode2d::empty();

    scene_3d
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 2.0, -10.0));

    let mut c = scene_3d
        .add_cube(1.0, 1.0, 1.0)
        .set_color(RED)
        .set_texture_from_file(Path::new("./examples/media/kitten.png"), "kitten");

    let mut r = scene_2d
        .add_rectangle(100.0, 100.0)
        .set_position(Vec2::new(-100.0, -100.0))
        .set_color(BLUE)
        .set_texture_from_memory(include_bytes!("./media/kitten.png"), "kitten_mem");

    let rot3d = Quat::from_axis_angle(Vec3::Y, 0.014);
    let rot2d = 0.01;

    // Render 3D and 2D scenes alternately
    while window
        .render(
            Some(&mut scene_3d),
            Some(&mut scene_2d),
            Some(&mut camera_3d),
            Some(&mut camera_2d),
            None,
            None,
        )
        .await
    {
        c.append_rotation(rot3d);
        r.append_rotation(rot2d);
    }
}
