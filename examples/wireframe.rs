use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: wireframe").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));

    let mut c = scene
        .add_cube(1.0, 1.0, 1.0)
        .set_color(RED)
        .set_points_color(Some(CYAN))
        .set_points_size(30.0, false) // false = screen pixels
        .set_lines_width(10.0, false) // false = screen pixels
        .set_surface_rendering_activation(false);

    let rot = Quat::from_axis_angle(Vec3::Y, 0.014);

    while window.render_3d(&mut scene, &mut camera).await {
        c.rotate(rot);
    }
}
