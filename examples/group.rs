use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: group").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));

    let mut g1 = scene.add_group().set_position(Vec3::new(2.0, 0.0, 0.0));
    let mut g2 = scene.add_group().set_position(Vec3::new(-2.0, 0.0, 0.0));

    g1.add_cube(1.0, 5.0, 1.0);
    g1.add_cube(5.0, 1.0, 1.0);
    g1.set_color_recursive(RED);

    g2.add_cube(1.0, 5.0, 1.0);
    g2.add_cube(1.0, 1.0, 5.0);
    g2.set_color_recursive(GREEN);

    let rot1 = Quat::from_axis_angle(Vec3::Y, 0.014);
    let rot2 = Quat::from_axis_angle(Vec3::X, 0.014);

    while window.render_3d(&mut scene, &mut camera).await {
        g1.rotate(rot1);
        g2.rotate(rot2);
    }
}
