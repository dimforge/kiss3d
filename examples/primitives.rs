use kiss3d::prelude::*;
use rand::random;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: primitives").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));

    let mut c = scene
        .add_cube(1.0, 1.0, 1.0)
        .set_color(Color::new(random(), random(), random(), 1.0))
        .set_position(Vec3::new(2.0, 0.0, 0.0));
    let mut s = scene
        .add_sphere(0.5)
        .set_color(Color::new(random(), random(), random(), 1.0))
        .set_position(Vec3::new(4.0, 0.0, 0.0));
    let mut p = scene
        .add_cone(0.5, 1.0)
        .set_color(Color::new(random(), random(), random(), 1.0))
        .set_position(Vec3::new(-2.0, 0.0, 0.0));
    let mut y = scene
        .add_cylinder(0.5, 1.0)
        .set_color(Color::new(random(), random(), random(), 1.0))
        .set_position(Vec3::new(-4.0, 0.0, 0.0));
    let mut a = scene
        .add_capsule(0.5, 1.0)
        .set_color(Color::new(random(), random(), random(), 1.0));

    let rot = Quat::from_axis_angle(Vec3::Y, 0.014);

    while window.render_3d(&mut scene, &mut camera).await {
        c.rotate(rot);
        s.rotate(rot);
        p.rotate(rot);
        y.rotate(rot);
        a.rotate(rot);
    }
}
