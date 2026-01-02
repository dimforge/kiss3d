use kiss3d::prelude::*;
use rand::random;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: primitives_scale").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 0.0, 15.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));
    let mut primitives = scene.add_group();

    // NOTE: scaling is not possible.
    for i in 0usize..11 {
        let dim: f32 = (0.4 + random::<f32>()) / 2.0;
        let dim2 = dim / 2.0;

        let offset = i as f32 * 1.0 - 5.0;

        primitives
            .add_cube(dim2, dim2, dim2)
            .translate(Vec3::new(offset, 1.0, 0.0))
            .set_color(Color::new(random(), random(), random(), 1.0));
        primitives
            .add_sphere(dim2)
            .translate(Vec3::new(offset, -1.0, 0.0))
            .set_color(Color::new(random(), random(), random(), 1.0));
        primitives
            .add_cone(dim2, dim)
            .translate(Vec3::new(offset, 2.0, 0.0))
            .set_color(Color::new(random(), random(), random(), 1.0));
        primitives
            .add_cylinder(dim2, dim)
            .translate(Vec3::new(offset, -2.0, 0.0))
            .set_color(Color::new(random(), random(), random(), 1.0));
        primitives
            .add_capsule(dim2, dim)
            .translate(Vec3::new(offset, 0.0, 0.0))
            .set_color(Color::new(random(), random(), random(), 1.0));
    }

    let rot = Quat::from_axis_angle(Vec3::Y, 0.014);

    while window.render_3d(&mut scene, &mut camera).await {
        primitives.rotate(rot);
    }
}
