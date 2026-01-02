use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: custom_mesh").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));

    let a = Vec3::new(-1.0, -1.0, 0.0);
    let b = Vec3::new(1.0, -1.0, 0.0);
    let c = Vec3::new(0.0, 1.0, 0.0);

    let vertices = vec![a, b, c];
    let indices = vec![[0, 1, 2]];

    let mesh = Rc::new(RefCell::new(GpuMesh3d::new(
        vertices, indices, None, None, false,
    )));
    let mut c = scene
        .add_mesh(mesh, Vec3::new(1.0, 1.0, 1.0))
        .set_color(RED)
        .enable_backface_culling(false);

    let rot = Quat::from_axis_angle(Vec3::Y, 0.014);

    while window.render_3d(&mut scene, &mut camera).await {
        c.rotate(rot);
    }
}
