use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: custom_mesh_shared").await;
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

    // TOTO: it would be better to do: MeshManager::add(Rc....) directly.
    MeshManager3d::get_global_manager(|mm| mm.add(mesh.clone(), "custom_mesh"));

    let mut c1 = scene
        .add_geom_with_name("custom_mesh", Vec3::new(1.0, 1.0, 1.0))
        .unwrap()
        .set_color(RED)
        .enable_backface_culling(false);
    let mut c2 = scene
        .add_geom_with_name("custom_mesh", Vec3::new(1.0, 1.0, 1.0))
        .unwrap()
        .set_color(GREEN)
        .enable_backface_culling(false);

    let rot1 = Quat::from_axis_angle(Vec3::Y, 0.014);
    let rot2 = Quat::from_axis_angle(Vec3::Y, -0.014);

    while window.render_3d(&mut scene, &mut camera).await {
        c1.rotate(rot1);
        c2.rotate(rot2);
    }
}
