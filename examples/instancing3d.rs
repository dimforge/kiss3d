use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    env_logger::init();
    let mut window = Window::new("Kiss3d: instancing 3D").await;
    let mut camera =
        OrbitCamera3d::new(Vec3::new(200.0, 200.0, 200.0), Vec3::new(75.0, 75.0, 75.0));
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(1000.0))
        .set_position(Vec3::new(200.0, 200.0, 200.0));
    let mut c = scene.add_cube(1.0, 1.0, 1.0);
    let mut instances = vec![];

    for i in 0..100 {
        for j in 0..100 {
            for k in 0..100 {
                let ii = i as f32;
                let jj = j as f32;
                let kk = k as f32;
                let color = Color::new(ii / 100.0, jj / 100.0, kk / 100.0 + 0.1, 1.0);
                instances.push(InstanceData3d {
                    position: Vec3::new(ii, jj, kk) * 1.5,
                    color,
                    #[rustfmt::skip]
                    deformation: Mat3::from_cols_array(&[
                        1.0, ii * 0.004, kk * 0.004,
                        ii * 0.004, 1.0, jj * 0.004,
                        kk * 0.004, jj * 0.004, 1.0,
                    ]),
                    ..Default::default()
                });
            }
        }
    }

    c.set_instances(&instances);

    let rot = Quat::from_axis_angle(Vec3::Y, 0.014);

    while window.render_3d(&mut scene, &mut camera).await {
        c.rotate(rot);
    }
}
