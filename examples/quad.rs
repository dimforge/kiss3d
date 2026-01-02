use kiss3d::prelude::*;
use rand::random;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: quad").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));

    let mut c = scene
        .add_quad(5.0, 4.0, 100, 100)
        .set_color(Color::new(random(), random(), random(), 1.0));

    let mut time = 0.016f32;

    while window.render_3d(&mut scene, &mut camera).await {
        c.modify_vertices(&mut |coords| {
            for v in coords.iter_mut() {
                v.z = time.sin()
                    * (((v.x + time) * 4.0).cos() + time.sin() * ((v.y + time) * 4.0 + time).cos())
                    / 2.0
            }
        });
        c.recompute_normals();

        time = time + 0.016;
    }
}
