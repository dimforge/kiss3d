use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: points").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));

    while window.render_3d(&mut scene, &mut camera).await {
        let a = Vec3::new(-0.1, -0.1, 0.0);
        let b = Vec3::new(0.0, 0.1, 0.0);
        let c = Vec3::new(0.1, -0.1, 0.0);

        window.draw_point(a, RED, 5.0);
        window.draw_point(b, GREEN, 15.0);
        window.draw_point(c, BLUE, 25.0);
    }
}
