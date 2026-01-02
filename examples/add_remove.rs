use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: add_remove").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 0.0, -2.0));

    let mut cube = scene.add_cube(1.0, 1.0, 1.0);
    let mut added = true;

    while window.render_3d(&mut scene, &mut camera).await {
        if added {
            cube.remove();
        } else {
            cube = scene.add_cube(1.0, 1.0, 1.0).set_color(RED);
        }

        added = !added;
    }
}
