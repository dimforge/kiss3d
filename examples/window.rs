use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: window").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();

    window.set_background_color(LIGHT_BLUE);

    while window.render_3d(&mut scene, &mut camera).await {}
}
