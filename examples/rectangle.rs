use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: rectangle").await;
    let mut camera = PanZoomCamera2d::new(Vec2::ZERO, 5.0);
    let mut scene = SceneNode2d::empty();
    let mut c = scene.add_rectangle(100.0, 150.0).set_color(RED);

    let rot = 0.014;

    while window.render_2d(&mut scene, &mut camera).await {
        c.rotate(rot);
    }
}
