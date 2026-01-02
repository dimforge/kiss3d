use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: 2D lines").await;
    let mut camera = PanZoomCamera2d::default();
    let mut scene = SceneNode2d::empty();

    while window.render_2d(&mut scene, &mut camera).await {
        let a = Vec2::new(-200.0, -200.0);
        let b = Vec2::new(0.0, 200.0);
        let c = Vec2::new(200.0, -200.0);

        window.draw_line_2d(a, b, RED, 2.0);
        window.draw_line_2d(b, c, GREEN, 2.0);
        window.draw_line_2d(c, a, BLUE, 2.0);
    }
}
