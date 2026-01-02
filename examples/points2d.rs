use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: points 2D").await;
    let mut camera = PanZoomCamera2d::new(Vec2::new(0.0, -200.0), 1.0);
    let mut scene = SceneNode2d::empty();

    while window.render_2d(&mut scene, &mut camera).await {
        let a = Vec2::new(-50.0, -200.0);
        let b = Vec2::new(0.0, -200.0);
        let c = Vec2::new(50.0, -200.0);

        window.draw_point_2d(a, RED, 5.0);
        window.draw_point_2d(b, GREEN, 15.0);
        window.draw_point_2d(c, BLUE, 25.0);
    }
}
