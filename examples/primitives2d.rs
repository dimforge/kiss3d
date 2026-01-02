use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: rectangle").await;
    let mut camera = PanZoomCamera2d::new(Vec2::ZERO, 2.0);
    let mut scene = SceneNode2d::empty();

    let mut rect = scene
        .add_rectangle(50.0, 150.0)
        .set_color(GREEN)
        .set_lines_width(10.0, false)
        .set_lines_color(Some(WHITE));
    let mut circ = scene
        .add_circle(50.0)
        .translate(Vec2::new(200.0, 0.0))
        .set_color(BLUE)
        .set_lines_width(5.0, false)
        .set_lines_color(Some(MAGENTA));

    let rot_rect = 0.014;
    let rot_circ = -0.014;

    while window.render_2d(&mut scene, &mut camera).await {
        rect.append_rotation(rot_rect);
        circ.append_rotation(rot_circ);
    }
}
