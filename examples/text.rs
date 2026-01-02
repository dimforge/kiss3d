use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: text").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    let font = Font::default();

    while window.render_3d(&mut scene, &mut camera).await {
        window.draw_text("Hello birds!", Vec2::ZERO, 120.0, &font, CYAN);

        let ascii = " !\"#$%&'`()*+,-_./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^abcdefghijklmnopqrstuvwxyz{|}~";
        window.draw_text(ascii, Vec2::new(0.0, 120.0), 60.0, &font, YELLOW);
    }
}
