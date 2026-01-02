//! Test of kiss3d's 2D camera. Just moves a cross around the screen whenever the mouse is clicked. Shows conversions between co-ordinate systems.
use kiss3d::prelude::*;

/// main program
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Mouse events").await;
    let mut camera = PanZoomCamera2d::default();
    let mut scene = SceneNode2d::empty();
    let draw_colour = Color::new(0.5, 1.0, 0.5, 1.0);
    let mut last_pos = Vec2::new(0.0f32, 0.0f32);
    let mut sel_pos = Vec2::new(0.0f32, 0.0f32);
    while window.render_2d(&mut scene, &mut camera).await {
        for event in window.events().iter() {
            match event.value {
                WindowEvent::FramebufferSize(x, y) => {
                    println!("frame buffer size event {}, {}", x, y);
                }
                WindowEvent::MouseButton(button, Action::Press, modif) => {
                    println!("mouse press event on {:?} with {:?}", button, modif);
                    let window_size = Vec2::new(window.size()[0] as f32, window.size()[1] as f32);
                    sel_pos = camera.unproject(last_pos, window_size);
                    println!(
                        "conv {:?} to {:?} win size {:?} ",
                        last_pos, sel_pos, window_size
                    );
                }
                WindowEvent::Key(key, action, modif) => {
                    println!("key event {:?} on {:?} with {:?}", key, action, modif);
                }
                WindowEvent::CursorPos(x, y, _modif) => {
                    last_pos = Vec2::new(x as f32, y as f32);
                }
                WindowEvent::Close => {
                    println!("close event");
                }
                _ => {}
            }
        }
        const CROSS_SIZE: f32 = 10.0;
        let up = Vec2::new(CROSS_SIZE, 0.0);
        window.draw_line_2d(sel_pos - up, sel_pos + up, draw_colour, 2.0);

        let right = Vec2::new(0.0, CROSS_SIZE);
        window.draw_line_2d(sel_pos - right, sel_pos + right, draw_colour, 2.0);
    }
}
