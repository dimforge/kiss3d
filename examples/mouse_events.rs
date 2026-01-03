//! Demonstrates mouse events and coordinate system conversions.
//!
//! A small cross follows the cursor, and clicking places a persistent cross at that location.
use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Mouse events").await;
    let mut camera = PanZoomCamera2d::default();
    let mut scene = SceneNode2d::empty();

    let cursor_color = RED;
    let click_color = GREEN;

    let mut cursor_pos = Vec2::new(0.0f32, 0.0f32);
    let mut click_positions: Vec<Vec2> = Vec::new();

    while window.render_2d(&mut scene, &mut camera).await {
        let window_size = Vec2::new(window.size()[0] as f32, window.size()[1] as f32);

        for event in window.events().iter() {
            match event.value {
                WindowEvent::MouseButton(button, Action::Press, modif) => {
                    println!("mouse press event on {:?} with {:?}", button, modif);
                    let world_pos = camera.unproject(cursor_pos, window_size);
                    click_positions.push(world_pos);
                    println!(
                        "placed cross at {:?} (screen: {:?})",
                        world_pos, cursor_pos
                    );
                }
                WindowEvent::CursorPos(x, y, _modif) => {
                    cursor_pos = Vec2::new(x as f32, y as f32);
                }
                _ => {}
            }
        }

        // Draw cross following cursor
        let cursor_world_pos = camera.unproject(cursor_pos, window_size);
        draw_cross(&mut window, cursor_world_pos, 40.0, cursor_color, 8.0);

        // Draw persistent crosses at click positions
        for &pos in &click_positions {
            draw_cross(&mut window, pos, 25.0, click_color, 4.0);
        }
    }
}

fn draw_cross(window: &mut Window, pos: Vec2, size: f32, color: Color, thickness: f32) {
    let h = Vec2::new(size, 0.0);
    let v = Vec2::new(0.0, size);
    window.draw_line_2d(pos - h, pos + h, color, thickness);
    window.draw_line_2d(pos - v, pos + v, color, thickness);
}
