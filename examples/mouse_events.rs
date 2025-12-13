//! Test of kiss3d's planar camera. Just moves a cross around the screen whenever the mouse is clicked. Shows conversions between co-ordinate systems.
extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::event::{Action, WindowEvent};
use kiss3d::light::Light;
use kiss3d::planar_camera::*;
use kiss3d::window::Window;

/// main program
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Mouse events").await;
    let mut camera = kiss3d::planar_camera::PlanarFixedView::new();
    window.set_light(Light::StickToCamera);
    let draw_colour = na::Point3::new(0.5, 1.0, 0.5);
    let mut last_pos = na::Point2::new(0.0f32, 0.0f32);
    let mut sel_pos = na::Point2::new(0.0f32, 0.0f32);
    while window
        .render_with(None, Some(&mut camera), None, None)
        .await
    {
        for event in window.events().iter() {
            match event.value {
                WindowEvent::FramebufferSize(x, y) => {
                    println!("frame buffer size event {}, {}", x, y);
                }
                WindowEvent::MouseButton(button, Action::Press, modif) => {
                    println!("mouse press event on {:?} with {:?}", button, modif);
                    let window_size =
                        na::Vector2::new(window.size()[0] as f32, window.size()[1] as f32);
                    sel_pos = camera.unproject(&last_pos, &window_size);
                    println!(
                        "conv {:?} to {:?} win size {:?} ",
                        last_pos, sel_pos, window_size
                    );
                }
                WindowEvent::Key(key, action, modif) => {
                    println!("key event {:?} on {:?} with {:?}", key, action, modif);
                }
                WindowEvent::CursorPos(x, y, _modif) => {
                    last_pos = na::Point2::new(x as f32, y as f32);
                }
                WindowEvent::Close => {
                    println!("close event");
                }
                _ => {}
            }
        }
        const CROSS_SIZE: f32 = 10.0;
        let up = na::Vector2::new(CROSS_SIZE, 0.0);
        window.draw_planar_line(&(sel_pos - up), &(sel_pos + up), &draw_colour, 2.0);

        let right = na::Vector2::new(0.0, CROSS_SIZE);
        window.draw_planar_line(&(sel_pos - right), &(sel_pos + right), &draw_colour, 2.0);
    }
}
