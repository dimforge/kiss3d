use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: events").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();

    while window.render_3d(&mut scene, &mut camera).await {
        for mut event in window.events().iter() {
            match event.value {
                WindowEvent::Key(button, Action::Press, _) => {
                    println!("You pressed the button: {:?}", button);
                    println!("Do not try to press escape: the event is inhibited!");
                    event.inhibited = true // override the default keyboard handler
                }
                WindowEvent::Key(button, Action::Release, _) => {
                    println!("You released the button: {:?}", button);
                    println!("Do not try to press escape: the event is inhibited!");
                    event.inhibited = true // override the default keyboard handler
                }
                WindowEvent::MouseButton(button, Action::Press, mods) => {
                    println!("You pressed the mouse button: {:?}", button);
                    println!("You pressed the mouse button with modifiers: {:?}", mods);
                    // dont override the default mouse handler
                }
                WindowEvent::MouseButton(button, Action::Release, mods) => {
                    println!("You released the mouse button: {:?}", button);
                    println!("You released the mouse button with modifiers: {:?}", mods);
                    // dont override the default mouse handler
                }
                WindowEvent::CursorPos(x, y, _) => {
                    println!("Cursor pos: ({} , {})", x, y);
                    // dont override the default mouse handler
                }
                WindowEvent::Scroll(xshift, yshift, _) => {
                    println!("Cursor pos: ({} , {})", xshift, yshift);
                    // dont override the default mouse handler
                }
                _ => {}
            }
        }
    }
}
