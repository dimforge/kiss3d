extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::camera::FirstPersonStereo;
use kiss3d::event::{Action, Key, WindowEvent};
use kiss3d::light::Light;
use kiss3d::post_processing::OculusStereo;
use kiss3d::window::Window;
use na::Point3;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new_with_size("Kiss3d: stereo", 1280, 800).await;

    let mut c = window.add_cube(1.0, 1.0, 1.0);

    let eye = Point3::new(0.0f32, 0.0, 10.0);
    let at = Point3::new(0.0f32, 0.0, 0.0);
    let mut camera = FirstPersonStereo::new(eye, at, 0.3f32);

    c.set_color(1.0, 0.0, 0.0);

    window.set_light(Light::StickToCamera);

    let mut oculus_stereo = OculusStereo::new();

    while window
        .render_with_camera_and_effect(&mut camera, &mut oculus_stereo)
        .await
    {
        for event in window.events().iter() {
            match event.value {
                WindowEvent::Key(Key::Numpad1, Action::Release, _) => {
                    let ipd = camera.ipd();
                    camera.set_ipd(ipd + 0.1f32);
                }
                WindowEvent::Key(Key::Numpad2, Action::Release, _) => {
                    let ipd = camera.ipd();
                    camera.set_ipd(ipd - 0.1f32);
                }
                _ => {}
            }
        }
    }
}
