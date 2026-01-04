use kiss3d::post_processing::OculusStereo;
use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new_with_size("Kiss3d: stereo", 1280, 800).await;
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));
    let mut cube = scene.add_cube(1.0, 1.0, 1.0).set_color(RED);

    let eye = Vec3::new(1.0f32, 2.0, 10.0);
    let at = Vec3::ZERO;
    let mut camera = FirstPersonCamera3dStereo::new(eye, at, 0.3f32);

    let mut oculus_stereo = OculusStereo::new();

    while window
        .render(
            Some(&mut scene),
            None,
            Some(&mut camera),
            None,
            None,
            Some(&mut oculus_stereo),
        )
        .await
    {
        cube.rotate(Quat::from_rotation_y(0.02));
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
