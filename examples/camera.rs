use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let eye = Vec3::new(10.0f32, 10.0, 10.0);
    let at = Vec3::ZERO;
    let mut first_person = FirstPersonCamera3d::new(eye, at);
    let mut arc_ball = OrbitCamera3d::new(eye, at);
    let mut use_arc_ball = true;

    let mut window = Window::new("Kiss3d: camera").await;
    let mut scene = SceneNode3d::empty();

    while !window.should_close() {
        // rotate the arc-ball camera.
        let curr_yaw = arc_ball.yaw();
        arc_ball.set_yaw(curr_yaw + 0.05);

        // update the current camera.
        for event in window.events().iter() {
            match event.value {
                WindowEvent::Key(key, Action::Release, _) => {
                    if key == Key::Numpad1 {
                        use_arc_ball = true
                    } else if key == Key::Numpad2 {
                        use_arc_ball = false
                    }
                }
                _ => {}
            }
        }

        window.draw_line(Vec3::ZERO, Vec3::X, RED, 2.0, false);
        window.draw_line(Vec3::ZERO, Vec3::Y, GREEN, 2.0, false);
        window.draw_line(Vec3::ZERO, Vec3::Z, BLUE, 2.0, false);

        if use_arc_ball {
            window.render_3d(&mut scene, &mut arc_ball).await;
        } else {
            window.render_3d(&mut scene, &mut first_person).await;
        }
    }
}
