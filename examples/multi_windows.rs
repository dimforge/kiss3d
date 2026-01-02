use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window1 = Some(Window::new("Kiss3d multi-window 1").await);
    let mut window2 = Some(Window::new("Kiss3d multi-window 2").await);

    let mut camera1 = OrbitCamera3d::default();
    let mut scene1 = SceneNode3d::empty();
    scene1
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 2.0, -2.0));

    let mut camera2 = OrbitCamera3d::default();
    let mut scene2 = SceneNode3d::empty();
    scene2
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 2.0, -2.0));

    let mut c1 = scene1.add_cube(1.0, 1.0, 1.0).set_color(RED);
    let mut c2 = scene2.add_cube(1.0, 1.0, 1.0).set_color(GREEN);

    // Only exit when both windows have been closed.
    while window1.is_some() || window2.is_some() {
        if let Some(window) = &mut window1 {
            if !window.render_3d(&mut scene1, &mut camera1).await {
                window1 = None;
            }
        }

        if let Some(window) = &mut window2 {
            if !window.render_3d(&mut scene2, &mut camera2).await {
                window2 = None;
            }
        }

        c1.rotate(Quat::from_axis_angle(Vec3::Y, 0.05));
        c2.rotate(Quat::from_axis_angle(Vec3::X, -0.05));
    }
}
