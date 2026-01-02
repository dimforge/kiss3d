use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    env_logger::init();
    let mut window = Window::new("Kiss3d: multi-light").await;
    window.set_background_color(Color::new(0.05, 0.05, 0.1, 1.0));
    window.set_ambient(0.1);

    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();

    // Add some geometry
    scene
        .add_cube(5.0, 0.2, 5.0)
        .set_position(Vec3::new(0.0, -1.0, 0.0))
        .set_color(GRAY);

    let mut cube = scene
        .add_cube(1.0, 1.0, 1.0)
        .set_position(Vec3::new(-1.0, 0.0, 0.0))
        .set_color(WHITE);

    scene
        .add_sphere(0.5)
        .set_position(Vec3::new(1.0, 0.0, 0.0))
        .set_color(WHITE);

    // Add a directional green light from above
    scene
        .add_light(
            Light::directional(Vec3::NEG_Y)
                .with_color(GREEN)
                .with_intensity(2.0),
        )
        .set_position(Vec3::new(0.0, 2.0, 0.0));

    // Add a red point light using add_light with a custom Light
    let mut red_light = scene
        .add_light(Light::point(20.0).with_color(RED).with_intensity(5.0))
        .set_position(Vec3::new(-3.0, 2.0, 0.0));

    // Add a blue point light
    let mut blue_light = scene
        .add_light(Light::point(20.0).with_color(BLUE).with_intensity(5.0))
        .set_position(Vec3::new(3.0, 2.0, 0.0));


    let rot = Quat::from_axis_angle(Vec3::Y, 0.01);
    let mut time = 0.0f32;

    while window.render_3d(&mut scene, &mut camera).await {
        cube.rotate(rot);

        // Animate the lights in a circular pattern
        time -= 0.02;
        red_light.set_position(Vec3::new(
            -3.0 * time.cos(),
            3.0 * time.sin(),
            2.0,
        ));
        blue_light.set_position(Vec3::new(
            2.0,
            3.0 * time.sin(),
            -3.0 * time.cos(),
        ));
    }
}
