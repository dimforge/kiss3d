#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!("The 'egui' feature must be enabled for this example to work.")
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    use kiss3d::prelude::*;

    let mut window = Window::new("Kiss3d: egui UI").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 0.5, 1.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));

    window.set_background_color(LIGHT_STEEL_BLUE);

    let mut cube = scene.add_cube(0.2, 0.2, 0.2).set_color(RED);

    // UI state
    let mut rotation_speed = 0.014;
    let mut opacity = 1.0;
    let mut cube_color = [1.0, 0.0, 0.0];

    // Render loop
    while window.render_3d(&mut scene, &mut camera).await {
        // Rotate cube
        let rot_current = Quat::from_axis_angle(Vec3::Y, rotation_speed);
        cube.rotate(rot_current);

        // Update cube color
        cube.set_color(Color::new(cube_color[0], cube_color[1], cube_color[2], opacity));

        // Draw UI
        window.draw_ui(|ctx| {
            egui::Window::new("Kiss3d egui Example")
                .default_width(300.0)
                .show(ctx, |ui| {
                    // Rotation control
                    ui.label("Rotation Speed:");
                    ui.add(egui::Slider::new(&mut rotation_speed, 0.0..=0.1));

                    ui.separator();

                    // Opacity control
                    ui.label("Opacity:");
                    ui.add(egui::Slider::new(&mut opacity, 0.0..=1.0));

                    // Color picker
                    ui.label("Cube Color:");

                    ui.horizontal(|ui| {
                        ui.color_edit_button_rgb(&mut cube_color);
                        if ui.button("Randomize").clicked() {
                            // Randomize cube color
                            cube_color = [rand::random(), rand::random(), rand::random()];
                        }
                    });
                });
        });
    }
}
