extern crate kiss3d;
extern crate nalgebra as na;

#[cfg(feature = "egui")]
use kiss3d::light::Light;
#[cfg(feature = "egui")]
use kiss3d::window::Window;
#[cfg(feature = "egui")]
use na::{UnitQuaternion, Vector3};

#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!("The 'egui' feature must be enabled for this example to work.")
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: egui UI").await;
    window.set_background_color(0.9, 0.9, 0.9);

    let mut cube = window.add_cube(0.2, 0.2, 0.2);
    cube.set_color(1.0, 0.0, 0.0);

    window.set_light(Light::StickToCamera);

    // UI state
    let mut rotation_speed = 0.014;
    let mut cube_color = [1.0, 0.0, 0.0];

    // Render loop
    while window.render().await {
        // Rotate cube
        let rot_current = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), rotation_speed);
        cube.prepend_to_local_rotation(&rot_current);

        // Update cube color
        cube.set_color(cube_color[0], cube_color[1], cube_color[2]);

        // Draw UI
        window.draw_ui(|ctx| {
            egui::Window::new("Kiss3d egui Example")
                .default_width(300.0)
                .show(ctx, |ui| {
                    // Rotation control
                    ui.label("Rotation Speed:");
                    ui.add(egui::Slider::new(&mut rotation_speed, 0.0..=0.1));

                    ui.separator();

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
