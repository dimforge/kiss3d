//! Color grading in the HDR tonemap pass, driven from an egui panel.
//!
//! A colorful scene with live grading controls (white balance, saturation,
//! contrast, gamma, hue) applied in linear space before tonemapping.
//!
//! Run with the `egui` feature: `cargo run --features egui --example color_grading`.

#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!("The 'egui' feature must be enabled: cargo run --features egui --example color_grading");
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    use kiss3d::post_processing::ColorGrading;
    use kiss3d::prelude::*;

    let mut window = Window::new("Kiss3d: color_grading").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 1.5, 9.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();

    window.set_background_color(Color::new(0.03, 0.03, 0.05, 1.0));
    scene
        .add_light(Light::point(80.0).with_intensity(5.0))
        .set_position(Vec3::new(3.0, 6.0, 6.0));
    scene.add_light(Light::directional(Vec3::new(-0.3, -0.6, -0.5)).with_intensity(1.5));

    let colors = [RED, Color::new(1.0, 0.6, 0.0, 1.0), YELLOW, LIME, BLUE];
    for (i, &col) in colors.iter().enumerate() {
        let x = (i as f32 - 2.0) * 1.8;
        let mut s = scene.add_sphere(0.7);
        s.translate(Vec3::new(x, 0.0, 0.0));
        s.set_color(col);
        s.set_roughness(0.4);
    }
    scene
        .add_cube(14.0, 0.2, 6.0)
        .translate(Vec3::new(0.0, -1.2, 0.0))
        .set_color(Color::new(0.5, 0.5, 0.5, 1.0));

    // UI state (ColorGrading is Copy).
    let mut grading = ColorGrading::default();

    while window.render_3d(&mut scene, &mut camera).await {
        window.hdr_settings_mut().color_grading = grading;

        window.draw_ui(|ctx| {
            egui::Window::new("Color grading")
                .default_width(280.0)
                .show(ctx, |ui| {
                    ui.label("White balance (RGB gain):");
                    ui.color_edit_button_rgb(&mut grading.white_balance);
                    ui.separator();
                    ui.add(
                        egui::Slider::new(&mut grading.saturation, 0.0..=3.0).text("saturation"),
                    );
                    ui.add(egui::Slider::new(&mut grading.contrast, 0.1..=3.0).text("contrast"));
                    ui.add(egui::Slider::new(&mut grading.gamma, 0.1..=3.0).text("gamma"));
                    ui.add(egui::Slider::new(&mut grading.hue, -3.1416..=3.1416).text("hue (rad)"));
                    ui.separator();
                    if ui.button("Reset").clicked() {
                        grading = ColorGrading::default();
                    }
                });
        });
    }
}
