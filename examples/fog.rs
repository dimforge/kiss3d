//! Distance fog and colored ambient light, driven from an egui panel.
//!
//! A receding corridor of cubes shows the fog falloff. The panel selects the fog
//! mode (linear / exponential / exponential-squared / off), its parameters, and
//! the global ambient light color.
//!
//! Run with the `egui` feature: `cargo run --features egui --example fog`.

#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!("The 'egui' feature must be enabled: cargo run --features egui --example fog");
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    use kiss3d::prelude::*;

    // Local mirror of FogMode for the egui radio buttons (PartialEq selection).
    #[derive(Copy, Clone, PartialEq)]
    enum Mode {
        Off,
        Linear,
        Exponential,
        ExponentialSquared,
    }

    let mut window = Window::new("Kiss3d: fog").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 3.0, 6.0), Vec3::new(0.0, 0.0, -20.0));
    let mut scene = SceneNode3d::empty();

    let mut fog_color = [0.55f32, 0.6, 0.7];
    window.set_background_color(Color::new(fog_color[0], fog_color[1], fog_color[2], 1.0));
    window.set_ambient(0.3);
    scene
        .add_light(Light::point(200.0).with_intensity(4.0))
        .set_position(Vec3::new(0.0, 8.0, 4.0));

    // A long corridor of cubes receding from the camera.
    for i in 0..40 {
        let z = -(i as f32) * 1.5;
        let c = 0.4 + 0.6 * ((i % 2) as f32);
        scene
            .add_cube(0.8, 1.6, 0.8)
            .translate(Vec3::new(if i % 2 == 0 { -1.5 } else { 1.5 }, 0.0, z));
        let mut floor = scene.add_cube(6.0, 0.1, 1.4);
        floor.translate(Vec3::new(0.0, -0.85, z));
        floor.set_color(Color::new(c, c, c, 1.0));
    }

    // UI state.
    let mut mode = Mode::Exponential;
    let mut density = 0.06f32;
    let mut linear_start = 5.0f32;
    let mut linear_end = 45.0f32;
    let mut height_falloff = 0.0f32;
    let mut ambient_color = [1.0f32, 1.0, 1.0];

    while window.render_3d(&mut scene, &mut camera).await {
        let col = Color::new(fog_color[0], fog_color[1], fog_color[2], 1.0);
        let fog = match mode {
            Mode::Off => Fog::default(),
            Mode::Linear => Fog::linear(col, linear_start, linear_end),
            Mode::Exponential => Fog::exponential(col, density),
            Mode::ExponentialSquared => Fog::exponential_squared(col, density),
        }
        .with_height_falloff(height_falloff);
        window.set_fog(fog);
        window.set_background_color(col);
        window.set_ambient_color(Color::new(
            ambient_color[0],
            ambient_color[1],
            ambient_color[2],
            1.0,
        ));

        window.draw_ui(|ctx| {
            egui::Window::new("Fog")
                .default_width(260.0)
                .show(ctx, |ui| {
                    ui.label("Mode:");
                    ui.radio_value(&mut mode, Mode::Off, "Off");
                    ui.radio_value(&mut mode, Mode::Linear, "Linear");
                    ui.radio_value(&mut mode, Mode::Exponential, "Exponential");
                    ui.radio_value(&mut mode, Mode::ExponentialSquared, "Exponential²");
                    ui.separator();
                    if mode == Mode::Linear {
                        ui.add(egui::Slider::new(&mut linear_start, 0.0..=40.0).text("start"));
                        ui.add(egui::Slider::new(&mut linear_end, 1.0..=120.0).text("end"));
                    } else {
                        ui.add(
                            egui::Slider::new(&mut density, 0.0..=0.3)
                                .text("density")
                                .logarithmic(true),
                        );
                    }
                    ui.add(
                        egui::Slider::new(&mut height_falloff, 0.0..=0.5).text("height falloff"),
                    );
                    ui.separator();
                    ui.label("Fog color:");
                    ui.color_edit_button_rgb(&mut fog_color);
                    ui.label("Ambient color:");
                    ui.color_edit_button_rgb(&mut ambient_color);
                });
        });
    }
}
