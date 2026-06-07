//! FXAA and CAS post-processing, selected from an egui panel.
//!
//! A picket of thin boxes (lots of near-vertical edges) makes aliasing obvious.
//! FXAA smooths edges; CAS sharpens detail. Pick the effect in the panel.
//!
//! Run with the `egui` feature: `cargo run --features egui --example antialiasing`.

#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!("The 'egui' feature must be enabled: cargo run --features egui --example antialiasing");
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    use kiss3d::post_processing::{Cas, Fxaa};
    use kiss3d::prelude::*;

    #[derive(Copy, Clone, PartialEq)]
    enum Mode {
        None,
        Fxaa,
        Cas,
    }

    let mut window = Window::new("Kiss3d: antialiasing").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 4.0, 10.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();

    window.set_background_color(Color::new(0.05, 0.05, 0.08, 1.0));
    scene
        .add_light(Light::point(120.0).with_intensity(4.0))
        .set_position(Vec3::new(4.0, 8.0, 6.0));

    // A picket of thin tall boxes — lots of near-vertical edges to alias.
    for i in 0..21 {
        let x = (i as f32 - 10.0) * 0.6;
        let mut b = scene.add_cube(0.08, 4.0, 0.08);
        b.translate(Vec3::new(x, 0.0, 0.0));
        b.set_color(Color::new(0.9, 0.9, 0.95, 1.0));
    }

    let mut fxaa = Fxaa::new();
    let mut cas = Cas::new(0.6);

    // UI state.
    let mut mode = Mode::Fxaa;
    let mut sharpness = 0.6f32;

    loop {
        cas.set_sharpness(sharpness);

        let still_open = match mode {
            Mode::Fxaa => {
                window
                    .render(
                        Some(&mut scene),
                        None,
                        Some(&mut camera),
                        None,
                        None,
                        Some(&mut fxaa),
                    )
                    .await
            }
            Mode::Cas => {
                window
                    .render(
                        Some(&mut scene),
                        None,
                        Some(&mut camera),
                        None,
                        None,
                        Some(&mut cas),
                    )
                    .await
            }
            Mode::None => {
                window
                    .render(Some(&mut scene), None, Some(&mut camera), None, None, None)
                    .await
            }
        };
        if !still_open {
            break;
        }

        window.draw_ui(|ctx| {
            egui::Window::new("Anti-aliasing")
                .default_width(240.0)
                .show(ctx, |ui| {
                    ui.radio_value(&mut mode, Mode::None, "None");
                    ui.radio_value(&mut mode, Mode::Fxaa, "FXAA");
                    ui.radio_value(&mut mode, Mode::Cas, "CAS (sharpen)");
                    ui.separator();
                    ui.add_enabled(
                        mode == Mode::Cas,
                        egui::Slider::new(&mut sharpness, 0.0..=1.0).text("CAS sharpness"),
                    );
                });
        });
    }
}
