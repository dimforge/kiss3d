//! Anti-aliasing comparison: MSAA vs FXAA, viewed through a pixel magnifier.
//!
//! AA differences are notoriously hard to see at 1:1, so this example renders a
//! deliberately alias-prone scene (a fence of thin, slightly tilted pickets — lots
//! of high-contrast near-diagonal edges) and shows a **magnified loupe** of the
//! screen center in the corner, sampled nearest-neighbour so individual pixels read
//! as blocks. Toggle the modes in the panel and watch the jagged staircase in the
//! loupe smooth out:
//!
//! * **MSAA ×4** — multisamples geometry edges. Crisp interior, smooth silhouettes.
//!   This is set on the window (it resolves before tonemapping).
//! * **FXAA** — a cheap post-process that smooths any luminance edge, at the cost of
//!   slightly softening fine detail. Works on top of MSAA.
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
    use kiss3d::post_processing::{Fxaa, Loupe};
    use kiss3d::prelude::*;

    let mut window = Window::new("Kiss3d: antialiasing").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 1.5, 9.0), Vec3::new(0.0, 0.5, 0.0));
    let mut scene = SceneNode3d::empty();

    window.set_background_color(Color::new(0.02, 0.02, 0.03, 1.0));
    scene
        .add_light(Light::point(150.0).with_intensity(5.0))
        .set_position(Vec3::new(4.0, 8.0, 6.0));

    // A fence of thin, tall pickets. Tilting the whole group a few degrees turns the
    // vertical edges into near-diagonals, which produce the most obvious staircase
    // aliasing. High-contrast white-on-near-black maximizes edge visibility.
    let mut fence = scene.add_group();
    for i in 0..27 {
        let x = (i as f32 - 13.0) * 0.42;
        fence
            .add_cube(0.07, 4.0, 0.07)
            .set_color(Color::new(0.95, 0.95, 1.0, 1.0))
            .translate(Vec3::new(x, 0.0, 0.0));
    }
    // A few thin horizontal rails crossing the pickets add near-horizontal edges too.
    for j in 0..3 {
        let y = j as f32 * 1.2 - 0.7;
        fence
            .add_cube(11.0, 0.06, 0.06)
            .set_color(Color::new(1.0, 0.85, 0.4, 1.0))
            .translate(Vec3::new(0.0, y, 0.06));
    }
    fence.rotate(Quat::from_axis_angle(Vec3::Z, 0.10));

    // The loupe magnifies the rendered scene. To inspect FXAA under magnification it
    // *wraps* an Fxaa effect (only one post-processing effect can be active at once),
    // toggled by setting/clearing the inner effect.
    let mut loupe = Loupe::new();

    // UI state.
    let mut msaa = true;
    let mut fxaa = false;
    let mut prev_fxaa = !fxaa;
    let mut zoom = 8.0f32;

    loop {
        window.set_samples(if msaa {
            NumSamples::Four
        } else {
            NumSamples::One
        });
        if fxaa != prev_fxaa {
            loupe.set_inner(fxaa.then(|| Box::new(Fxaa::new()) as Box<_>));
            prev_fxaa = fxaa;
        }
        loupe.set_zoom(zoom);

        let still_open = window
            .render(
                Some(&mut scene),
                None,
                Some(&mut camera),
                None,
                None,
                Some(&mut loupe),
            )
            .await;
        if !still_open {
            break;
        }

        let samples = window.samples();
        window.draw_ui(|ctx| {
            egui::Window::new("Anti-aliasing")
                .default_width(260.0)
                .show(ctx, |ui| {
                    ui.label("Drag to orbit. The loupe magnifies the screen center.");
                    ui.separator();
                    ui.checkbox(&mut msaa, "MSAA ×4 (geometry edges)");
                    ui.checkbox(&mut fxaa, "FXAA (post-process)");
                    ui.separator();
                    ui.add(egui::Slider::new(&mut zoom, 4.0..=20.0).text("Loupe zoom"));
                    ui.separator();
                    ui.label(format!(
                        "Effective: {} {}",
                        if samples > 1 {
                            format!("MSAA ×{samples}")
                        } else {
                            "no MSAA".to_string()
                        },
                        if fxaa { "+ FXAA" } else { "" }
                    ));
                });
        });
    }
}
