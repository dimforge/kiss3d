//! Screen-space 2D global illumination (`Gi2d`): emitter discs light a surface,
//! occluder discs cast soft shadows, and light bleeds in color. A bright orbiting
//! emitter sweeps moving penumbras across a field of occluders.
//!
//! An egui panel exposes every `Gi2d` dial — solver (direct ray-march vs. radiance
//! cascades), jump-flood SDF occluders, ambient, resolution scale, march steps/
//! distance, rays-per-pixel and temporal blend (direct solver), and cascade levels /
//! base directions (cascade solver) — plus the emitter intensity.
//!
//! Run with the `egui` feature: `cargo run --features egui --example global_illumination2d`.

#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!(
        "The 'egui' feature must be enabled: cargo run --features egui --example global_illumination2d"
    );
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    use kiss3d::post_processing::{Gi2d, GiEmitter2d, GiOccluder2d};
    use kiss3d::prelude::*;

    let mut window = Window::new("Kiss3d: 2D global illumination").await;
    window.set_background_color(Color::new(0.0, 0.0, 0.0, 1.0));
    window.set_bloom_enabled(true);

    let mut camera = PanZoomCamera2d::new(Vec2::ZERO, 1.5);
    let mut scene = SceneNode2d::empty();

    // A large surface to catch the light (GI modulates whatever is drawn).
    scene
        .add_rectangle(1600.0, 1200.0)
        .set_color(Color::new(0.85, 0.85, 0.9, 1.0));

    // A grid of static occluder discs, drawn dark and registered with the GI solver.
    // Kept under `MAX_OCCLUDERS` (64): a 5-row by 11-column grid = 55 discs.
    let occluder_radius = 26.0;
    let mut occluders: Vec<GiOccluder2d> = Vec::new();
    for row in -2..=2 {
        for col in -5..=5 {
            let stagger = if row % 2 == 0 { 0.0 } else { 55.0 };
            let p = Vec2::new(col as f32 * 110.0 + stagger, row as f32 * 95.0);
            scene
                .add_circle(occluder_radius)
                .translate(p)
                .set_color(Color::new(0.05, 0.05, 0.06, 1.0));
            occluders.push(GiOccluder2d::new(p, occluder_radius));
        }
    }

    // A static warm emitter, plus a moving bright emitter we orbit each frame.
    let warm = Color::new(1.5, 0.7, 0.2, 1.0);
    let cool = Color::new(0.4, 0.8, 2.0, 1.0);
    scene
        .add_circle(22.0)
        .set_color(warm)
        .translate(Vec2::new(-260.0, -200.0));

    let mut mover = scene.add_circle(20.0);
    mover.set_color(cool);

    let mut gi = Gi2d::new();
    gi.set_occluders(&occluders);

    // Live-editable GI settings (defaults match Gi2d::new()).
    let mut use_cascades = false;
    let mut use_sdf = true;
    let mut ambient = [0.06f32, 0.06, 0.08];
    let mut resolution_scale = 2u32;
    let mut max_steps = 32u32;
    let mut max_distance = 2000.0f32;
    let mut emitter_intensity = 3.0f32;
    let mut rays = 8u32;
    let mut temporal_blend = 0.85f32;
    let mut cascade_levels = 5u32;
    let mut base_directions = 16u32;

    let mut t = 0.0f32;
    while !window.should_close() {
        t += 0.012;
        let mover_pos = Vec2::new(t.cos() * 300.0, t.sin() * 220.0);
        mover.set_position(mover_pos);
        gi.set_emitters(&[
            GiEmitter2d::new(
                Vec2::new(-260.0, -200.0),
                22.0,
                warm,
                emitter_intensity * 0.85,
            ),
            GiEmitter2d::new(mover_pos, 20.0, cool, emitter_intensity),
        ]);

        window.draw_ui(|ctx| {
            egui::Window::new("Global Illumination").show(ctx, |ui| {
                ui.label(if use_cascades {
                    "solver: radiance cascades"
                } else {
                    "solver: direct ray-march"
                });
                ui.checkbox(&mut use_cascades, "Radiance cascades");
                ui.checkbox(&mut use_sdf, "Jump-flood SDF occluders");

                ui.separator();
                ui.label("Common");
                ui.horizontal(|ui| {
                    ui.label("ambient");
                    ui.color_edit_button_rgb(&mut ambient);
                });
                ui.add(
                    egui::Slider::new(&mut resolution_scale, 1..=4).text("resolution scale (1/n)"),
                );
                ui.add(egui::Slider::new(&mut max_steps, 8..=128).text("max march steps"));
                ui.add(egui::Slider::new(&mut max_distance, 200.0..=4000.0).text("max distance"));
                ui.add(
                    egui::Slider::new(&mut emitter_intensity, 0.0..=8.0).text("emitter intensity"),
                );

                ui.separator();
                ui.label("Direct ray-march");
                ui.add_enabled(
                    !use_cascades,
                    egui::Slider::new(&mut rays, 1..=64).text("rays / pixel"),
                );
                ui.add_enabled(
                    !use_cascades,
                    egui::Slider::new(&mut temporal_blend, 0.0..=0.95).text("temporal blend"),
                );

                ui.separator();
                ui.label("Radiance cascades");
                ui.add_enabled(
                    use_cascades,
                    egui::Slider::new(&mut cascade_levels, 1..=8).text("cascade levels"),
                );
                ui.add_enabled_ui(use_cascades, |ui| {
                    egui::ComboBox::from_label("base directions")
                        .selected_text(format!("{base_directions}"))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut base_directions, 4, "4");
                            ui.selectable_value(&mut base_directions, 16, "16");
                            ui.selectable_value(&mut base_directions, 64, "64");
                        });
                });
            });
        });

        // Apply the panel state to the GI effect.
        gi.set_radiance_cascades(use_cascades);
        gi.set_sdf_occluders(use_sdf);
        gi.set_ambient(Color::new(ambient[0], ambient[1], ambient[2], 1.0));
        gi.set_resolution_scale(resolution_scale);
        gi.set_max_steps(max_steps);
        gi.set_max_distance(max_distance);
        gi.set_rays(rays);
        gi.set_temporal_blend(temporal_blend);
        gi.set_cascade_count(cascade_levels);
        gi.set_cascade_base_directions(base_directions);

        gi.set_camera(&camera);
        if !window
            .render_2d_with(&mut scene, &mut camera, &mut gi)
            .await
        {
            break;
        }
    }
}
