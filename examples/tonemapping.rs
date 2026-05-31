//! Compare the rasterizer's HDR tonemapping operators side by side.
//!
//! kiss3d renders into a linear HDR film and resolves it with a selectable
//! tonemap operator. Use the egui panel to switch between them on a scene of
//! saturated colors and a bright emissive bar, and watch how each handles
//! saturation and highlights:
//!
//! * **None** — clamp only (the pre-HDR look): most saturated, but bright values
//!   hard-clip.
//! * **ACES** — cinematic, but desaturates colors and skews some hues ("washed
//!   out"); kept mainly for comparison.
//! * **Reinhard** — simple `x/(1+x)`; dims and desaturates broadly.
//! * **AgX** — neutral filmic: graceful highlight roll-off, no hue skews, gentle
//!   desaturation only near white.
//! * **Khronos PBR Neutral** (default) — preserves in-gamut saturation,
//!   desaturating only true highlights; the least "washed out".
//! * **Tony McMapface** — Tomasz Stachowiak's perceptual display transform,
//!   sampled from its baked CC0 3D LUT.
//!
//! A checkbox switches between the rasterizer and the GPU path tracer — the same
//! operator applies to both, so you can compare the tonemappers in either backend.
//!
//! Run with the `egui` feature: `cargo run --features egui --example tonemapping`.

#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!("The 'egui' feature must be enabled: cargo run --features egui --example tonemapping");
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    use kiss3d::post_processing::Tonemap;
    use kiss3d::prelude::*;
    use kiss3d::renderer::RayTracer;

    let mut window = Window::new("Kiss3d: tonemapping comparison").await;
    window.set_background_color(Color::new(0.09, 0.10, 0.13, 1.0));
    window.set_ambient(0.3);

    let mut camera =
        OrbitCamera3d::new_with_frustum(0.9, 0.1, 100.0, Vec3::new(0.0, 1.6, 6.5), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();

    scene
        .add_cube(14.0, 0.3, 14.0)
        .set_position(Vec3::new(0.0, -1.4, 0.0))
        .set_color(Color::new(0.7, 0.7, 0.72, 1.0));

    // A row of vivid, saturated spheres — the colors tonemappers treat differently.
    for (i, color) in [
        Color::new(0.95, 0.05, 0.05, 1.0),
        Color::new(0.95, 0.5, 0.05, 1.0),
        Color::new(0.9, 0.85, 0.1, 1.0),
        Color::new(0.05, 0.7, 0.1, 1.0),
        Color::new(0.1, 0.3, 0.95, 1.0),
        Color::new(0.6, 0.1, 0.85, 1.0),
    ]
    .iter()
    .copied()
    .enumerate()
    {
        scene
            .add_sphere(0.55)
            .set_position(Vec3::new(i as f32 * 1.2 - 3.0, 0.0, 0.0))
            .set_color(color);
    }

    // A bright emissive bar (HDR > 1) to show each operator's highlight roll-off.
    scene
        .add_cube(7.0, 0.5, 0.5)
        .set_position(Vec3::new(0.0, 1.7, -1.5))
        .set_color(WHITE)
        .set_emissive(Color::new(6.0, 5.0, 2.5, 1.0));

    scene
        .add_light(
            Light::directional(Vec3::new(-0.4, -0.7, -0.5))
                .with_color(Color::new(1.0, 0.97, 0.92, 1.0))
                .with_intensity(3.0),
        )
        .set_position(Vec3::new(4.0, 6.0, 3.0));

    // The same tonemap operators apply to both backends, so the comparison holds
    // whether the scene is rasterized or path-traced.
    let mut raytracer = RayTracer::new();

    // UI state (start at the default operator).
    let mut tonemap = Tonemap::default();
    let mut exposure = 1.0f32;
    let mut bloom = false;
    let mut pathtrace = false;

    loop {
        let still_open = if pathtrace {
            raytracer.set_tonemap(tonemap);
            raytracer.set_exposure(exposure);
            window
                .render_raytraced(&mut scene, &mut camera, &mut raytracer)
                .await
        } else {
            window.set_tonemap(tonemap);
            window.set_exposure(exposure);
            window.set_bloom_enabled(bloom);
            window.render_3d(&mut scene, &mut camera).await
        };
        if !still_open {
            break;
        }

        window.draw_ui(|ctx| {
            egui::Window::new("Tonemapping")
                .default_width(260.0)
                .show(ctx, |ui| {
                    ui.checkbox(&mut pathtrace, "Path tracing (vs. rasterizer)");
                    ui.separator();
                    ui.label("Operator:");
                    ui.radio_value(&mut tonemap, Tonemap::None, "None (clamp)");
                    ui.radio_value(&mut tonemap, Tonemap::Aces, "ACES");
                    ui.radio_value(&mut tonemap, Tonemap::Reinhard, "Reinhard");
                    ui.radio_value(&mut tonemap, Tonemap::AgX, "AgX  (neutral filmic)");
                    ui.radio_value(
                        &mut tonemap,
                        Tonemap::Neutral,
                        "Khronos PBR Neutral (default)",
                    );
                    ui.radio_value(&mut tonemap, Tonemap::TonyMcMapface, "Tony McMapface (LUT)");
                    ui.separator();
                    ui.add(egui::Slider::new(&mut exposure, 0.1..=4.0).text("Exposure"));
                    ui.add_enabled(
                        !pathtrace,
                        egui::Checkbox::new(&mut bloom, "Bloom (rasterizer only)"),
                    );
                });
        });
    }
}
