//! Depth of field (DoF) on the rasterizer.
//!
//! A long row of colored spheres recedes into the distance. A thin-lens blur keeps
//! whatever sits at the focal distance sharp and progressively blurs everything
//! nearer and farther — the classic photographic shallow-depth-of-field look.
//!
//! Drag the *focal distance* slider to rack focus along the row, open the aperture
//! (lower f-stops) for a shallower in-focus band and stronger background blur, and
//! switch between the smooth *Gaussian* blur and the harder-edged *Bokeh* discs.
//!
//! Run with the `egui` feature: `cargo run --features egui --example depth_of_field`.

#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!(
        "The 'egui' feature must be enabled: cargo run --features egui --example depth_of_field"
    );
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    use kiss3d::prelude::*;

    let mut window = Window::new("Kiss3d: depth of field").await;
    // Look straight down the row so the depth gradient (and thus the blur) is
    // obvious from front to back.
    let mut camera = OrbitCamera3d::new(Vec3::new(2.5, 2.0, 6.0), Vec3::new(0.0, 0.0, -8.0));
    let mut scene = SceneNode3d::empty();

    window.set_ambient(0.15);
    scene.add_light(Light::directional(Vec3::new(-0.4, -0.8, -0.3)).with_intensity(2.5));

    // A ground plane to give the blur something continuous to fall off over.
    let mut floor = scene.add_cube(40.0, 0.1, 40.0);
    floor.set_position(Vec3::new(0.0, -1.0, -10.0));
    floor.set_color(Color::new(0.35, 0.35, 0.4, 1.0));
    floor.set_roughness(0.9);

    // A row of spheres marching away from the camera, one every 2 units.
    let palette = [
        Color::new(0.95, 0.4, 0.4, 1.0),
        Color::new(0.95, 0.7, 0.35, 1.0),
        Color::new(0.9, 0.9, 0.4, 1.0),
        Color::new(0.45, 0.85, 0.5, 1.0),
        Color::new(0.4, 0.7, 0.95, 1.0),
        Color::new(0.6, 0.5, 0.95, 1.0),
        Color::new(0.9, 0.5, 0.85, 1.0),
        Color::new(0.95, 0.95, 0.95, 1.0),
    ];
    for (i, color) in palette.iter().enumerate() {
        let mut s = scene.add_sphere(0.7);
        s.set_position(Vec3::new(0.0, -0.3, -2.0 * i as f32));
        s.set_color(*color);
        s.set_metallic(0.1);
        s.set_roughness(0.4);
    }

    let mut enabled = true;
    let mut dof = DofSettings {
        focal_distance: 6.0,
        // A wide aperture so the shallow depth of field reads clearly at this scale.
        aperture_f_stops: 0.12,
        ..DofSettings::default()
    };
    let mut gaussian = dof.mode == DepthOfFieldMode::Gaussian;

    while window.render_3d(&mut scene, &mut camera).await {
        dof.mode = if gaussian {
            DepthOfFieldMode::Gaussian
        } else {
            DepthOfFieldMode::Bokeh
        };
        window.set_dof_enabled(enabled);
        *window.dof_settings_mut() = dof;

        window.draw_ui(|ctx| {
            egui::Window::new("Depth of field")
                .default_width(300.0)
                .show(ctx, |ui| {
                    ui.checkbox(&mut enabled, "Enabled");
                    ui.add_enabled_ui(enabled, |ui| {
                        ui.checkbox(&mut gaussian, "Gaussian (off = Bokeh)");
                        ui.add(
                            egui::Slider::new(&mut dof.focal_distance, 1.0..=24.0)
                                .text("focal distance"),
                        );
                        ui.add(
                            egui::Slider::new(&mut dof.aperture_f_stops, 0.05..=16.0)
                                .logarithmic(true)
                                .text("aperture (f-stops)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut dof.max_coc_diameter, 4.0..=96.0)
                                .text("max blur (px)"),
                        );
                        ui.add(egui::Slider::new(&mut dof.num_taps, 8..=96).text("gather taps"));
                    });
                    ui.separator();
                    ui.label("Spheres are spaced 2 units apart, starting at the camera.");
                });
        });
    }
}
