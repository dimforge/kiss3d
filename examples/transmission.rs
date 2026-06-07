//! Interactive refractive-glass (transmission) playground.
//!
//! Three glass shapes (sphere, cube, capsule) sit in front of a colorful, lit
//! environment — a rainbow arc of spheres, a back row of metals, and a few
//! emissive accent orbs — so you can watch the glass refract the scene behind it.
//! An egui panel drives every glass parameter live:
//!   - base color, transmission, roughness, metallic, reflectance
//!   - refraction: index of refraction (IOR) and volume thickness
//!   - Beer-Lambert volume tint (color + distance, toggleable)
//!
//! Drag to orbit. Crank `thickness` and `ior` for strong lensing; raise
//! `roughness` for frosted glass; lower `transmission` to fade back to a solid
//! dielectric. The skybox provides image-based lighting and the reflections.
//!
//! Run with the `egui` feature: `cargo run --features egui --example transmission`.

#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!("The 'egui' feature must be enabled: cargo run --features egui --example transmission");
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    use kiss3d::prelude::*;
    use std::path::Path;

    let mut window = Window::new("Kiss3d: transmission").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 1.5, 9.0), Vec3::new(0.0, 0.5, 0.0));
    let mut scene = SceneNode3d::empty();

    // The skybox is the image-based light source and what the metals reflect; the
    // glass refracts both it and the colored objects placed in front of it.
    window.set_skybox_from_file(Path::new("./examples/media/skybox.png"));
    window.set_ambient(0.0);
    scene.add_light(Light::directional(Vec3::new(-0.4, -0.8, -0.5)).with_intensity(2.5));
    scene
        .add_light(Light::point(60.0).with_intensity(5.0))
        .set_position(Vec3::new(3.0, 4.0, 4.0));

    // --- Backdrop: a colorful, lit environment for the glass to refract. ---
    // A rainbow arc of diffuse spheres.
    let rainbow = [
        Color::new(0.90, 0.15, 0.15, 1.0),
        Color::new(0.95, 0.55, 0.10, 1.0),
        Color::new(0.95, 0.90, 0.15, 1.0),
        Color::new(0.20, 0.75, 0.30, 1.0),
        Color::new(0.15, 0.55, 0.95, 1.0),
        Color::new(0.45, 0.25, 0.85, 1.0),
        Color::new(0.85, 0.25, 0.65, 1.0),
    ];
    for (i, c) in rainbow.iter().enumerate() {
        let t = i as f32 / (rainbow.len() as f32 - 1.0);
        let x = (t - 0.5) * 9.0;
        let y = 0.4 + (t * std::f32::consts::PI).sin() * 2.2; // arc
        let mut s = scene.add_sphere(0.7);
        s.translate(Vec3::new(x, y, -4.5));
        s.set_color(*c);
        s.set_roughness(0.35);
    }
    // A back row of metals for richer reflections seen through the glass.
    for i in 0..6 {
        let x = (i as f32 - 2.5) * 1.9;
        let mut s = scene.add_sphere(0.6);
        s.translate(Vec3::new(x, -1.6, -6.5));
        s.set_color(Color::new(0.80, 0.80, 0.85, 1.0));
        s.set_metallic(1.0);
        s.set_roughness(0.1 + 0.14 * i as f32);
    }
    // Emissive accent orbs — magnified into sharp bright points through the glass.
    for (x, y) in [(-3.5f32, 2.6f32), (3.2, 2.9), (0.0, -2.4)] {
        let mut o = scene.add_sphere(0.22);
        o.translate(Vec3::new(x, y, -3.5));
        o.set_color(WHITE);
        o.set_emissive(Color::new(3.0, 2.6, 2.0, 1.0));
        o.set_casts_shadows(false);
    }

    // --- Foreground glass shapes, driven by the sliders and spinning in place. ---
    // They OVERLAP at staggered depths (front -> back: clear sphere, red cube, green
    // capsule) so each is partly behind the previous. With 1 transmission step the
    // shapes behind are invisible where they fall behind another glass shape (only
    // the opaque scene refracts); raising "transmission steps" reveals each layer of
    // glass-through-glass. Each shape carries its own volume tint so the effect reads.
    let glass_tints = [
        [0.85f32, 0.92, 1.0], // front sphere: near-clear
        [0.95f32, 0.35, 0.30], // middle cube: red
        [0.35f32, 0.85, 0.45], // back capsule: green
    ];
    let mut glass = Vec::new();
    {
        // A big clear sphere in FRONT, with the red cube and green capsule mostly
        // behind it. With 1 transmission step the shapes behind are hidden where the
        // front sphere covers them (it refracts only the opaque scene); raising
        // "transmission steps" makes the colored glass show THROUGH the front sphere,
        // one layer at a time.
        let mut s = scene.add_sphere(1.2);
        s.translate(Vec3::new(0.0, 0.5, 2.8));
        glass.push(s);
        let mut c = scene.add_cube(1.2, 1.2, 1.2);
        c.translate(Vec3::new(1.4, 0.5, 0.6));
        glass.push(c);
        let mut cap = scene.add_capsule(0.5, 1.2);
        cap.translate(Vec3::new(-1.4, 0.4, 0.6));
        glass.push(cap);
    }

    // UI state (glass material parameters).
    let mut color = [0.90f32, 0.95, 1.0];
    let mut transmission = 1.0f32;
    let mut roughness = 0.05f32;
    let mut metallic = 0.0f32;
    let mut reflectance = 0.75f32;
    let mut ior = 1.15f32;
    let mut thickness = 1.5f32;
    // Each shape has its own tint color (above); this slider scales how dense it is.
    let mut atten_distance = 2.5f32;
    let mut tinted = true;
    let mut spin = true;
    let mut blur_quality = TransmissionBlurQuality::High;
    let mut steps = 4u32;

    while window.render_3d(&mut scene, &mut camera).await {
        // Roughness-blur quality + glass-behind-glass passes for the refraction.
        {
            let ts = window.transmission_settings_mut();
            ts.blur_quality = blur_quality;
            ts.steps = steps;
        }

        // Apply the current material to every glass shape (each keeps its own tint).
        for (i, s) in glass.iter_mut().enumerate() {
            s.set_color(Color::new(color[0], color[1], color[2], 1.0));
            s.set_transmission(transmission);
            s.set_roughness(roughness);
            s.set_metallic(metallic);
            s.set_reflectance(reflectance);
            s.set_ior(ior);
            s.set_thickness(thickness);
            let distance = if tinted { atten_distance } else { f32::INFINITY };
            let tint = glass_tints[i];
            s.set_attenuation(Color::new(tint[0], tint[1], tint[2], 1.0), distance);
            if spin {
                s.rotate(Quat::from_axis_angle(Vec3::Z, 0.01));
            }
        }

        window.draw_ui(|ctx| {
            egui::Window::new("Glass material")
                .default_width(300.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("base color");
                        ui.color_edit_button_rgb(&mut color);
                    });
                    ui.add(egui::Slider::new(&mut transmission, 0.0..=1.0).text("transmission"));
                    ui.add(egui::Slider::new(&mut roughness, 0.0..=1.0).text("roughness"));
                    ui.add(egui::Slider::new(&mut metallic, 0.0..=1.0).text("metallic"));
                    ui.add(egui::Slider::new(&mut reflectance, 0.0..=1.0).text("reflectance"));
                    ui.separator();
                    ui.label("Refraction");
                    ui.add(egui::Slider::new(&mut ior, 1.0..=2.5).text("ior"));
                    ui.add(egui::Slider::new(&mut thickness, 0.0..=5.0).text("thickness"));
                    egui::ComboBox::from_label("blur quality")
                        .selected_text(format!("{:?}", blur_quality))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut blur_quality,
                                TransmissionBlurQuality::Low,
                                "Low",
                            );
                            ui.selectable_value(
                                &mut blur_quality,
                                TransmissionBlurQuality::Medium,
                                "Medium",
                            );
                            ui.selectable_value(
                                &mut blur_quality,
                                TransmissionBlurQuality::High,
                                "High",
                            );
                        });
                    ui.add(
                        egui::Slider::new(&mut steps, 1..=4)
                            .text("transmission steps (glass thru glass)"),
                    );
                    ui.separator();
                    ui.label("Volume tint (Beer-Lambert; each shape has its own color)");
                    ui.checkbox(&mut tinted, "tinted");
                    ui.add_enabled(
                        tinted,
                        egui::Slider::new(&mut atten_distance, 0.1..=10.0).text("tint distance"),
                    );
                    ui.separator();
                    ui.checkbox(&mut spin, "spin shapes");
                });
        });
    }
}
