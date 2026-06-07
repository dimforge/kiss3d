//! Extended PBR surface properties on the rasterizer.
//!
//! A grid of spheres whose clearcoat, anisotropy, reflectance and diffuse
//! transmission are driven live from an egui panel — the StandardMaterial
//! extensions. Drag to orbit; the differences are clearest
//! with the highlight moving across each sphere.
//!
//! Run with the `egui` feature: `cargo run --features egui --example material_pbr`.

#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!("The 'egui' feature must be enabled: cargo run --features egui --example material_pbr");
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    use kiss3d::prelude::*;

    let mut window = Window::new("Kiss3d: material_pbr").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();

    window.set_background_color(Color::new(0.02, 0.02, 0.03, 1.0));
    window.set_ambient(0.15);
    scene.add_light(Light::directional(Vec3::new(-0.4, -0.8, -0.6)).with_intensity(3.0));
    scene
        .add_light(Light::point(80.0).with_intensity(5.0))
        .set_position(Vec3::new(4.0, 6.0, 8.0));

    // A 4x4 grid of spheres; columns vary the swept parameter, rows pick which.
    let cols = 4;
    let rows = 4;
    let spacing = 2.2;
    let mut spheres = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let x = (c as f32 - (cols as f32 - 1.0) * 0.5) * spacing;
            let y = ((rows as f32 - 1.0) * 0.5 - r as f32) * spacing;
            let mut s = scene.add_sphere(0.9);
            s.translate(Vec3::new(x, y, 0.0));
            s.set_color(Color::new(0.8, 0.15, 0.12, 1.0));
            spheres.push(s);
        }
    }

    // UI state.
    let mut metallic = 1.0f32;
    let mut roughness = 0.35f32;
    let mut reflectance = 0.5f32;
    let mut clearcoat = 0.0f32;
    let mut clearcoat_roughness = 0.1f32;
    let mut anisotropy = 0.0f32;
    let mut transmission = 0.0f32;
    let mut color = [0.8f32, 0.15, 0.12];

    while window.render_3d(&mut scene, &mut camera).await {
        // Apply the current material settings to every sphere.
        for s in spheres.iter_mut() {
            s.set_color(Color::new(color[0], color[1], color[2], 1.0));
            s.set_metallic(metallic);
            s.set_roughness(roughness);
            s.set_reflectance(reflectance);
            s.set_clearcoat(clearcoat, clearcoat_roughness);
            s.set_anisotropy(anisotropy, 0.0);
            s.set_transmission(transmission);
        }

        window.draw_ui(|ctx| {
            egui::Window::new("Material")
                .default_width(280.0)
                .show(ctx, |ui| {
                    ui.color_edit_button_rgb(&mut color);
                    ui.add(egui::Slider::new(&mut metallic, 0.0..=1.0).text("metallic"));
                    ui.add(egui::Slider::new(&mut roughness, 0.0..=1.0).text("roughness"));
                    ui.add(egui::Slider::new(&mut reflectance, 0.0..=1.0).text("reflectance"));
                    ui.separator();
                    ui.add(egui::Slider::new(&mut clearcoat, 0.0..=1.0).text("clearcoat"));
                    ui.add(
                        egui::Slider::new(&mut clearcoat_roughness, 0.0..=1.0)
                            .text("clearcoat roughness"),
                    );
                    ui.separator();
                    ui.add(egui::Slider::new(&mut anisotropy, -1.0..=1.0).text("anisotropy"));
                    ui.add(egui::Slider::new(&mut transmission, 0.0..=1.0).text("transmission"));
                });
        });
    }
}
