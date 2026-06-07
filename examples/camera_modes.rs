//! Orthographic projection, physical exposure, and render layers — via egui.
//!
//! The panel toggles perspective/orthographic projection on the `OrbitCamera3d`,
//! sets the physical exposure (EV100), and chooses which render layer the camera
//! shows. Red cubes are on layer 0, green spheres on layer 1.
//!
//! Run with the `egui` feature: `cargo run --features egui --example camera_modes`.

#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!("The 'egui' feature must be enabled: cargo run --features egui --example camera_modes");
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    use kiss3d::prelude::*;

    #[derive(Copy, Clone, PartialEq)]
    enum Layers {
        Both,
        Layer0,
        Layer1,
    }

    let mut window = Window::new("Kiss3d: camera_modes").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 3.0, 12.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();

    window.set_background_color(Color::new(0.02, 0.02, 0.03, 1.0));
    window.set_ambient(0.2);
    scene
        .add_light(Light::point(120.0).with_intensity(5.0))
        .set_position(Vec3::new(4.0, 8.0, 8.0));

    // Layer 0: red cubes in a back row.
    for i in 0..5 {
        let x = (i as f32 - 2.0) * 2.0;
        let mut c = scene.add_cube(1.0, 1.0, 1.0);
        c.translate(Vec3::new(x, 0.0, -2.0));
        c.set_color(RED);
        c.set_render_layers(1 << 0);
    }
    // Layer 1: green spheres in a front row.
    for i in 0..5 {
        let x = (i as f32 - 2.0) * 2.0;
        let mut s = scene.add_sphere(0.6);
        s.translate(Vec3::new(x, 0.0, 2.0));
        s.set_color(LIME);
        s.set_render_layers(1 << 1);
    }

    // UI state.
    let mut orthographic = false;
    let mut ev100 = 7.0f32;
    let mut layers = Layers::Both;

    while window.render_3d(&mut scene, &mut camera).await {
        camera.set_projection(if orthographic {
            Projection::Orthographic
        } else {
            Projection::Perspective
        });
        camera.set_render_layers(match layers {
            Layers::Both => u32::MAX,
            Layers::Layer0 => 1 << 0,
            Layers::Layer1 => 1 << 1,
        });
        window.set_exposure_value(Exposure { ev100 });

        window.draw_ui(|ctx| {
            egui::Window::new("Camera")
                .default_width(240.0)
                .show(ctx, |ui| {
                    ui.checkbox(&mut orthographic, "Orthographic projection");
                    ui.separator();
                    ui.add(egui::Slider::new(&mut ev100, 0.0..=16.0).text("exposure (EV100)"));
                    ui.label("(lower EV = brighter)");
                    ui.separator();
                    ui.label("Visible render layers:");
                    ui.radio_value(&mut layers, Layers::Both, "Both");
                    ui.radio_value(&mut layers, Layers::Layer0, "Layer 0 (red cubes)");
                    ui.radio_value(&mut layers, Layers::Layer1, "Layer 1 (green spheres)");
                });
        });
    }
}
