//! Equirectangular skybox background for the rasterizer.
//!
//! Builds a procedural HDR sky (gradient + sun) so the example is self-contained,
//! and lets you rotate it / change its intensity from an egui panel. Pass a path
//! to an equirectangular `.hdr`/EXR on the command line to use a real environment
//! instead.
//!
//! Setting a skybox also enables image-based lighting (IBL), so the metallic
//! spheres reflect the environment and pick up its fill light.
//!
//! Run with the `egui` feature: `cargo run --features egui --example skybox`.

#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!("The 'egui' feature must be enabled: cargo run --features egui --example skybox");
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    use kiss3d::prelude::*;
    #[cfg(not(target_arch = "wasm32"))]
    use std::path::Path;

    let mut window = Window::new("Kiss3d: skybox").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 1.0, 6.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();

    // A few objects in front of the sky.
    scene.add_light(Light::directional(Vec3::new(-0.5, -0.7, -0.4)).with_intensity(2.5));
    for i in 0..4 {
        let x = (i as f32 - 1.5) * 1.6;
        scene
            .add_sphere(0.6)
            .translate(Vec3::new(x, 0.0, 0.0))
            .set_metallic(1.0)
            .set_roughness(0.1 + 0.25 * i as f32)
            .set_color(Color::new(0.9, 0.9, 0.92, 1.0));
    }

    // Load an HDRI from the command line (native only), or fall back to the bundled
    // sky — embedded into the binary on wasm, which has no filesystem.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let from_file = std::env::args()
            .nth(1)
            .map(|p| window.set_skybox_from_file(Path::new(&p)))
            .unwrap_or(false);
        if !from_file {
            window.set_skybox_from_file(Path::new("./examples/media/skybox.png"));
        }
    }
    #[cfg(target_arch = "wasm32")]
    window.set_skybox_from_memory(include_bytes!("media/skybox.png"));

    let mut rotation = 0.0f32;
    let mut intensity = 1.0f32;
    let mut enabled = true;

    while window.render_3d(&mut scene, &mut camera).await {
        window.set_skybox_orientation(rotation, intensity);

        window.draw_ui(|ctx| {
            egui::Window::new("Skybox")
                .default_width(240.0)
                .show(ctx, |ui| {
                    ui.checkbox(&mut enabled, "Enabled");
                    ui.add(egui::Slider::new(&mut rotation, 0.0..=6.2832).text("rotation (rad)"));
                    ui.add(egui::Slider::new(&mut intensity, 0.0..=4.0).text("intensity"));
                });
        });

        // Apply the enable toggle outside the UI closure (which borrows the UI ctx).
        if !enabled {
            window.clear_skybox();
        } else if !window.has_skybox() {
            #[cfg(not(target_arch = "wasm32"))]
            window.set_skybox_from_file(Path::new("./examples/media/skybox.png"));
            #[cfg(target_arch = "wasm32")]
            window.set_skybox_from_memory(include_bytes!("media/skybox.png"));
        }
    }
}
