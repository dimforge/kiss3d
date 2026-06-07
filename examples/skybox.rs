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
    use std::path::Path;

    let mut window = Window::new("Kiss3d: skybox").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 1.0, 6.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();

    // A few objects in front of the sky.
    scene
        .add_light(Light::directional(Vec3::new(-0.5, -0.7, -0.4)).with_intensity(2.5));
    for i in 0..4 {
        let x = (i as f32 - 1.5) * 1.6;
        let mut s = scene.add_sphere(0.6);
        s.translate(Vec3::new(x, 0.0, 0.0));
        s.set_metallic(1.0);
        s.set_roughness(0.1 + 0.25 * i as f32);
        s.set_color(Color::new(0.9, 0.9, 0.92, 1.0));
    }

    // Load an HDRI from the command line, or fall back to a procedural sky.
    let from_file = std::env::args()
        .nth(1)
        .map(|p| window.set_skybox_from_file(Path::new(&p)))
        .unwrap_or(false);
    if !from_file {
        window.set_skybox_image(&procedural_sky(1024, 512));
    }

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
                    ui.add(
                        egui::Slider::new(&mut rotation, 0.0..=6.2832).text("rotation (rad)"),
                    );
                    ui.add(egui::Slider::new(&mut intensity, 0.0..=4.0).text("intensity"));
                });
        });

        // Apply the enable toggle outside the UI closure (which borrows the UI ctx).
        if !enabled {
            window.clear_skybox();
        } else if !window.has_skybox() {
            window.set_skybox_image(&procedural_sky(1024, 512));
        }
    }
}

/// Builds a simple procedural equirectangular HDR sky: a blue zenith→horizon
/// gradient over a darker ground, with a warm sun disc. The latitude/longitude
/// convention matches the skybox shader (`v = acos(dir.y)/π`, `u` around Y).
#[cfg(feature = "egui")]
fn procedural_sky(w: u32, h: u32) -> image::DynamicImage {
    use std::f32::consts::PI;
    let sun_dir = glamx::Vec3::new(-0.5, 0.6, -0.4).normalize();
    let buf = image::ImageBuffer::from_fn(w, h, |x, y| {
        let u = (x as f32 + 0.5) / w as f32;
        let v = (y as f32 + 0.5) / h as f32;
        // Invert the shader mapping to get a world direction for this texel.
        let theta = v * PI; // 0 at +Y (up), PI at -Y (down)
        let phi = (u - 0.5) * 2.0 * PI; // around Y
        let dir = glamx::Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin());

        let up = dir.y.max(0.0);
        // Sky gradient (zenith blue -> warm horizon) above, dim ground below.
        let sky = glamx::Vec3::new(0.25, 0.45, 0.85) * up
            + glamx::Vec3::new(0.7, 0.75, 0.8) * (1.0 - up) * 0.6;
        let ground = glamx::Vec3::new(0.12, 0.11, 0.10);
        let mut col = if dir.y >= 0.0 { sky } else { ground };

        // Warm sun disc + glow.
        let s = dir.dot(sun_dir).clamp(-1.0, 1.0);
        let sun = (s.max(0.0)).powf(2000.0) * 60.0 + (s.max(0.0)).powf(50.0) * 1.5;
        col += glamx::Vec3::new(1.0, 0.9, 0.7) * sun;

        image::Rgb([col.x, col.y, col.z])
    });
    image::DynamicImage::ImageRgb32F(buf)
}
