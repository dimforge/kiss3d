//! Reflection probes + screen-space reflections (SSR) on the rasterizer.
//!
//! A reflective floor and a row of chrome spheres.
//! Two reflection features add localized, view-dependent reflections on top of
//! the global skybox IBL:
//!
//! - A **runtime-captured reflection probe** renders the live scene into a
//!   parallax-corrected environment map, so the floor and spheres reflect each
//!   other (press *Recapture* after moving things).
//! - A **baked-image reflection probe** (a colored gradient) shows how a probe's
//!   localized influence box overrides the skybox inside its volume.
//! - **SSR** ray-marches the depth/G-buffer to add sharp on-screen reflections
//!   that fall back to the probe/skybox where the screen has no data.
//!
//! Toggle each from the egui panel to compare.
//!
//! Run with the `egui` feature: `cargo run --features egui --example reflections`.

use std::path::Path;

#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!("The 'egui' feature must be enabled: cargo run --features egui --example reflections");
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    use kiss3d::prelude::*;

    let mut window = Window::new("Kiss3d: reflections").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 2.5, 9.0), Vec3::new(0.0, 0.5, 0.0));
    let mut scene = SceneNode3d::empty();

    window.set_ambient(0.1);
    scene.add_light(Light::directional(Vec3::new(-0.5, -0.8, -0.4)).with_intensity(2.5));
    window.set_skybox_from_file(Path::new("./examples/media/skybox.png"));

    // Reflective floor: a brushed-metal slab. A slightly rough (not perfect-mirror)
    // floor blurs the SSR reflections, so the partial reflections at steep
    // top-down angles read as soft and natural instead of hollow rings (pure SSR
    // can't reflect rays pointing back toward the camera, leaving gaps a mirror
    // would expose).
    let mut floor = scene.add_cube(24.0, 0.1, 24.0);
    floor.set_position(Vec3::new(0.0, -1.0, 0.0));
    floor.set_color(Color::new(0.5, 0.5, 0.55, 1.0));
    floor.set_metallic(1.0);
    floor.set_roughness(0.18);

    // Chrome spheres of increasing roughness. They orbit the middle sphere and bob
    // up and down (animated in the render loop). Each is stored with its orbit
    // radius and starting angle so the initial layout matches the original row; the
    // middle sphere has radius 0, so it stays centered and just bobs in place.
    let palette = [
        Color::new(0.95, 0.64, 0.54, 1.0),
        Color::new(0.55, 0.85, 0.65, 1.0),
        Color::new(0.55, 0.7, 0.95, 1.0),
        Color::new(0.9, 0.85, 0.5, 1.0),
        Color::new(0.85, 0.55, 0.85, 1.0),
    ];
    let mut spheres = Vec::new();
    for (i, color) in palette.iter().enumerate() {
        let mut s = scene.add_sphere(0.9);
        s.set_color(*color);
        s.set_metallic(0.9);
        s.set_roughness(0.05 + 0.18 * i as f32);
        // Put the (moving) spheres on render layer 1 so they're excluded from the
        // probe capture below — a single-point probe distorts nearby objects, so
        // SSR (which reflects them accurately) handles them instead. The floor
        // stays on the default layer 0, which the probe does capture.
        s.set_render_layers(0b10);
        let offset = (i as f32 - 2.0) * 2.4; // signed distance from the middle sphere
        let radius = offset.abs();
        let base_angle = if offset >= 0.0 { 0.0 } else { std::f32::consts::PI };
        spheres.push((s, radius, base_angle));
    }

    // The runtime probe captures only render layer 0 (the floor + skybox), not the
    // dynamic spheres on layer 1.
    window.set_reflection_capture_layers(0b01);

    // Probe 0: runtime-captured, covering the whole scene (reflects real geometry).
    let captured = window
        .add_reflection_probe(ReflectionProbe {
            center: Vec3::new(0.0, 0.5, 0.0),
            half_extents: Vec3::new(12.0, 5.0, 12.0),
            falloff: 2.0,
            intensity: 1.0,
            rotation: 0.0,
        })
        .expect("probe slot 0");

    // Probe 1: a baked colored gradient inside a small box on the right, to show a
    // localized probe overriding the skybox reflection within its influence volume.
    let baked = window
        .add_reflection_probe(ReflectionProbe::new(Vec3::new(5.0, 0.5, 0.0), 2.2))
        .expect("probe slot 1");
    window.set_reflection_probe_image(baked, &colored_gradient(256, 128));

    let mut t = 0.0f32;
    let mut animate = true;
    let mut ssr_enabled = true;
    let mut probe_intensity = 1.0f32;
    // Global SSR march/quality settings (shared by all objects).
    let mut ssr = SsrSettings::default();
    // Per-object SSR properties, applied to the spheres below; tweak from the UI.
    let mut sphere_ssr = SsrMaterial::default();
    // Per-object gating demo: whether the floor receives SSR at all.
    let mut floor_receives_ssr = true;

    while window.render_3d(&mut scene, &mut camera).await {
        if animate {
            t += 0.015;
        }
        // Orbit each sphere around the middle one in the XZ plane (the middle
        // sphere has radius 0, so it stays put) and bob them up and down, with a
        // per-sphere phase so they ripple rather than move in lockstep.
        for (i, (sphere, radius, base_angle)) in spheres.iter_mut().enumerate() {
            let angle = *base_angle + t * 0.8;
            let x = *radius * angle.cos();
            let z = *radius * angle.sin();
            let y = 0.7 + 0.7 * (t * 1.6 + i as f32 * 0.7).sin();
            sphere.set_position(Vec3::new(x, y, z));
            sphere.set_ssr(Some(sphere_ssr));
        }
        // Gate the floor's SSR on/off per object.
        floor.set_ssr(if floor_receives_ssr {
            Some(SsrMaterial::default())
        } else {
            None
        });
        // Re-capture the probe each frame so its reflections track the motion.
        window.capture_reflection_probe(captured);

        window.set_ssr_enabled(ssr_enabled);
        *window.ssr_settings_mut() = ssr;
        if let Some(p) = window.reflection_probe_mut(captured) {
            p.intensity = probe_intensity;
        }

        window.draw_ui(|ctx| {
            egui::Window::new("Reflections")
                .default_width(300.0)
                .show(ctx, |ui| {
                    ui.checkbox(&mut animate, "Animate (orbit + bob)");
                    ui.separator();
                    ui.checkbox(&mut ssr_enabled, "Screen-space reflections");
                    ui.add_enabled_ui(ssr_enabled, |ui| {
                        ui.label("Global (march quality):");
                        ui.add(egui::Slider::new(&mut ssr.intensity, 0.0..=2.0).text("intensity"));
                        ui.add(egui::Slider::new(&mut ssr.max_steps, 8..=128).text("max steps"));
                        ui.add(egui::Slider::new(&mut ssr.thickness, 0.01..=3.0).text("thickness"));
                        ui.add(
                            egui::Slider::new(&mut ssr.max_distance, 1.0..=200.0).text("max distance"),
                        );
                        ui.add(
                            egui::Slider::new(&mut ssr.roughness_cutoff, 0.0..=1.0)
                                .text("roughness cutoff"),
                        );
                        ui.add(egui::Slider::new(&mut ssr.edge_fade, 0.0..=0.5).text("edge fade"));
                        ui.separator();
                        ui.label("Per-object (spheres):");
                        ui.add(
                            egui::Slider::new(&mut sphere_ssr.intensity, 0.0..=2.0)
                                .text("sphere intensity"),
                        );
                        ui.checkbox(&mut sphere_ssr.infinite_thick, "infinite thick");
                        ui.checkbox(&mut sphere_ssr.distance_attenuation, "distance attenuation");
                        ui.checkbox(&mut sphere_ssr.fresnel, "fresnel");
                        ui.separator();
                        ui.checkbox(&mut floor_receives_ssr, "floor receives SSR");
                    });
                    ui.separator();
                    ui.add(
                        egui::Slider::new(&mut probe_intensity, 0.0..=2.0).text("probe intensity"),
                    );
                    ui.separator();
                    ui.label("Probe 0: runtime capture (whole scene)");
                    ui.label("Probe 1: baked gradient (right box)");
                });
        });
    }
}

/// A vivid equirectangular gradient used as a baked reflection probe, so its
/// localized influence is obvious against the sky.
#[cfg(feature = "egui")]
fn colored_gradient(w: u32, h: u32) -> image::DynamicImage {
    let buf = image::ImageBuffer::from_fn(w, h, |_x, y| {
        let t = y as f32 / h as f32;
        let top = glamx::Vec3::new(1.0, 0.2, 0.5);
        let bottom = glamx::Vec3::new(0.2, 0.5, 1.0);
        let col = top * (1.0 - t) + bottom * t;
        image::Rgb([col.x * 1.5, col.y * 1.5, col.z * 1.5])
    });
    image::DynamicImage::ImageRgb32F(buf)
}
