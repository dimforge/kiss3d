//! HDR film, tonemapping and bloom on the rasterization pipeline.
//!
//! The rasterizer now renders into an `Rgba16Float` HDR target and resolves it
//! with a tonemap + bloom pass, so emissive objects whose color exceeds `1.0`
//! glow instead of clipping to white. This example sets up several emissive
//! spheres of increasing brightness and lets you tweak the HDR knobs at runtime:
//!
//! * `B` toggles bloom on/off.
//! * `1` / `2` / `3` select the ACES / Reinhard / none tonemap operators.
//! * `Up` / `Down` raise / lower the exposure.

use kiss3d::event::{Action, Key};
use kiss3d::post_processing::Tonemap;
use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: hdr_bloom").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 2.0, 12.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();

    window.set_background_color(Color::new(0.01, 0.01, 0.02, 1.0));
    scene
        .add_light(Light::point(60.0))
        .set_position(Vec3::new(0.0, 8.0, 8.0));

    // A row of emissive spheres with intensities from dim to very bright. The
    // brightest ones go well above 1.0, so they only bloom thanks to the HDR
    // film + bloom pass.
    let intensities = [0.5_f32, 1.0, 2.0, 4.0, 8.0];
    for (i, &intensity) in intensities.iter().enumerate() {
        let x = (i as f32 - (intensities.len() as f32 - 1.0) * 0.5) * 2.5;
        let mut sphere = scene.add_sphere(0.7);
        sphere.translate(Vec3::new(x, 0.0, 0.0));
        // Warm emissive color scaled past 1.0 for the bright ones.
        sphere.set_color(Color::new(0.02, 0.02, 0.02, 1.0));
        sphere.set_emissive(Color::new(
            intensity,
            intensity * 0.7,
            intensity * 0.3,
            1.0,
        ));
    }

    // A non-emissive floor so the lit, sub-1.0 part of the image is visible too.
    scene
        .add_cube(20.0, 0.2, 20.0)
        .translate(Vec3::new(0.0, -1.5, 0.0))
        .set_color(Color::new(0.3, 0.3, 0.35, 1.0));

    // Enable bloom and start from neutral exposure / ACES tonemapping.
    window.set_tonemap(Tonemap::Aces);
    window.set_bloom_enabled(true);
    window.set_bloom(1.0, 0.08);
    window.set_exposure(1.0);

    while window.render_3d(&mut scene, &mut camera).await {
        if window.canvas().get_key(Key::B) == Action::Press {
            let enabled = window.hdr_settings().bloom_enabled;
            window.set_bloom_enabled(!enabled);
        }
        if window.canvas().get_key(Key::Key1) == Action::Press {
            window.set_tonemap(Tonemap::Aces);
        }
        if window.canvas().get_key(Key::Key2) == Action::Press {
            window.set_tonemap(Tonemap::Reinhard);
        }
        if window.canvas().get_key(Key::Key3) == Action::Press {
            window.set_tonemap(Tonemap::None);
        }
        if window.canvas().get_key(Key::Up) == Action::Press {
            let e = window.hdr_settings().exposure;
            window.set_exposure((e * 1.02).min(16.0));
        }
        if window.canvas().get_key(Key::Down) == Action::Press {
            let e = window.hdr_settings().exposure;
            window.set_exposure((e / 1.02).max(0.05));
        }
    }
}
