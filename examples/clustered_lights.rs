//! Clustered (forward+) lighting: hundreds of small, moving, colored point lights
//! over a field of geometry, plus one shadow-casting directional "sun".
//!
//! The point lights have short attenuation radii and don't cast shadows, so they
//! flow through the clustered forward+ path (light culling in a compute pass +
//! per-cluster shading). The sun stays in the fixed primary tier and keeps full
//! shadow-map support. On WebGL2 (no compute / storage), only the first few lights
//! light the scene via the legacy fixed path.

use kiss3d::prelude::*;

const GRID: usize = 16; // GRID×GRID point lights
const SPACING: f32 = 2.0;

// A simple rainbow palette so neighbouring lights have distinct colors.
fn hue(t: f32) -> Color {
    let h = (t.fract() + 1.0).fract() * 6.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as u32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    Color::new(r, g, b, 1.0)
}

#[kiss3d::main]
async fn main() {
    env_logger::init();
    let mut window = Window::new("Kiss3d: clustered lighting").await;
    window.set_background_color(Color::new(0.02, 0.02, 0.04, 1.0));
    window.set_ambient(0.02);

    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 18.0, 28.0), Vec3::new(0.0, 0.0, 0.0));
    let mut scene = SceneNode3d::empty();

    let extent = GRID as f32 * SPACING;

    // Ground plane.
    scene
        .add_cube(extent + 4.0, 0.2, extent + 4.0)
        .set_position(Vec3::new(0.0, -0.6, 0.0))
        .set_color(Color::new(0.5, 0.5, 0.55, 1.0));

    // A field of pillars for the point lights to wash over.
    let pillars = (GRID / 2) as i32;
    for ix in -pillars..pillars {
        for iz in -pillars..pillars {
            let h = 1.0 + ((ix * 7 + iz * 13) & 3) as f32 * 0.6;
            scene
                .add_cube(0.7, h, 0.7)
                .set_position(Vec3::new(
                    ix as f32 * SPACING * 2.0 + 1.0,
                    h * 0.5 - 0.5,
                    iz as f32 * SPACING * 2.0 + 1.0,
                ))
                .set_color(Color::new(0.8, 0.8, 0.8, 1.0));
        }
    }

    // Shadow-casting directional "sun" (stays in the primary tier).
    scene
        .add_light(
            Light::directional(Vec3::new(-0.4, -1.0, -0.3))
                .with_color(Color::new(1.0, 0.97, 0.9, 1.0))
                .with_intensity(0.6),
        )
        .set_position(Vec3::new(0.0, 20.0, 0.0));

    // A grid of small, colored, non-shadowing point lights → clustered tier.
    let mut lights = Vec::new();
    for i in 0..GRID {
        for j in 0..GRID {
            let color = hue((i * GRID + j) as f32 / (GRID * GRID) as f32);
            let node = scene
                .add_light(
                    Light::point(4.0)
                        .with_color(color)
                        .with_intensity(6.0)
                        .with_casts_shadows(false),
                )
                .set_position(Vec3::new(0.0, 1.0, 0.0));
            lights.push(node);
        }
    }
    println!(
        "Spawned {} point lights (+1 directional). Clustered lighting: {}.",
        lights.len(),
        if Context::get().supports_clustered_lighting() {
            "ON"
        } else {
            "fallback (fixed 8-light path)"
        }
    );

    let mut t = 0.0f32;
    while window.render_3d(&mut scene, &mut camera).await {
        t += 0.015;
        for (n, node) in lights.iter_mut().enumerate() {
            let i = (n / GRID) as f32;
            let j = (n % GRID) as f32;
            // Grid anchor for this light.
            let ax = (i - (GRID as f32 - 1.0) * 0.5) * SPACING;
            let az = (j - (GRID as f32 - 1.0) * 0.5) * SPACING;
            // Orbit the anchor in a horizontal circle (phase offset per light) so
            // the colored pools sweep across the floor, plus a vertical bob.
            let phase = t + (i + j) * 0.5;
            let radius = SPACING * 1.5;
            let x = ax + radius * phase.cos();
            let z = az + radius * (phase * 0.7).sin();
            let y = 1.2 + 0.9 * (phase * 1.3).sin();
            node.set_position(Vec3::new(x, y, z));
        }
    }
}
