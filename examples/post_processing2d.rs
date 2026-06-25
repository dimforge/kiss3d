use kiss3d::prelude::*;
use kiss3d::post_processing::Crt;

// Demonstrates the 2D post-processing stack: HDR bloom (bright, >1.0 colors bleed
// light) plus a CRT stylization effect (curvature, chromatic aberration, scanlines
// and vignette) applied via `render_2d_with`.
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: 2D post-processing").await;
    window.set_background_color(Color::new(0.02, 0.02, 0.05, 1.0));
    window.set_bloom_enabled(true);

    let mut camera = PanZoomCamera2d::new(Vec2::ZERO, 2.0);
    let mut scene = SceneNode2d::empty();

    // Over-bright HDR colors (components > 1) so the bloom pass picks them up.
    let neon_orange = Color::new(3.0, 1.2, 0.1, 1.0);
    let neon_cyan = Color::new(0.1, 2.4, 3.0, 1.0);
    let neon_pink = Color::new(3.0, 0.4, 1.8, 1.0);

    scene
        .add_rectangle(240.0, 120.0)
        .set_color(neon_cyan)
        .translate(Vec2::new(-180.0, 110.0));

    scene
        .add_circle(70.0)
        .set_color(neon_orange)
        .translate(Vec2::new(170.0, 100.0));

    let mut pulse = scene
        .add_circle(60.0)
        .set_color(neon_pink)
        .translate(Vec2::new(0.0, -120.0));

    let mut crt = Crt::new();
    crt.set_curvature(0.18);
    crt.set_scanlines(0.3, 600.0);

    let mut t = 0.0f32;
    while window.render_2d_with(&mut scene, &mut camera, &mut crt).await {
        t += 0.05;
        let s = 1.0 + 0.25 * (0.5 + 0.5 * t.sin());
        pulse.set_local_scale(s, s);
    }
}
