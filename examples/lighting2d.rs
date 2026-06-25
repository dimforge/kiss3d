use kiss3d::light2d::{Light2d, Light2dManager};
use kiss3d::prelude::*;

// Demonstrates dynamic 2D lighting: a grid of lit sprites (LitMaterial2d) under
// several moving colored point lights and a sweeping spot light, plus a dim
// ambient term. With a normal map the sprites would shade per-pixel; here they are
// flat, so you see each light's smooth radial falloff and the spot's cone.
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: 2D lighting").await;
    window.set_background_color(Color::new(0.0, 0.0, 0.0, 1.0));

    let mut camera = PanZoomCamera2d::new(Vec2::ZERO, 2.5);
    let mut scene = SceneNode2d::empty();

    // A grid of flat gray lit tiles to catch the light.
    let tile = 64.0;
    let gap = 4.0;
    for gy in -4..=4 {
        for gx in -6..=6 {
            scene
                .add_lit_sprite(tile, tile)
                .set_color(Color::new(0.8, 0.8, 0.85, 1.0))
                .set_lit_params(LitParams::default().with_specular(0.4, 32.0))
                .translate(Vec2::new(
                    gx as f32 * (tile + gap),
                    gy as f32 * (tile + gap),
                ));
        }
    }

    Light2dManager::get_global_manager(|m| m.set_ambient(Color::new(0.06, 0.06, 0.08, 1.0)));

    let mut t = 0.0f32;
    while window.render_2d(&mut scene, &mut camera).await {
        t += 0.016;

        let red = Light2d::point(
            Vec2::new(t.cos() * 260.0, t.sin() * 160.0),
            Color::new(1.0, 0.3, 0.2, 1.0),
            3.0,
            260.0,
        )
        .with_height(70.0);

        let blue = Light2d::point(
            Vec2::new((t * 1.3 + 2.0).cos() * 220.0, (t * 0.9).cos() * 180.0),
            Color::new(0.3, 0.5, 1.0, 1.0),
            3.0,
            260.0,
        )
        .with_height(70.0);

        // A spot light sweeping its aim direction around.
        let spot = Light2d::spot(
            Vec2::new(0.0, 220.0),
            Vec2::new((t * 0.7).sin() * 0.6, -1.0),
            Color::new(1.0, 1.0, 0.8, 1.0),
            4.0,
            420.0,
            0.25,
            0.5,
        )
        .with_height(60.0);

        Light2dManager::get_global_manager(|m| m.set_lights(&[red, blue, spot]));
    }
}
