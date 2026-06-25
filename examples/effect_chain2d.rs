use kiss3d::post_processing::{Crt, Gi2d, GiEmitter2d, GiOccluder2d, PostProcessingEffect};
use kiss3d::prelude::*;

// Demonstrates chaining multiple post-processing effects with `render_2d_with_chain`:
// screen-space global illumination (`Gi2d`) feeds its lit result into a CRT
// stylizer (`Crt`). Each effect is a full-screen pass; the chain ping-pongs between
// two targets and the last writes the frame.
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: chained 2D post-processing").await;
    window.set_background_color(Color::new(0.0, 0.0, 0.0, 1.0));
    window.set_bloom_enabled(true);

    let mut camera = PanZoomCamera2d::new(Vec2::ZERO, 1.5);
    let mut scene = SceneNode2d::empty();

    // A surface plus a couple of occluders for the GI pass to shadow.
    scene
        .add_rectangle(1400.0, 1000.0)
        .set_color(Color::new(0.8, 0.82, 0.9, 1.0));
    let occluder_pos = [Vec2::new(-150.0, 30.0), Vec2::new(140.0, -50.0)];
    let occluder_radius = 55.0;
    let occluders: Vec<GiOccluder2d> = occluder_pos
        .iter()
        .map(|&p| {
            scene
                .add_circle(occluder_radius)
                .translate(p)
                .set_color(Color::new(0.04, 0.04, 0.05, 1.0));
            GiOccluder2d::new(p, occluder_radius)
        })
        .collect();

    let glow = Color::new(2.6, 1.4, 0.4, 1.0);
    let mut emitter = scene.add_circle(20.0);
    emitter.set_color(glow);

    let mut gi = Gi2d::new();
    gi.set_ambient(Color::new(0.05, 0.05, 0.07, 1.0));
    gi.set_rays(48);
    gi.set_occluders(&occluders);

    let mut crt = Crt::new();
    crt.set_curvature(0.16);
    crt.set_scanlines(0.25, 700.0);

    let mut t = 0.0f32;
    while {
        gi.set_camera(&camera);
        let mut chain: [&mut dyn PostProcessingEffect; 2] = [&mut gi, &mut crt];
        window
            .render_2d_with_chain(&mut scene, &mut camera, &mut chain)
            .await
    } {
        t += 0.012;
        let pos = Vec2::new(t.cos() * 280.0, t.sin() * 200.0);
        emitter.set_position(pos);
        gi.set_emitters(&[GiEmitter2d::new(pos, 20.0, glow, 3.0)]);
    }
}
