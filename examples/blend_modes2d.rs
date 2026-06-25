use kiss3d::prelude::*;

// Demonstrates the 2D surface blend modes (`Blend2d`). Overlapping translucent
// circles are composited with, from left to right: alpha, additive, screen and
// multiply blending — so you can see how each mode combines the overlaps.
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: 2D blend modes").await;
    let mut camera = PanZoomCamera2d::new(Vec2::ZERO, 2.0);
    let mut scene = SceneNode2d::empty();

    // (label position, blend mode, base color triple) for each cluster.
    let clusters = [
        (-330.0, Blend2d::Alpha, [RED, GREEN, BLUE]),
        (-110.0, Blend2d::Additive, [RED, GREEN, BLUE]),
        (110.0, Blend2d::Screen, [RED, GREEN, BLUE]),
        (330.0, Blend2d::Multiply, [WHITE, CYAN, YELLOW]),
    ];

    // Three overlapping circles per cluster, arranged in a small triangle.
    let offsets = [
        Vec2::new(0.0, 32.0),
        Vec2::new(-28.0, -18.0),
        Vec2::new(28.0, -18.0),
    ];

    let mut groups = Vec::new();
    for (cx, blend, colors) in clusters {
        let mut group = scene.add_group();
        for (off, color) in offsets.iter().zip(colors) {
            group
                .add_circle(46.0)
                .translate(Vec2::new(cx, 0.0) + *off)
                .set_color(Color { a: 0.7, ..color })
                .set_blend(blend);
        }
        groups.push(group);
    }

    while window.render_2d(&mut scene, &mut camera).await {
        for group in &mut groups {
            group.append_rotation(0.005);
        }
    }
}
