use kiss3d::prelude::*;
use std::path::Path;

// Demonstrates sprites: a plain sprite quad, sprite-sheet frame animation
// (`SpriteSheet` + `set_sprite_frame`), and a 9-slice panel whose corners keep
// their size while the center stretches. The kitten texture stands in for both a
// 2x2 sheet and a 9-slice panel so the example needs no extra assets.
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: 2D sprites").await;
    let mut camera = PanZoomCamera2d::new(Vec2::ZERO, 2.0);
    let mut scene = SceneNode2d::empty();

    let kitten = Path::new("./examples/media/kitten.png");

    // A plain sprite quad showing the whole texture.
    scene
        .add_sprite(160.0, 160.0)
        .translate(Vec2::new(-260.0, 0.0))
        .set_texture_from_file(kitten, "kitten");

    // The same texture treated as a 2x2 sprite sheet; we cycle through its 4 frames.
    let sheet = SpriteSheet::new(2, 2);
    let mut animated = scene
        .add_sprite(160.0, 160.0)
        .set_texture_with_name("kitten");
    animated.set_sprite_frame(&sheet, 0);

    // A 9-slice panel, scaled wide: corners keep their size, edges/center stretch.
    scene
        .add_nine_slice(
            Vec2::new(360.0, 160.0),
            Border::uniform(28.0),
            Border::uniform(0.25),
        )
        .translate(Vec2::new(240.0, 0.0))
        .set_texture_with_name("kitten");

    let mut frame = 0u32;
    while window.render_2d(&mut scene, &mut camera).await {
        // Advance the sprite-sheet animation a few times per second.
        if frame.is_multiple_of(20) {
            animated.set_sprite_frame(&sheet, frame / 20);
        }
        frame = frame.wrapping_add(1);
    }
}
