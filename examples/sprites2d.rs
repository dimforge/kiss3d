use kiss3d::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

// Demonstrates sprites with Kenney's CC0 pixel art (see examples/media/credits.txt):
// a grid of characters, each animated by alternating its two sprite-sheet frames
// (`SpriteSheet` + `set_sprite_frame` over a 9x3 sheet), plus a 9-slice UI panel on
// its own row whose border keeps its size while the center stretches. Textures load
// with nearest-neighbor filtering so the pixels stay crisp and neighboring sheet
// cells don't bleed into each other.
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: 2D sprites").await;
    let mut camera = PanZoomCamera2d::new(Vec2::ZERO, 1.5);
    let mut scene = SceneNode2d::empty();

    let sheet = SpriteSheet::new(9, 3);

    // Each character is a pair of frames on the sheet; alternating them reads as a
    // little walk cycle.
    let frame_pairs: [[u32; 2]; 8] = [
        [0, 1],   // green astronaut
        [2, 3],   // blue astronaut
        [4, 5],   // pink astronaut
        [6, 7],   // yellow astronaut
        [9, 10],  // tan astronaut
        [11, 12], // boxy critter
        [13, 14], // fish
        [15, 16], // rocket
    ];

    // Lay the characters out in a 4-column grid, centered horizontally, above the
    // UI-panel row.
    const COLS: usize = 4;
    let cell = 120.0;
    let sprite = 92.0;
    // Two character rows plus the panel row, centered vertically around the origin
    // (rows land at +cell, 0, -cell).
    let top = cell;

    let mut characters = Vec::new();
    for (i, &pair) in frame_pairs.iter().enumerate() {
        let col = (i % COLS) as f32;
        let row = (i / COLS) as f32;
        let x = (col - (COLS as f32 - 1.0) * 0.5) * cell;
        let y = top - row * cell;

        let mut node = scene.add_sprite(sprite, sprite);
        node.translate(Vec2::new(x, y));
        // The first sprite registers the shared character sheet; the rest reuse it.
        // Embed on wasm (no filesystem); read from disk on native.
        if i == 0 {
            #[cfg(not(target_arch = "wasm32"))]
            node.set_texture_from_file_pixelated(
                Path::new("./examples/media/characters.png"),
                "characters",
            );
            #[cfg(target_arch = "wasm32")]
            node.set_texture_from_memory_pixelated(
                include_bytes!("./media/characters.png"),
                "characters",
            );
        } else {
            node.set_texture_with_name("characters");
        }
        node.set_sprite_frame(&sheet, pair[0]);
        characters.push((node, pair));
    }

    // A 9-slice UI panel on its own row below the character grid: the rounded border
    // keeps its size while the center stretches. The panel is a separate texture.
    let panel_y = top - 2.0 * cell;
    let mut panel = scene.add_nine_slice(
        Vec2::new(500.0, 80.0),
        Border::uniform(24.0),
        Border::uniform(0.22),
    );
    panel.translate(Vec2::new(0.0, panel_y));
    #[cfg(not(target_arch = "wasm32"))]
    panel.set_texture_from_file_pixelated(Path::new("./examples/media/ui_panel.png"), "ui_panel");
    #[cfg(target_arch = "wasm32")]
    panel.set_texture_from_memory_pixelated(include_bytes!("./media/ui_panel.png"), "ui_panel");

    let mut frame = 0u32;
    while window.render_2d(&mut scene, &mut camera).await {
        // A few times per second, flip every character between its two frames.
        if frame.is_multiple_of(12) {
            let phase = ((frame / 12) % 2) as usize;
            for (node, pair) in characters.iter_mut() {
                node.set_sprite_frame(&sheet, pair[phase]);
            }
        }
        frame = frame.wrapping_add(1);
    }
}
