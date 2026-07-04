use kiss3d::prelude::*;
use std::path::Path;

// Demonstrates the `Tilemap`: a grid of tiles drawn from one texture atlas as a
// single mesh / draw call. The kitten texture stands in for a 2x2 tile atlas (its
// four quadrants), and an animated wave reshuffles the tile indices each frame.
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: 2D tilemap").await;
    let mut camera = PanZoomCamera2d::new(Vec2::ZERO, 1.0);
    let mut scene = SceneNode2d::empty();

    let columns = 16;
    let rows = 12;
    let sheet = SpriteSheet::new(2, 2); // 4 tiles
    let mut tilemap = Tilemap::new(columns, rows, Vec2::new(44.0, 44.0), sheet);

    // The atlas texture, assigned to the tilemap's node.
    tilemap
        .node()
        .set_texture_from_file(Path::new("./examples/media/kitten.png"), "kitten");
    scene.add_child(tilemap.node());

    let mut frame = 0u32;
    while window.render_2d(&mut scene, &mut camera).await {
        // A few times per second, rewrite the whole grid with a moving diagonal
        // wave selecting one of the four atlas tiles per cell.
        if frame.is_multiple_of(8) {
            let phase = frame / 8;
            let mut indices = Vec::with_capacity((columns * rows) as usize);
            for row in 0..rows {
                for col in 0..columns {
                    indices.push((col + row + phase) % 4);
                }
            }
            tilemap.fill(&indices);
        }
        frame = frame.wrapping_add(1);
    }
}
