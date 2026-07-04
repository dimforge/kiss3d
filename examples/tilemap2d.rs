use kiss3d::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

// Demonstrates the `Tilemap`: a grid of tiles drawn from one texture atlas as a
// single mesh / draw call. The tiles come from Kenney's CC0 "Tiny Town" atlas
// (12x11 tiles, see examples/media/credits.txt): a grass meadow with a rounded
// dirt patch, and an animated diagonal "wind" of flowers sweeping across it —
// rewritten into the tilemap every few frames to show dynamic fills.

// Terrain tile indices into the 12x11 atlas (index = atlas_row * 12 + atlas_col).
const GRASS: u32 = 0; // plain grass
const GRASS_TUFT: u32 = 1; // grass with a darker tuft
const GRASS_FLOWERS: u32 = 2; // grass speckled with flowers
                              // A dirt patch, drawn as a 9-slice of full-bleed tiles that blend dirt into the
                              // surrounding grass (corners/edges are rounded transitions; the center is solid).
const DIRT_TL: u32 = 12;
const DIRT_T: u32 = 13;
const DIRT_TR: u32 = 14;
const DIRT_L: u32 = 24;
const DIRT_C: u32 = 25;
const DIRT_R: u32 = 26;
const DIRT_BL: u32 = 36;
const DIRT_B: u32 = 37;
const DIRT_BR: u32 = 38;

// Inclusive tile bounds of the dirt patch within the map.
const PATCH: (u32, u32, u32, u32) = (5, 9, 4, 7); // (col0, col1, row0, row1)

// Pick the right 9-slice dirt tile for a cell known to be inside the patch.
fn dirt_tile(col: u32, row: u32) -> u32 {
    let (c0, c1, r0, r1) = PATCH;
    let (left, right) = (col == c0, col == c1);
    let (top, bottom) = (row == r0, row == r1);
    match (top, bottom, left, right) {
        (true, _, true, _) => DIRT_TL,
        (true, _, _, true) => DIRT_TR,
        (_, true, true, _) => DIRT_BL,
        (_, true, _, true) => DIRT_BR,
        (true, _, _, _) => DIRT_T,
        (_, true, _, _) => DIRT_B,
        (_, _, true, _) => DIRT_L,
        (_, _, _, true) => DIRT_R,
        _ => DIRT_C,
    }
}

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: 2D tilemap").await;
    let mut camera = PanZoomCamera2d::new(Vec2::ZERO, 1.0);
    let mut scene = SceneNode2d::empty();

    let columns = 16;
    let rows = 12;
    let sheet = SpriteSheet::new(12, 11); // Kenney "Tiny Town" atlas: 12x11 tiles
    let mut tilemap = Tilemap::new(columns, rows, Vec2::new(48.0, 48.0), sheet);

    // The atlas texture for the tilemap's node. Embed on wasm (no filesystem); read
    // from disk on native.
    #[cfg(not(target_arch = "wasm32"))]
    tilemap
        .node()
        .set_texture_from_file_pixelated(Path::new("./examples/media/tiny_town.png"), "tiny_town");
    #[cfg(target_arch = "wasm32")]
    tilemap
        .node()
        .set_texture_from_memory_pixelated(include_bytes!("./media/tiny_town.png"), "tiny_town");
    scene.add_child(tilemap.node());

    let mut frame = 0u32;
    while window.render_2d(&mut scene, &mut camera).await {
        // A few times per second, rebuild the grid: a grass meadow with a dirt
        // patch, overlaid by a diagonal band of flowers that sweeps across.
        if frame.is_multiple_of(6) {
            let phase = frame / 6;
            let (c0, c1, r0, r1) = PATCH;
            let mut indices = Vec::with_capacity((columns * rows) as usize);
            for row in 0..rows {
                for col in 0..columns {
                    let tile = if col >= c0 && col <= c1 && row >= r0 && row <= r1 {
                        dirt_tile(col, row)
                    } else if (col + row + phase) % 12 < 2 {
                        // The moving flower band.
                        GRASS_FLOWERS
                    } else if (col.wrapping_mul(7) ^ row.wrapping_mul(13)) % 4 == 0 {
                        // A little static variation so the meadow isn't uniform.
                        GRASS_TUFT
                    } else {
                        GRASS
                    };
                    indices.push(tile);
                }
            }
            tilemap.fill(&indices);
        }
        frame = frame.wrapping_add(1);
    }
}
