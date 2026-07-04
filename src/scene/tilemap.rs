//! A [`Tilemap`]: a grid of tiles drawn from a single texture atlas as one mesh.
//!
//! A tilemap bakes one textured quad per non-empty tile into a single
//! [`SceneNode2d`] mesh, so a whole map is one draw call sharing the standard 2D
//! material (and thus blend modes, the camera, etc.). Tiles index a [`SpriteSheet`]
//! atlas; updating a tile rebuilds the mesh.

use crate::resource::vertex_index::VertexIndex;
use crate::resource::GpuMesh2d;
use crate::scene::sprite::SpriteSheet;
use crate::scene::SceneNode2d;
use glamx::Vec2;
use std::cell::RefCell;
use std::rc::Rc;

/// A grid of tiles rendered as a single atlas-textured mesh.
///
/// Build one with [`Tilemap::new`], assign the atlas texture to its
/// [`node`](Tilemap::node) (`set_texture_*`), add that node to the scene, and edit
/// the map with [`set_tile`](Tilemap::set_tile) / [`fill`](Tilemap::fill).
pub struct Tilemap {
    node: SceneNode2d,
    columns: u32,
    rows: u32,
    tile_size: Vec2,
    sheet: SpriteSheet,
    tiles: Vec<u32>,
}

impl Tilemap {
    /// Tile value meaning "no tile here" (the cell is left empty / transparent).
    pub const EMPTY: u32 = u32::MAX;

    /// Creates a `columns` × `rows` tilemap of `tile_size`-world-unit tiles, indexing
    /// frames of `sheet`. The map starts entirely empty and is centered on the origin.
    pub fn new(columns: u32, rows: u32, tile_size: Vec2, sheet: SpriteSheet) -> Tilemap {
        let tiles = vec![Self::EMPTY; (columns * rows) as usize];
        let mesh = build_mesh(columns, rows, tile_size, &sheet, &tiles);
        let node = SceneNode2d::mesh(Rc::new(RefCell::new(mesh)), Vec2::ONE);
        Tilemap {
            node,
            columns,
            rows,
            tile_size,
            sheet,
            tiles,
        }
    }

    /// The scene node holding the tilemap mesh. Clone it to add it to the scene or to
    /// set its atlas texture, e.g. `tilemap.node().set_texture_with_name("atlas")`.
    pub fn node(&self) -> SceneNode2d {
        self.node.clone()
    }

    /// The map dimensions in tiles, `(columns, rows)`.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.columns, self.rows)
    }

    /// The frame index at `(col, row)`, or [`Self::EMPTY`] if out of range / empty.
    pub fn tile(&self, col: u32, row: u32) -> u32 {
        if col < self.columns && row < self.rows {
            self.tiles[(row * self.columns + col) as usize]
        } else {
            Self::EMPTY
        }
    }

    /// Sets the frame index at `(col, row)` (use [`Self::EMPTY`] to clear it) and
    /// rebuilds the mesh. Out-of-range coordinates are ignored.
    pub fn set_tile(&mut self, col: u32, row: u32, index: u32) {
        if col >= self.columns || row >= self.rows {
            return;
        }
        self.tiles[(row * self.columns + col) as usize] = index;
        self.rebuild();
    }

    /// Replaces every tile from a row-major slice (extra cells are left empty) and
    /// rebuilds the mesh once.
    pub fn fill(&mut self, indices: &[u32]) {
        for (dst, &src) in self.tiles.iter_mut().zip(indices.iter()) {
            *dst = src;
        }
        self.rebuild();
    }

    /// Rebuilds the mesh from the current tile grid.
    fn rebuild(&mut self) {
        let (coords, faces, uvs) = build_mesh_data(
            self.columns,
            self.rows,
            self.tile_size,
            &self.sheet,
            &self.tiles,
        );
        // Replace the mesh contents in place so the node keeps its identity.
        self.node.modify_vertices(&mut |v| {
            v.clear();
            v.extend_from_slice(&coords);
        });
        self.node.modify_faces(&mut |f| {
            f.clear();
            f.extend_from_slice(&faces);
        });
        self.node.modify_uvs(&mut |u| {
            u.clear();
            u.extend_from_slice(&uvs);
        });
    }
}

/// Builds the `(coords, faces, uvs)` of the tilemap mesh: one quad per non-empty tile.
fn build_mesh_data(
    columns: u32,
    rows: u32,
    tile_size: Vec2,
    sheet: &SpriteSheet,
    tiles: &[u32],
) -> (Vec<Vec2>, Vec<[VertexIndex; 3]>, Vec<Vec2>) {
    let mut coords = Vec::new();
    let mut uvs = Vec::new();
    let mut faces = Vec::new();

    let half = Vec2::new(columns as f32, rows as f32) * tile_size * 0.5;

    for row in 0..rows {
        for col in 0..columns {
            let index = tiles[(row * columns + col) as usize];
            if index == Tilemap::EMPTY {
                continue;
            }

            // Tile corners in world space (row 0 at the top, world y up).
            let x0 = col as f32 * tile_size.x - half.x;
            let x1 = x0 + tile_size.x;
            let y1 = half.y - row as f32 * tile_size.y;
            let y0 = y1 - tile_size.y;

            let (uv_min, uv_max) = sheet.frame_uv(index);

            let base = coords.len() as VertexIndex;
            // top-left, top-right, bottom-right, bottom-left
            coords.push(Vec2::new(x0, y1));
            coords.push(Vec2::new(x1, y1));
            coords.push(Vec2::new(x1, y0));
            coords.push(Vec2::new(x0, y0));
            uvs.push(Vec2::new(uv_min.x, uv_min.y));
            uvs.push(Vec2::new(uv_max.x, uv_min.y));
            uvs.push(Vec2::new(uv_max.x, uv_max.y));
            uvs.push(Vec2::new(uv_min.x, uv_max.y));

            faces.push([base, base + 1, base + 2]);
            faces.push([base, base + 2, base + 3]);
        }
    }

    // A mesh must never be empty (zero vertices breaks buffer creation); emit a
    // single degenerate triangle when the whole map is empty.
    if coords.is_empty() {
        coords.push(Vec2::ZERO);
        uvs.push(Vec2::ZERO);
        faces.push([0, 0, 0]);
    }

    (coords, faces, uvs)
}

fn build_mesh(
    columns: u32,
    rows: u32,
    tile_size: Vec2,
    sheet: &SpriteSheet,
    tiles: &[u32],
) -> GpuMesh2d {
    let (coords, faces, uvs) = build_mesh_data(columns, rows, tile_size, sheet, tiles);
    GpuMesh2d::new(coords, faces, Some(uvs), true)
}
