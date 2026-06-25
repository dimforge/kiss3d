//! Sprite helpers: sprite-sheet frame layout, 9-slice borders, and the mesh
//! builders behind [`SceneNode2d::sprite`](crate::scene::SceneNode2d::sprite) and
//! [`SceneNode2d::nine_slice`](crate::scene::SceneNode2d::nine_slice).

use crate::resource::vertex_index::VertexIndex;
use crate::resource::GpuMesh2d;
use glamx::Vec2;

/// A regular grid of equally-sized frames packed into one texture (a sprite sheet /
/// texture atlas), used to animate or index a sprite by frame number.
///
/// Frames are numbered left-to-right, top-to-bottom starting at 0. Pair it with
/// [`SceneNode2d::set_sprite_frame`](crate::scene::SceneNode2d::set_sprite_frame) to
/// show one frame, or step `index` over time for flip-book animation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SpriteSheet {
    /// Number of frame columns across the texture.
    pub columns: u32,
    /// Number of frame rows down the texture.
    pub rows: u32,
}

impl SpriteSheet {
    /// A sheet with `columns` × `rows` equally-sized frames.
    pub fn new(columns: u32, rows: u32) -> Self {
        assert!(columns > 0 && rows > 0, "sprite sheet needs a non-empty grid");
        SpriteSheet { columns, rows }
    }

    /// The total number of frames in the sheet.
    pub fn len(&self) -> u32 {
        self.columns * self.rows
    }

    /// Whether the sheet has no frames (never true for a sheet built with [`Self::new`]).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `(min, max)` UV rectangle of frame `index` (wrapping past the last frame),
    /// with UV origin at the top-left of the texture.
    pub fn frame_uv(&self, index: u32) -> (Vec2, Vec2) {
        let index = index % self.len();
        let col = index % self.columns;
        let row = index / self.columns;
        let cell = Vec2::new(1.0 / self.columns as f32, 1.0 / self.rows as f32);
        let min = Vec2::new(col as f32, row as f32) * cell;
        (min, min + cell)
    }
}

/// Per-edge insets (left, right, top, bottom), reused for both the world-space border
/// width and the texture-space (UV) border of a [9-slice](SceneNode2d::nine_slice) sprite.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Border {
    /// Left inset.
    pub left: f32,
    /// Right inset.
    pub right: f32,
    /// Top inset.
    pub top: f32,
    /// Bottom inset.
    pub bottom: f32,
}

impl Border {
    /// A border with the same inset on all four edges.
    pub fn uniform(inset: f32) -> Self {
        Border {
            left: inset,
            right: inset,
            top: inset,
            bottom: inset,
        }
    }

    /// A border with independent horizontal and vertical insets.
    pub fn symmetric(horizontal: f32, vertical: f32) -> Self {
        Border {
            left: horizontal,
            right: horizontal,
            top: vertical,
            bottom: vertical,
        }
    }
}

/// Builds the 16-vertex / 9-quad mesh of a 9-slice sprite of total size `size`
/// (centered on the origin), whose corner cells keep `world` size while the center
/// and edge cells stretch, sampling the texture split by the `uv` border.
pub(crate) fn nine_slice_mesh(size: Vec2, world: Border, uv: Border) -> GpuMesh2d {
    let hx = size.x * 0.5;
    let hy = size.y * 0.5;

    // Column x positions (left→right) and row y positions (top→bottom, world y-up).
    let xs = [-hx, -hx + world.left, hx - world.right, hx];
    let ys = [hy, hy - world.top, -hy + world.bottom, -hy];
    // Matching UV columns/rows, UV origin at the texture's top-left.
    let us = [0.0, uv.left, 1.0 - uv.right, 1.0];
    let vs = [0.0, uv.top, 1.0 - uv.bottom, 1.0];

    let mut coords = Vec::with_capacity(16);
    let mut uvs = Vec::with_capacity(16);
    for j in 0..4 {
        for i in 0..4 {
            coords.push(Vec2::new(xs[i], ys[j]));
            uvs.push(Vec2::new(us[i], vs[j]));
        }
    }

    // Two triangles per cell. Backface culling is off for 2D, so winding is free.
    let mut faces = Vec::with_capacity(18);
    for j in 0..3u32 {
        for i in 0..3u32 {
            let tl = (j * 4 + i) as VertexIndex;
            let tr = tl + 1;
            let bl = tl + 4;
            let br = bl + 1;
            faces.push([tl, bl, br]);
            faces.push([tl, br, tr]);
        }
    }

    GpuMesh2d::new(coords, faces, Some(uvs), false)
}
