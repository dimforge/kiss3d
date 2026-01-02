// This whole file is strongly inspired by: https://github.com/jeaye/q3/blob/master/src/client/ui/ttf/glyph.rs
// available under the BSD-3 licence.
// It has been modified to work with gl-rs, nalgebra, and rust-freetype

use glamx::Vec2;

/// A ttf glyph.
pub struct Glyph {
    #[doc(hidden)]
    pub tex: Vec2,
    #[doc(hidden)]
    pub advance: Vec2,
    #[doc(hidden)]
    pub dimensions: Vec2,
    #[doc(hidden)]
    pub offset: Vec2,
    #[doc(hidden)]
    pub buffer: Vec<u8>,
}

impl Glyph {
    /// Creates a new empty glyph.
    pub fn new(tex: Vec2, advance: Vec2, dimensions: Vec2, offset: Vec2, buffer: Vec<u8>) -> Glyph {
        Glyph {
            tex,
            advance,
            dimensions,
            offset,
            buffer,
        }
    }
}
