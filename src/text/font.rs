use std::borrow::Borrow;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use rusttype;

/// A TrueType font for text rendering.
///
/// `Font` wraps a `rusttype::Font` and can be loaded from a file or memory.
/// Use with [`Window::draw_text()`](crate::window::Window::draw_text) to render text.
pub struct Font {
    font: rusttype::Font<'static>,
}

impl Font {
    /// Loads a TrueType font from a file.
    ///
    /// # Arguments
    /// * `path` - Path to the .ttf font file
    ///
    /// # Returns
    /// `Some(Font)` if the font was successfully loaded, `None` otherwise
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::text::Font;
    /// # use std::path::Path;
    /// let font = Font::new(Path::new("assets/MyFont.ttf")).unwrap();
    /// ```
    pub fn new(path: &Path) -> Option<Font> {
        let mut memory = Vec::new();
        let mut file = File::open(path).unwrap();
        let _ = file.read_to_end(&mut memory).unwrap();
        Font::from_bytes(&memory)
    }

    /// Loads a TrueType font from a byte slice in memory.
    ///
    /// # Arguments
    /// * `memory` - Byte slice containing the .ttf font data
    ///
    /// # Returns
    /// `Some(Font)` if the font was successfully parsed, `None` otherwise
    ///
    /// # Example
    /// ```ignore
    /// # use kiss3d::text::Font;
    /// const FONT_DATA: &[u8] = include_bytes!("MyFont.ttf");
    /// let font = Font::from_bytes(FONT_DATA).unwrap();
    /// ```
    pub fn from_bytes(memory: &[u8]) -> Option<Font> {
        let font = rusttype::Font::try_from_vec(memory.to_vec()).unwrap();
        Some(Font { font })
    }

    /// Returns the default built-in font.
    ///
    /// This provides a singleton instance of the WorkSans-Regular font that's
    /// embedded in kiss3d. The font is cached, so subsequent calls return the
    /// same instance.
    ///
    /// # Returns
    /// An `Arc<Font>` reference to the default font
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::prelude::*;
    /// # use glamx::Vec2;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// let font = Font::default();
    /// window.draw_text("Hello", Vec2::new(10.0, 10.0), 60.0, &font, WHITE);
    /// # }
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Arc<Font> {
        const DATA: &[u8] = include_bytes!("WorkSans-Regular.ttf");
        static DEFAULT_FONT_SINGLETON: OnceLock<Arc<Font>> = OnceLock::new();

        DEFAULT_FONT_SINGLETON
            .get_or_init(|| {
                Arc::new(Font::from_bytes(DATA).expect("Default font creation failed."))
            })
            .clone()
    }

    /// Returns a reference to the underlying rusttype font.
    ///
    /// This provides access to rusttype's advanced font manipulation features.
    ///
    /// # Returns
    /// A reference to the `rusttype::Font`
    #[inline]
    pub fn font(&self) -> &rusttype::Font<'static> {
        &self.font
    }

    /// Returns a unique identifier for the font instance.
    ///
    /// This is used internally for font caching and management.
    ///
    /// # Arguments
    /// * `font` - The font to get the UID for
    ///
    /// # Returns
    /// A unique identifier (memory address) for the font
    #[inline]
    pub fn uid(font: &Arc<Font>) -> usize {
        (*font).borrow() as *const Font as usize
    }
}
