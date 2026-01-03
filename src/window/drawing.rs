//! Drawing methods for 2D and 3D primitives.

use std::sync::Arc;

use glamx::{Vec2, Vec3};

use crate::color::Color;
use crate::renderer::{Polyline2d, Polyline3d};
use crate::text::Font;

use super::Window;

impl Window {
    /// Draws a 3D line for the current frame.
    ///
    /// The line is only drawn during the next frame. To keep a line visible,
    /// call this method every frame from within your render loop.
    ///
    /// # Arguments
    /// * `a` - The starting point of the line in 3D space
    /// * `b` - The ending point of the line in 3D space
    /// * `color` - RGBA color (each component from 0.0 to 1.0)
    /// * `width` - Line width in pixels
    /// * `perspective` - Indicates if the rendered line size is affected by perspective (gets
    ///   smaller as camera gets further)
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # use kiss3d::color::RED;
    /// # use glamx::Vec3;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// let start = Vec3::new(0.0, 0.0, 0.0);
    /// let end = Vec3::new(1.0, 1.0, 1.0);
    /// window.draw_line(start, end, RED, 2.0, false);
    /// # }
    /// ```
    #[inline]
    pub fn draw_line(&mut self, a: Vec3, b: Vec3, color: Color, width: f32, perspective: bool) {
        self.polyline_renderer.draw_line(a, b, color, width, perspective);
    }

    /// Draws a 2D line for the current frame.
    ///
    /// The line is only drawn during the next frame. To keep a line visible,
    /// call this method every frame from within your render loop.
    ///
    /// # Arguments
    /// * `a` - The starting point of the line in 2D space
    /// * `b` - The ending point of the line in 2D space
    /// * `color` - RGBA color (each component from 0.0 to 1.0)
    /// * `width` - Line width in pixels
    #[inline]
    pub fn draw_line_2d(&mut self, a: Vec2, b: Vec2, color: Color, width: f32) {
        self.polyline_renderer_2d.draw_line(a, b, color, width);
    }

    /// Draws a 2D polyline (connected line segments) with configurable width.
    ///
    /// The polyline is only drawn during the next frame. To keep it visible,
    /// call this method every frame from within your render loop.
    ///
    /// Takes a reference to avoid allocations - segments are built immediately.
    ///
    /// # Arguments
    /// * `polyline` - The 2D polyline to draw
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::prelude::*;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// # let mut camera = OrbitCamera3d::default();
    /// # let mut scene = SceneNode3d::empty();
    /// let polyline = Polyline2d::new(vec![
    ///     Vec2::new(0.0, 0.0),
    ///     Vec2::new(100.0, 100.0),
    ///     Vec2::new(200.0, 0.0),
    /// ])
    /// .with_color(RED)
    /// .with_width(5.0);
    /// window.draw_polyline_2d(&polyline);
    /// # }
    /// ```
    #[inline]
    pub fn draw_polyline_2d(&mut self, polyline: &Polyline2d) {
        self.polyline_renderer_2d.draw_polyline(polyline);
    }

    /// Draws a 2D point for the current frame.
    ///
    /// The point is only drawn during the next frame. To keep a point visible,
    /// call this method every frame from within your render loop.
    ///
    /// # Arguments
    /// * `pt` - The position of the point in 2D space
    /// * `color` - RGBA color (each component from 0.0 to 1.0)
    /// * `size` - The point size in pixels
    #[inline]
    pub fn draw_point_2d(&mut self, pt: Vec2, color: Color, size: f32) {
        self.point_renderer_2d.draw_point(pt, color, size);
    }

    /// Draws a 3D point for the current frame.
    ///
    /// The point is only drawn during the next frame. To keep a point visible,
    /// call this method every frame from within your render loop.
    ///
    /// # Arguments
    /// * `pt` - The position of the point in 3D space
    /// * `color` - RGBA color (each component from 0.0 to 1.0)
    /// * `size` - The point size in pixels
    #[inline]
    pub fn draw_point(&mut self, pt: Vec3, color: Color, size: f32) {
        self.point_renderer.draw_point(pt, color, size);
    }

    /// Draws a polyline (connected line segments) with configurable width.
    ///
    /// The polyline is only drawn during the next frame. To keep it visible,
    /// call this method every frame from within your render loop.
    ///
    /// # Arguments
    /// * `polyline` - The polyline to draw
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::prelude::*;
    /// # use kiss3d::renderer::Polyline3d;
    /// # use glamx::Vec3;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// let polyline = Polyline3d::new(vec![
    ///     Vec3::new(0.0, 0.0, 0.0),
    ///     Vec3::new(1.0, 1.0, 0.0),
    ///     Vec3::new(2.0, 0.0, 0.0),
    /// ])
    /// .with_color(RED)
    /// .with_width(5.0);
    /// window.draw_polyline(&polyline);
    /// # }
    /// ```
    #[inline]
    pub fn draw_polyline(&mut self, polyline: &Polyline3d) {
        self.polyline_renderer.draw_polyline(polyline);
    }

    /// Draws text for the current frame.
    ///
    /// The text is only drawn during the next frame. To keep text visible,
    /// call this method every frame from within your render loop.
    ///
    /// # Arguments
    /// * `text` - The string to display
    /// * `pos` - The position in 2D screen coordinates
    /// * `scale` - The text scale factor
    /// * `font` - A reference to the font to use
    /// * `color` - RGBA color (each component from 0.0 to 1.0)
    #[inline]
    pub fn draw_text(
        &mut self,
        text: &str,
        pos: Vec2,
        scale: f32,
        font: &Arc<Font>,
        color: Color,
    ) {
        self.text_renderer.draw_text(text, pos, scale, font, color);
    }
}
