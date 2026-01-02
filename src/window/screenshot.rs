//! Screenshot functionality.

use image::{imageops, ImageBuffer, Rgb};

use super::Window;

impl Window {
    /// Captures the current framebuffer as raw RGB pixel data.
    ///
    /// Reads all pixels currently displayed on the screen into a buffer.
    /// The buffer is automatically resized to fit the screen dimensions.
    /// Pixels are stored in RGB format (3 bytes per pixel), row by row from bottom to top.
    ///
    /// # Arguments
    /// * `out` - The output buffer. It will be resized to width × height × 3 bytes.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let window = Window::new("Example").await;
    /// let mut pixels = Vec::new();
    /// window.snap(&mut pixels);
    /// // pixels now contains RGB data
    /// # }
    /// ```
    pub fn snap(&self, out: &mut Vec<u8>) {
        let (width, height) = self.canvas.size();
        self.snap_rect(out, 0, 0, width as usize, height as usize)
    }

    /// Captures a rectangular region of the framebuffer as raw RGB pixel data.
    ///
    /// Reads a specific rectangular region of pixels from the screen.
    /// Pixels are stored in RGB format (3 bytes per pixel).
    ///
    /// # Arguments
    /// * `out` - The output buffer. It will be resized to width × height × 3 bytes.
    /// * `x` - The x-coordinate of the rectangle's bottom-left corner
    /// * `y` - The y-coordinate of the rectangle's bottom-left corner
    /// * `width` - The width of the rectangle in pixels
    /// * `height` - The height of the rectangle in pixels
    pub fn snap_rect(&self, out: &mut Vec<u8>, x: usize, y: usize, width: usize, height: usize) {
        self.canvas.read_pixels(out, x, y, width, height);
    }

    /// Captures the current framebuffer as an image.
    ///
    /// Returns an `ImageBuffer` containing the current screen content.
    /// The image is automatically flipped vertically to match the expected orientation
    /// (OpenGL's bottom-left origin is converted to top-left).
    ///
    /// # Returns
    /// An `ImageBuffer<Rgb<u8>, Vec<u8>>` containing the screen pixels
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let window = Window::new("Example").await;
    /// let image = window.snap_image();
    /// image.save("screenshot.png").unwrap();
    /// # }
    /// ```
    pub fn snap_image(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (width, height) = self.canvas.size();
        let mut buf = Vec::new();
        self.snap(&mut buf);
        let img_opt = ImageBuffer::from_vec(width, height, buf);
        let img = img_opt.expect("Buffer created from window was not big enough for image.");
        imageops::flip_vertical(&img)
    }
}
