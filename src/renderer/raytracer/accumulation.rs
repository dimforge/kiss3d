//! Progressive accumulation buffer for the path tracer.
//!
//! Radiance is accumulated as a running mean in a `Rgba32Float` storage buffer
//! (one `vec4<f32>` per pixel). A storage *buffer* is used rather than a
//! read-write storage *texture* because read-write storage textures are not
//! universally supported across wgpu backends, whereas read-write storage
//! buffers are core functionality.

use crate::context::Context;

/// Holds the per-pixel radiance accumulator and its current resolution.
pub struct Accumulation {
    /// `array<vec4<f32>>` of length `width * height`.
    pub buffer: wgpu::Buffer,
    /// Current width in pixels.
    pub width: u32,
    /// Current height in pixels.
    pub height: u32,
}

const PIXEL_SIZE: u64 = 16; // vec4<f32>

impl Accumulation {
    /// Creates a new accumulation buffer sized for `width * height` pixels.
    pub fn new(width: u32, height: u32) -> Accumulation {
        Accumulation {
            buffer: Self::make_buffer(width, height),
            width,
            height,
        }
    }

    fn make_buffer(width: u32, height: u32) -> wgpu::Buffer {
        let ctxt = Context::get();
        let count = (width.max(1) as u64) * (height.max(1) as u64);
        ctxt.create_buffer_simple(
            Some("rt_accumulation"),
            count * PIXEL_SIZE,
            wgpu::BufferUsages::STORAGE,
        )
    }

    /// Resizes the buffer if the resolution changed. Returns `true` if the buffer
    /// was recreated (in which case accumulation must restart).
    pub fn ensure(&mut self, width: u32, height: u32) -> bool {
        if width == self.width && height == self.height {
            return false;
        }
        self.buffer = Self::make_buffer(width, height);
        self.width = width;
        self.height = height;
        true
    }
}
