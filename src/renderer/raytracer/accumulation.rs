//! Progressive accumulation buffer for the path tracer.
//!
//! Radiance is accumulated as a running mean in a `Rgba32Float` storage buffer
//! (one `vec4<f32>` per pixel). A storage *buffer* is used rather than a
//! read-write storage *texture* because read-write storage textures are not
//! universally supported across wgpu backends, whereas read-write storage
//! buffers are core functionality.
//!
//! The radiance and the two denoiser GUIDE channels (first-hit albedo and
//! first-hit world normal) share **one** buffer, laid out as three contiguous
//! regions of `width * height` pixels each: `[radiance | albedo | normal]`.
//! Packing them into a single binding keeps the path-tracing compute stage within
//! WebGPU's limit of 8 storage buffers per stage (browsers expose exactly 8).

use crate::context::Context;

/// Holds the per-pixel radiance accumulator plus the denoiser guide channels, all
/// in one buffer (see the module docs for the layout).
pub struct Accumulation {
    /// `array<vec4<f32>>` of length `3 * width * height`: region 0 is the radiance
    /// running mean, region 1 the first-hit albedo, region 2 the first-hit normal.
    /// Each region is `width * height` pixels; region `k` starts at pixel
    /// `k * width * height`.
    pub buffer: wgpu::Buffer,
    /// Current width in pixels.
    pub width: u32,
    /// Current height in pixels.
    pub height: u32,
}

const PIXEL_SIZE: u64 = 16; // vec4<f32>
/// Number of regions packed into the buffer (radiance, albedo guide, normal guide).
pub const REGIONS: u64 = 3;

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
            count * REGIONS * PIXEL_SIZE,
            // COPY_SRC so the guide regions can be read back by a denoiser.
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
    }

    /// Resizes the buffer if the resolution changed. Returns `true` if it was
    /// recreated (in which case accumulation must restart).
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
