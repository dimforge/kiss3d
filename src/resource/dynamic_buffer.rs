//! Dynamic uniform buffer for batched GPU writes.
//!
//! This module provides a `DynamicUniformBuffer` that batches uniform data writes
//! to reduce the number of `write_buffer` calls per frame. Instead of writing each
//! object's uniforms individually, data is accumulated in CPU memory and flushed
//! to the GPU in a single operation.

use crate::context::Context;
use bytemuck::Pod;
use std::mem;

/// A dynamic uniform buffer that batches writes for better performance.
///
/// This buffer accumulates uniform data in CPU memory and flushes it to the GPU
/// in a single `write_buffer` call. Each entry is aligned to the GPU's minimum
/// uniform buffer offset alignment, allowing dynamic offsets in bind groups.
///
/// # Usage
///
/// ```ignore
/// let mut buffer = DynamicUniformBuffer::<MyUniforms>::new("my_uniforms");
///
/// // In render loop:
/// buffer.clear();
/// for object in objects {
///     let offset = buffer.push(&object.uniforms);
///     // Store offset for later use in render pass
/// }
/// buffer.flush(); // Single GPU write
/// // Now render using stored offsets
/// ```
pub struct DynamicUniformBuffer<T: Pod> {
    /// CPU-side data accumulator
    data: Vec<u8>,
    /// GPU buffer
    buffer: wgpu::Buffer,
    /// Current capacity in bytes
    capacity: u64,
    /// Alignment for each entry (from device limits)
    alignment: u64,
    /// Size of each entry (aligned)
    aligned_size: u64,
    /// Number of entries currently in the buffer
    count: usize,
    /// Label for debugging
    label: &'static str,
    /// Marker for the uniform type
    _marker: std::marker::PhantomData<T>,
}

impl<T: Pod> DynamicUniformBuffer<T> {
    /// Creates a new dynamic uniform buffer with a default initial capacity.
    ///
    /// # Arguments
    /// * `label` - Debug label for the GPU buffer
    pub fn new(label: &'static str) -> Self {
        Self::with_capacity(label, 256) // Start with space for 256 entries
    }

    /// Creates a new dynamic uniform buffer with the specified initial capacity.
    ///
    /// # Arguments
    /// * `label` - Debug label for the GPU buffer
    /// * `initial_capacity` - Initial number of entries to allocate space for
    pub fn with_capacity(label: &'static str, initial_capacity: usize) -> Self {
        let ctxt = Context::get();
        let alignment = ctxt.device.limits().min_uniform_buffer_offset_alignment as u64;

        // Calculate aligned size for each entry
        let unaligned_size = mem::size_of::<T>() as u64;
        let aligned_size = unaligned_size.div_ceil(alignment) * alignment;

        let capacity = aligned_size * initial_capacity as u64;

        let buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            data: Vec::with_capacity(capacity as usize),
            buffer,
            capacity,
            alignment,
            aligned_size,
            count: 0,
            label,
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns the alignment requirement for uniform buffer offsets.
    #[inline]
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Returns the aligned size of each entry.
    #[inline]
    pub fn aligned_size(&self) -> u64 {
        self.aligned_size
    }

    /// Returns the number of entries currently in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the buffer contains no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clears the buffer for the next frame.
    ///
    /// This resets the CPU-side data but doesn't deallocate memory.
    pub fn clear(&mut self) {
        self.data.clear();
        self.count = 0;
    }

    /// Pushes a uniform entry and returns its byte offset in the buffer.
    ///
    /// The offset can be used with `render_pass.set_bind_group()` to select
    /// this entry using dynamic offsets. Note: `flush()` must be called
    /// after all pushes and before rendering.
    ///
    /// # Arguments
    /// * `value` - The uniform data to push
    ///
    /// # Returns
    /// The byte offset of this entry in the buffer (aligned to device requirements)
    pub fn push(&mut self, value: &T) -> u32 {
        let offset = (self.count as u64 * self.aligned_size) as u32;

        // Write the actual data
        let bytes = bytemuck::bytes_of(value);
        self.data.extend_from_slice(bytes);

        // Pad to alignment
        let padding = self.aligned_size as usize - bytes.len();
        self.data.extend(std::iter::repeat_n(0u8, padding));

        self.count += 1;
        offset
    }

    /// Flushes accumulated data to the GPU buffer.
    ///
    /// This performs a single `write_buffer` call with all accumulated data.
    /// The buffer will be grown if necessary. Returns true if the buffer
    /// was reallocated (requiring bind group recreation).
    pub fn flush(&mut self) -> bool {
        if self.data.is_empty() {
            return false;
        }

        let required_size = self.data.len() as u64;

        // Grow buffer if needed
        let reallocated = if required_size > self.capacity {
            self.grow(required_size);
            true
        } else {
            false
        };

        let ctxt = Context::get();
        ctxt.write_buffer(&self.buffer, 0, &self.data);

        reallocated
    }

    /// Grows the GPU buffer to accommodate the required size.
    fn grow(&mut self, required_size: u64) {
        let ctxt = Context::get();

        // Double capacity until it's enough
        let mut new_capacity = self.capacity;
        while new_capacity < required_size {
            new_capacity *= 2;
        }

        self.buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: new_capacity,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.capacity = new_capacity;
    }

    /// Returns a reference to the underlying GPU buffer.
    #[inline]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Returns the current capacity of the buffer in bytes.
    #[inline]
    pub fn capacity(&self) -> u64 {
        self.capacity
    }
}
