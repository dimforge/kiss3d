//! Wrapper for a wgpu buffer object.

use crate::context::Context;
use bytemuck::{Pod, Zeroable};

/// A vector of elements that can be loaded to the GPU, on the RAM, or both.
pub struct GPUVec<T: Pod + Zeroable> {
    dirty: bool,
    len: usize,
    usage: wgpu::BufferUsages,
    buffer: Option<wgpu::Buffer>,
    data: Option<Vec<T>>,
}

impl<T: Pod + Zeroable> GPUVec<T> {
    /// Creates a new `GPUVec` that is not yet uploaded to the GPU.
    pub fn new(data: Vec<T>, buf_type: BufferType, _alloc_type: AllocationType) -> GPUVec<T> {
        let usage = buf_type.to_wgpu();
        GPUVec {
            dirty: true,
            len: data.len(),
            usage,
            buffer: None,
            data: Some(data),
        }
    }

    /// Creates a new empty `GPUVec`.
    pub fn new_empty(buf_type: BufferType, _alloc_type: AllocationType) -> GPUVec<T> {
        let usage = buf_type.to_wgpu();
        GPUVec {
            dirty: false,
            len: 0,
            usage,
            buffer: None,
            data: Some(Vec::new()),
        }
    }

    /// Is this vector empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The length of this vector.
    #[inline]
    pub fn len(&self) -> usize {
        if self.dirty {
            match self.data {
                Some(ref d) => d.len(),
                None => panic!("This should never happen."),
            }
        } else {
            self.len
        }
    }

    /// Mutably accesses the vector if it is available on RAM.
    ///
    /// This method will mark this vector as `dirty`.
    #[inline]
    pub fn data_mut(&mut self) -> &mut Option<Vec<T>> {
        self.dirty = true;
        &mut self.data
    }

    /// Immutably accesses the vector if it is available on RAM.
    #[inline]
    pub fn data(&self) -> &Option<Vec<T>> {
        &self.data
    }

    /// Returns `true` if this vector is already uploaded to the GPU.
    #[inline]
    pub fn is_on_gpu(&self) -> bool {
        self.buffer.is_some()
    }

    /// Returns `true` if the cpu data and gpu data are out of sync.
    #[inline]
    pub fn dirty(&self) -> bool {
        self.dirty
    }

    /// Returns `true` if the cpu data and gpu data are out of sync.
    /// Alias for `dirty()` for backwards compatibility.
    #[inline]
    pub fn trash(&self) -> bool {
        self.dirty
    }

    /// Returns `true` if this vector is available on RAM.
    ///
    /// Note that a `GPUVec` may be both on RAM and on the GPU.
    #[inline]
    pub fn is_on_ram(&self) -> bool {
        self.data.is_some()
    }

    /// Returns the wgpu buffer if it exists.
    #[inline]
    pub fn buffer(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }

    /// Returns the buffer usage flags.
    #[inline]
    pub fn usage(&self) -> wgpu::BufferUsages {
        self.usage
    }

    /// Loads the vector from the RAM to the GPU.
    ///
    /// If the vector is not available on RAM or already loaded to the GPU, nothing will happen.
    #[inline]
    pub fn load_to_gpu(&mut self) {
        let ctxt = Context::get();

        if let Some(ref data) = self.data {
            if data.is_empty() {
                return;
            }

            let bytes = bytemuck::cast_slice(data);

            if !self.is_on_gpu() {
                // Create new buffer
                self.len = data.len();
                let buffer = ctxt.create_buffer_init(
                    Some("GPUVec buffer"),
                    bytes,
                    self.usage | wgpu::BufferUsages::COPY_DST,
                );
                self.buffer = Some(buffer);
            } else if self.dirty {
                // Update existing buffer
                self.len = data.len();

                if let Some(ref buffer) = self.buffer {
                    let buffer_size = buffer.size() as usize;
                    let data_size = bytes.len();

                    if data_size <= buffer_size {
                        // Buffer is big enough, just update
                        ctxt.write_buffer(buffer, 0, bytes);
                    } else {
                        // Need to recreate buffer
                        let new_buffer = ctxt.create_buffer_init(
                            Some("GPUVec buffer"),
                            bytes,
                            self.usage | wgpu::BufferUsages::COPY_DST,
                        );
                        self.buffer = Some(new_buffer);
                    }
                }
            }
        }

        self.dirty = false;
    }

    /// Ensures the buffer is on the GPU and returns a reference to it.
    ///
    /// Returns None if the data is empty.
    #[inline]
    pub fn ensure_on_gpu(&mut self) -> Option<&wgpu::Buffer> {
        self.load_to_gpu();
        self.buffer.as_ref()
    }

    /// Unloads this resource from the GPU.
    #[inline]
    pub fn unload_from_gpu(&mut self) {
        self.len = self.len();
        self.buffer = None;
        self.dirty = false;
    }

    /// Removes this resource from the RAM.
    ///
    /// This is useful to save memory for vectors required on the GPU only.
    #[inline]
    pub fn unload_from_ram(&mut self) {
        if self.dirty && self.is_on_gpu() {
            self.load_to_gpu();
        }

        self.data = None;
    }
}

impl<T: Clone + Pod + Zeroable> GPUVec<T> {
    /// Returns this vector as an owned vector if it is available on RAM.
    ///
    /// If it has been uploaded to the GPU, and unloaded from the RAM, call `load_to_ram` first to
    /// make the data accessible.
    #[inline]
    pub fn to_owned(&self) -> Option<Vec<T>> {
        self.data.as_ref().cloned()
    }
}

/// Type of gpu buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferType {
    /// A vertex buffer (bindable as vertex data).
    Array,
    /// An index buffer (bindable as index data).
    ElementArray,
}

impl BufferType {
    /// Converts to wgpu buffer usages.
    #[inline]
    pub fn to_wgpu(self) -> wgpu::BufferUsages {
        match self {
            BufferType::Array => wgpu::BufferUsages::VERTEX,
            BufferType::ElementArray => wgpu::BufferUsages::INDEX,
        }
    }
}

/// Allocation type of gpu buffers.
///
/// Note: In wgpu, allocation hints are handled differently than in OpenGL.
/// These are kept for API compatibility but may not have the same effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationType {
    /// Data uploaded once, used many times (immutable meshes).
    StaticDraw,
    /// Data modified frequently.
    DynamicDraw,
    /// Data for immediate use (lines, points, text).
    StreamDraw,
}
