//! wgpu rendering context management.
//!
//! This module provides a global wgpu context that can be initialized and reset
//! across window recreations.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

// The global wgpu context singleton.
// We use RefCell<Option<>> instead of OnceLock to allow resetting the context
// when creating new windows (required for multi-window support).
thread_local! {
    static CONTEXT_SINGLETON: RefCell<Option<Context>> = const { RefCell::new(None) };
    // Track number of active windows to know when to reset the context
    static WINDOW_COUNT: Cell<usize> = const { Cell::new(0) };
}

/// The wgpu rendering context containing all GPU resources needed for rendering.
///
/// This struct is cloneable and thread-safe. It wraps wgpu resources in Arc
/// to allow sharing across the application.
#[derive(Clone)]
pub struct Context {
    /// The wgpu instance used for creating surfaces.
    pub instance: Arc<wgpu::Instance>,
    /// The wgpu device used for creating GPU resources.
    pub device: Arc<wgpu::Device>,
    /// The wgpu queue used for submitting commands.
    pub queue: Arc<wgpu::Queue>,
    /// The wgpu adapter information.
    pub adapter: Arc<wgpu::Adapter>,
    /// The preferred texture format for the surface.
    pub surface_format: wgpu::TextureFormat,
}

impl Context {
    /// Initializes or reinitializes the global wgpu context.
    ///
    /// This function is called when creating a window. For multi-window support,
    /// this will replace the existing context with a new one.
    ///
    /// # Arguments
    /// * `instance` - The wgpu instance
    /// * `device` - The wgpu device
    /// * `queue` - The wgpu queue
    /// * `adapter` - The wgpu adapter
    /// * `surface_format` - The preferred surface texture format
    pub fn init(
        instance: wgpu::Instance,
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter: wgpu::Adapter,
        surface_format: wgpu::TextureFormat,
    ) {
        CONTEXT_SINGLETON.with(|cell| {
            *cell.borrow_mut() = Some(Context {
                instance: Arc::new(instance),
                device: Arc::new(device),
                queue: Arc::new(queue),
                adapter: Arc::new(adapter),
                surface_format,
            });
        });
    }

    /// Gets a clone of the global wgpu context.
    ///
    /// # Panics
    /// Panics if the context has not been initialized via `init()`.
    pub fn get() -> Context {
        CONTEXT_SINGLETON.with(|cell| {
            cell.borrow()
                .as_ref()
                .expect("wgpu context not initialized. Call Context::init() first.")
                .clone()
        })
    }

    /// Checks if the context has been initialized.
    pub fn is_initialized() -> bool {
        CONTEXT_SINGLETON.with(|cell| cell.borrow().is_some())
    }

    /// Resets the global wgpu context, dropping all GPU resources.
    ///
    /// This should be called before thread-local storage destruction begins
    /// to avoid TLS access order issues with wgpu internals.
    ///
    /// After calling this, `is_initialized()` will return `false` and
    /// `get()` will panic until `init()` is called again.
    pub fn reset() {
        CONTEXT_SINGLETON.with(|cell| {
            // Explicitly destroy the device before dropping the context.
            // This ensures WebGPU resources are released immediately rather than
            // waiting for garbage collection, which is important for browsers
            // that limit the number of concurrent WebGPU contexts.
            if let Some(ctx) = cell.borrow().as_ref() {
                ctx.device.destroy();
            }
            *cell.borrow_mut() = None;
        });
    }

    /// Increments the window reference count.
    ///
    /// Called when a new window is created to track how many windows
    /// are using the context.
    pub fn increment_window_count() {
        WINDOW_COUNT.with(|count| {
            count.set(count.get() + 1);
        });
    }

    /// Decrements the window reference count and returns true if this was the last window.
    ///
    /// Called when a window is dropped. Returns true if all windows have been closed
    /// and it's safe to reset the context.
    pub fn decrement_window_count() -> bool {
        WINDOW_COUNT.with(|count| {
            let current = count.get();
            if current > 0 {
                count.set(current - 1);
                current == 1 // Was this the last window?
            } else {
                false
            }
        })
    }

    /// Returns the current number of active windows.
    pub fn window_count() -> usize {
        WINDOW_COUNT.with(|count| count.get())
    }

    /// Creates a new buffer on the GPU using a descriptor.
    ///
    /// # Arguments
    /// * `desc` - Buffer descriptor
    pub fn create_buffer(&self, desc: &wgpu::BufferDescriptor) -> wgpu::Buffer {
        self.device.create_buffer(desc)
    }

    /// Creates a new buffer on the GPU with specified parameters.
    ///
    /// # Arguments
    /// * `label` - Debug label for the buffer
    /// * `size` - Size of the buffer in bytes
    /// * `usage` - Buffer usage flags
    pub fn create_buffer_simple(
        &self,
        label: Option<&str>,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Creates a new buffer initialized with data.
    ///
    /// # Arguments
    /// * `label` - Debug label for the buffer
    /// * `contents` - The data to initialize the buffer with
    /// * `usage` - Buffer usage flags
    pub fn create_buffer_init(
        &self,
        label: Option<&str>,
        contents: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents,
                usage,
            })
    }

    /// Writes data to a buffer.
    ///
    /// # Arguments
    /// * `buffer` - The buffer to write to
    /// * `offset` - Byte offset into the buffer
    /// * `data` - The data to write
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.queue.write_buffer(buffer, offset, data);
    }

    /// Creates a new texture on the GPU.
    ///
    /// # Arguments
    /// * `desc` - Texture descriptor
    pub fn create_texture(&self, desc: &wgpu::TextureDescriptor) -> wgpu::Texture {
        self.device.create_texture(desc)
    }

    /// Creates a new sampler.
    ///
    /// # Arguments
    /// * `desc` - Sampler descriptor
    pub fn create_sampler(&self, desc: &wgpu::SamplerDescriptor) -> wgpu::Sampler {
        self.device.create_sampler(desc)
    }

    /// Creates a new bind group layout.
    ///
    /// # Arguments
    /// * `desc` - Bind group layout descriptor
    pub fn create_bind_group_layout(
        &self,
        desc: &wgpu::BindGroupLayoutDescriptor,
    ) -> wgpu::BindGroupLayout {
        self.device.create_bind_group_layout(desc)
    }

    /// Creates a new bind group.
    ///
    /// # Arguments
    /// * `desc` - Bind group descriptor
    pub fn create_bind_group(&self, desc: &wgpu::BindGroupDescriptor) -> wgpu::BindGroup {
        self.device.create_bind_group(desc)
    }

    /// Creates a new pipeline layout.
    ///
    /// # Arguments
    /// * `desc` - Pipeline layout descriptor
    pub fn create_pipeline_layout(
        &self,
        desc: &wgpu::PipelineLayoutDescriptor,
    ) -> wgpu::PipelineLayout {
        self.device.create_pipeline_layout(desc)
    }

    /// Creates a new render pipeline.
    ///
    /// # Arguments
    /// * `desc` - Render pipeline descriptor
    pub fn create_render_pipeline(
        &self,
        desc: &wgpu::RenderPipelineDescriptor,
    ) -> wgpu::RenderPipeline {
        self.device.create_render_pipeline(desc)
    }

    /// Creates a new shader module from WGSL source.
    ///
    /// # Arguments
    /// * `label` - Debug label for the shader
    /// * `source` - WGSL shader source code
    pub fn create_shader_module(&self, label: Option<&str>, source: &str) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
    }

    /// Creates a new command encoder.
    ///
    /// # Arguments
    /// * `label` - Debug label for the encoder
    pub fn create_command_encoder(&self, label: Option<&str>) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label })
    }

    /// Submits command buffers to the GPU queue.
    ///
    /// # Arguments
    /// * `command_buffers` - Iterator of command buffers to submit
    pub fn submit<I: IntoIterator<Item = wgpu::CommandBuffer>>(&self, command_buffers: I) {
        self.queue.submit(command_buffers);
    }

    /// Writes texture data to the GPU.
    ///
    /// # Arguments
    /// * `texture` - The texture to write to
    /// * `data` - The pixel data
    /// * `data_layout` - Layout of the pixel data
    /// * `size` - Size of the region to write
    pub fn write_texture(
        &self,
        texture: wgpu::TexelCopyTextureInfo,
        data: &[u8],
        data_layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        self.queue.write_texture(texture, data, data_layout, size);
    }

    /// Gets the depth texture format used for depth attachments.
    pub fn depth_format() -> wgpu::TextureFormat {
        wgpu::TextureFormat::Depth32Float
    }
}
