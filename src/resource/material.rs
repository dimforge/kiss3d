//! Trait implemented by materials.

use crate::camera::Camera2d;
use crate::camera::Camera3d;
use crate::light::LightCollection;
use crate::resource::{GpuMesh2d, GpuMesh3d};
use crate::scene::{InstancesBuffer2d, InstancesBuffer3d, ObjectData2d, ObjectData3d};
use glamx::{Pose2, Pose3, Vec2, Vec3};
use std::any::Any;

/// Context passed to materials during rendering.
///
/// This contains metadata about the render target. The actual render pass
/// is passed separately to enable batching multiple draw calls.
pub struct RenderContext {
    /// The surface format.
    pub surface_format: wgpu::TextureFormat,
    /// The sample count for MSAA.
    pub sample_count: u32,
    /// The viewport width in pixels.
    pub viewport_width: u32,
    /// The viewport height in pixels.
    pub viewport_height: u32,
}

/// Per-object GPU data for a material.
///
/// This trait is implemented by material-specific structs that hold
/// per-object GPU resources (uniform buffers, bind groups, etc.).
/// Each object in the scene has its own GpuData instance.
pub trait GpuData: Any {
    /// Returns self as Any for downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Returns self as mutable Any for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Trait implemented by materials.
///
/// Materials define how objects are rendered. The material itself holds
/// shared resources (pipeline, bind group layouts), while per-object
/// resources are stored in GpuData instances.
///
/// ## Two-Phase Rendering
///
/// For efficient batched uniform uploads, rendering uses two phases:
/// 1. **Prepare phase**: `prepare()` is called for each object to collect uniform data
/// 2. **Flush phase**: `flush()` uploads all collected data to GPU in one batch
/// 3. **Render phase**: `render()` is called for each object to issue draw calls
pub trait Material3d {
    /// Creates per-object GPU data for this material.
    ///
    /// This is called once when an object is created. The returned GpuData
    /// holds uniform buffers and other per-object GPU resources.
    fn create_gpu_data(&self) -> Box<dyn GpuData>;

    /// Called at the start of each frame before any objects are prepared.
    ///
    /// This allows materials to reset per-frame state, such as clearing
    /// dynamic uniform buffers. The default implementation does nothing.
    fn begin_frame(&mut self) {}

    /// Prepares uniform data for an object (phase 1).
    ///
    /// This method collects uniform data in CPU memory. The data will be
    /// uploaded to GPU when `flush()` is called. Returns an offset that
    /// should be stored in gpu_data for use during rendering.
    ///
    /// The default implementation does nothing (for materials that don't
    /// use batched uniforms).
    fn prepare(
        &mut self,
        _pass: usize,
        _transform: Pose3,
        _scale: Vec3,
        _camera: &mut dyn Camera3d,
        _lights: &LightCollection,
        _data: &ObjectData3d,
        _gpu_data: &mut dyn GpuData,
        _viewport_width: u32,
        _viewport_height: u32,
    ) {
    }

    /// Flushes collected uniform data to GPU (phase 2).
    ///
    /// This uploads all data collected during `prepare()` calls to the GPU
    /// in a single batch. The default implementation does nothing.
    fn flush(&mut self) {}

    /// Renders an object using this material (phase 3).
    ///
    /// # Arguments
    /// * `pass` - The render pass index (for multi-pass rendering)
    /// * `transform` - The object's world transform
    /// * `scale` - The object's scale
    /// * `camera` - The camera
    /// * `lights` - The collected scene lights
    /// * `data` - Object rendering properties (color, texture, etc.)
    /// * `mesh` - The object's mesh
    /// * `instances` - Instance data for instanced rendering
    /// * `gpu_data` - Per-object GPU resources created by `create_gpu_data`
    /// * `render_pass` - The active render pass to draw into
    /// * `context` - Render context with viewport info
    fn render(
        &mut self,
        pass: usize,
        transform: Pose3,
        scale: Vec3,
        camera: &mut dyn Camera3d,
        lights: &LightCollection,
        data: &ObjectData3d,
        mesh: &mut GpuMesh3d,
        instances: &mut InstancesBuffer3d,
        gpu_data: &mut dyn GpuData,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext,
    );
}

/// Context passed to 2D materials during rendering.
///
/// This contains metadata about the render target. The actual render pass
/// is passed separately to enable batching multiple draw calls.
pub struct RenderContext2d {
    /// The surface format.
    pub surface_format: wgpu::TextureFormat,
    /// The sample count for MSAA.
    pub sample_count: u32,
    /// The viewport width in pixels.
    pub viewport_width: u32,
    /// The viewport height in pixels.
    pub viewport_height: u32,
}

/// Context for 2D renderers that need to create their own render passes.
///
/// This is used by legacy renderers (text, points, polylines) that haven't
/// been updated to use the two-phase rendering approach.
pub struct RenderContext2dEncoder<'a> {
    /// The command encoder for this frame.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The color attachment view.
    pub color_view: &'a wgpu::TextureView,
    /// The surface format.
    pub surface_format: wgpu::TextureFormat,
    /// The sample count for MSAA.
    pub sample_count: u32,
    /// The viewport width in pixels.
    pub viewport_width: u32,
    /// The viewport height in pixels.
    pub viewport_height: u32,
}

/// A material for 2D objects.
///
/// ## Two-Phase Rendering
///
/// For efficient batched uniform uploads, rendering uses two phases:
/// 1. **Prepare phase**: `prepare()` is called for each object to collect uniform data
/// 2. **Flush phase**: `flush()` uploads all collected data to GPU in one batch
/// 3. **Render phase**: `render()` is called for each object to issue draw calls
pub trait Material2d {
    /// Creates per-object GPU data for this material.
    fn create_gpu_data(&self) -> Box<dyn GpuData>;

    /// Called at the start of each frame before any objects are prepared.
    ///
    /// This allows materials to reset per-frame state, such as clearing
    /// dynamic uniform buffers. The default implementation does nothing.
    fn begin_frame(&mut self) {}

    /// Prepares uniform data for an object (phase 1).
    ///
    /// This method collects uniform data in CPU memory. The data will be
    /// uploaded to GPU when `flush()` is called.
    ///
    /// The default implementation does nothing (for materials that don't
    /// use batched uniforms).
    fn prepare(
        &mut self,
        _transform: Pose2,
        _scale: Vec2,
        _camera: &mut dyn Camera2d,
        _data: &ObjectData2d,
        _mesh: &mut GpuMesh2d,
        _instances: &mut InstancesBuffer2d,
        _gpu_data: &mut dyn GpuData,
        _context: &RenderContext2d,
    ) {
    }

    /// Flushes collected uniform data to GPU (phase 2).
    ///
    /// This uploads all data collected during `prepare()` calls to the GPU
    /// in a single batch. The default implementation does nothing.
    fn flush(&mut self) {}

    /// Render the given 2D mesh using this material (phase 3).
    ///
    /// # Arguments
    /// * `transform` - The object's world transform
    /// * `scale` - The object's scale
    /// * `camera` - The 2D camera
    /// * `data` - Object rendering properties (color, texture, etc.)
    /// * `mesh` - The object's 2D mesh
    /// * `instances` - Instance data for instanced rendering
    /// * `gpu_data` - Per-object GPU resources created by `create_gpu_data`
    /// * `render_pass` - The active render pass to draw into
    /// * `context` - Render context with viewport info
    fn render(
        &mut self,
        transform: Pose2,
        scale: Vec2,
        camera: &mut dyn Camera2d,
        data: &ObjectData2d,
        mesh: &mut GpuMesh2d,
        instances: &mut InstancesBuffer2d,
        gpu_data: &mut dyn GpuData,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext2d,
    );
}
