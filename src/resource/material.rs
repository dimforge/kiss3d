//! Trait implemented by materials.

use crate::camera::Camera;
use crate::light::Light;
use crate::planar_camera::PlanarCamera;
use crate::resource::{GpuMesh, PlanarMesh};
use crate::scene::{InstancesBuffer, ObjectData, PlanarInstancesBuffer, PlanarObjectData};
use na::{Isometry2, Isometry3, Vector2, Vector3};
use std::any::Any;

/// Context passed to materials during rendering.
pub struct RenderContext<'a> {
    /// The command encoder for this frame.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The color attachment view (either surface texture or offscreen).
    pub color_view: &'a wgpu::TextureView,
    /// The depth attachment view.
    pub depth_view: &'a wgpu::TextureView,
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
pub trait Material {
    /// Creates per-object GPU data for this material.
    ///
    /// This is called once when an object is created. The returned GpuData
    /// holds uniform buffers and other per-object GPU resources.
    fn create_gpu_data(&self) -> Box<dyn GpuData>;

    /// Renders an object using this material.
    ///
    /// # Arguments
    /// * `pass` - The render pass index (for multi-pass rendering)
    /// * `transform` - The object's world transform
    /// * `scale` - The object's scale
    /// * `camera` - The camera
    /// * `light` - The scene light
    /// * `data` - Object rendering properties (color, texture, etc.)
    /// * `mesh` - The object's mesh
    /// * `instances` - Instance data for instanced rendering
    /// * `gpu_data` - Per-object GPU resources created by `create_gpu_data`
    /// * `context` - Render context with encoder and render targets
    fn render(
        &mut self,
        pass: usize,
        transform: &Isometry3<f32>,
        scale: &Vector3<f32>,
        camera: &mut dyn Camera,
        light: &Light,
        data: &ObjectData,
        mesh: &mut GpuMesh,
        instances: &mut InstancesBuffer,
        gpu_data: &mut dyn GpuData,
        context: &mut RenderContext,
    );
}

/// Context passed to planar materials during rendering.
pub struct PlanarRenderContext<'a> {
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
pub trait PlanarMaterial {
    /// Creates per-object GPU data for this material.
    fn create_gpu_data(&self) -> Box<dyn GpuData>;

    /// Render the given planar mesh using this material.
    fn render(
        &mut self,
        transform: &Isometry2<f32>,
        scale: &Vector2<f32>,
        camera: &mut dyn PlanarCamera,
        data: &PlanarObjectData,
        mesh: &mut PlanarMesh,
        instances: &mut PlanarInstancesBuffer,
        gpu_data: &mut dyn GpuData,
        context: &mut PlanarRenderContext,
    );
}
