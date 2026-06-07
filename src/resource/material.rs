//! Trait implemented by materials.

use crate::camera::Camera2d;
use crate::camera::Camera3d;
use crate::light::LightCollection;
use crate::resource::{GpuMesh2d, GpuMesh3d};
use crate::scene::{InstancesBuffer2d, InstancesBuffer3d, ObjectData2d, ObjectData3d};
use glamx::{Pose2, Pose3, Vec2, Vec3};
use std::any::Any;

/// Which transparency pass is being rendered.
///
/// The rasterizer draws opaque surfaces first (depth write, into the HDR film),
/// then transparent surfaces with weighted-blended order-independent transparency
/// into separate accumulation targets. Materials use this to draw only the
/// matching surfaces (and to pick the opaque vs. OIT pipeline).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum RenderPhase {
    /// Depth + view-position prepass (single target), rendered before the opaque
    /// pass to drive screen-space effects such as SSAO.
    Prepass,
    /// Opaque surfaces (alpha == 1), plus wireframe/point overlays.
    #[default]
    Opaque,
    /// Transparent surfaces (alpha < 1), drawn into the OIT accumulation targets.
    Transparent,
    /// Refractive (glass) surfaces, drawn into the resolved HDR scene after the
    /// opaque pass so they can sample the scene behind them (screen-space
    /// refraction). Single-sample; reads the transmission-background snapshot.
    Transmission,
}

/// Context passed to materials during rendering.
///
/// This contains metadata about the render target. The actual render pass
/// is passed separately to enable batching multiple draw calls.
pub struct RenderContext {
    /// Which transparency pass is being rendered (opaque vs. OIT transparent).
    pub phase: RenderPhase,
    /// The surface format.
    pub surface_format: wgpu::TextureFormat,
    /// The sample count for MSAA.
    pub sample_count: u32,
    /// The viewport width in pixels.
    pub viewport_width: u32,
    /// The viewport height in pixels.
    pub viewport_height: u32,
    /// Render-layer mask of the camera being rendered. An object is drawn only
    /// when its own layer mask shares a bit with this one. `u32::MAX` (the
    /// default) renders every layer.
    pub render_layers: u32,
    /// Forces back-face culling off for this pass. Set by the planar-reflector
    /// mirror render, whose reflected projection flips triangle winding (so normal
    /// back-face culling would render closed objects inside-out).
    pub force_no_cull: bool,
    /// Shadow-map GPU resources supplied by the window's shadow mapper, folded into
    /// the object material's view (group 0) bind group.
    ///
    /// When `None`, materials fall back to their own neutral "no shadows" resources
    /// so rendering stays correct when shadows are disabled. Cloning the handles is
    /// cheap (they are reference-counted).
    pub shadow: Option<ShadowResources>,
}

/// The shadow mapper's GPU resources, handed to the object material so it can bind
/// them as part of its view (group 0) bind group — rather than a separate group, so
/// the per-object deform group fits within WebGPU's 4-bind-group cap. All fields are
/// cheap-to-clone reference-counted wgpu handles.
#[derive(Clone)]
pub struct ShadowResources {
    /// Depth atlas array view (sampled for comparison).
    pub atlas: wgpu::TextureView,
    /// Comparison sampler for hardware PCF.
    pub compare_sampler: wgpu::Sampler,
    /// Shadow uniforms buffer.
    pub uniform: wgpu::Buffer,
    /// Colored-transmittance atlas array view.
    pub transmittance: wgpu::TextureView,
    /// Filtering sampler for the transmittance atlas.
    pub transmittance_sampler: wgpu::Sampler,
}

/// The environment-lighting (IBL) resources a window supplies to materials each
/// frame: a mip-chained equirectangular environment map plus its orientation and
/// intensity. Materials that support image-based lighting consume this in
/// [`Material3d::set_environment_lighting`].
pub struct EnvLight<'a> {
    /// View over the mip-chained equirectangular environment.
    pub view: &'a wgpu::TextureView,
    /// Sampler for the environment (trilinear).
    pub sampler: &'a wgpu::Sampler,
    /// Number of mip levels (max sampleable LOD is `mip_count - 1`).
    pub mip_count: u32,
    /// Luminance multiplier.
    pub intensity: f32,
    /// Y-axis rotation in radians (matches the skybox).
    pub rotation: f32,
}

/// One reflection probe's placement, as supplied to materials each frame in
/// [`ProbeLighting`]. Mirrors `renderer::ReflectionProbe` but decoupled from the
/// renderer module so materials only depend on the trait crate.
#[derive(Copy, Clone, Debug)]
pub struct ProbeData {
    /// World-space center (capture viewpoint).
    pub center: Vec3,
    /// Half-extents of the parallax/influence box (world AABB), centered on `center`.
    pub half_extents: Vec3,
    /// Soft-edge width (world units) over which the probe fades to the global env.
    pub falloff: f32,
    /// Luminance multiplier.
    pub intensity: f32,
    /// Y-axis rotation (radians), matching the skybox convention.
    pub rotation: f32,
    /// Array layer holding this probe's equirectangular map.
    pub layer: u32,
}

/// The reflection-probe resources a window supplies to materials each frame: the
/// shared mip-chained equirectangular probe array plus the active probe records.
/// Consumed in [`Material3d::set_reflection_probes`].
pub struct ProbeLighting<'a> {
    /// View over the mip-chained equirectangular probe array (one layer per probe).
    pub array_view: &'a wgpu::TextureView,
    /// Active probes (at most `renderer::MAX_PROBES`).
    pub probes: &'a [ProbeData],
    /// Maximum sampleable LOD of the probe array (max roughness → this mip).
    pub max_lod: f32,
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

    /// Whether this material draws during the transparent ([`RenderPhase::Transparent`])
    /// pass — the order-independent-transparency geometry pass, which has different
    /// (multiple) render targets than the opaque pass.
    ///
    /// Defaults to `false`: a material is only invoked in the opaque phase. A custom
    /// material that does not build a pipeline matching the OIT pass's targets must
    /// keep this `false`, otherwise its pipeline will be incompatible with that pass.
    fn renders_in_transparent_phase(&self) -> bool {
        false
    }

    /// Supplies (or clears) the image-based-lighting environment for this frame.
    ///
    /// Called once per frame by the window with the active skybox environment, or
    /// `None` when no skybox/IBL is set. Materials that don't support IBL ignore
    /// it (the default no-op).
    fn set_environment_lighting(&mut self, _env: Option<EnvLight<'_>>) {}

    /// Supplies (or clears) the reflection probes for this frame: the shared
    /// equirectangular probe array plus the active probe records. Reflective
    /// surfaces inside a probe's influence box sample it (parallax-corrected)
    /// instead of the global environment. `None` (or an empty probe list)
    /// disables probes. Default no-op (materials without probe support).
    fn set_reflection_probes(&mut self, _probes: Option<ProbeLighting<'_>>) {}

    /// Supplies (or clears) the screen-space ambient-occlusion texture for this
    /// frame. The material samples it per pixel to darken ambient lighting.
    /// `None` disables it. Default no-op.
    fn set_ssao(&mut self, _ao: Option<&wgpu::TextureView>) {}

    /// Supplies (or clears) the transmission background — the resolved opaque scene
    /// color (with a blurred mip chain) that refractive (glass) objects sample to
    /// refract the scene behind them. `None` falls back to a placeholder. No-op by
    /// default (materials without refractive-transmission support).
    fn set_transmission_background(&mut self, _bg: Option<&wgpu::TextureView>) {}

    /// Toggles reflection-probe *capture mode* for the next frame uniform: while
    /// on, the material renders with the fixed-light (non-clustered) path, since
    /// the per-face capture views have no clustered cull data. Default no-op.
    fn set_capture_mode(&mut self, _on: bool) {}

    /// Sets a world-space clip plane `(a, b, c, d)` for the next frame: fragments
    /// with `dot((a,b,c), world_pos) + d < 0` are discarded. Used by reflector
    /// capture to clip geometry behind the mirror. `None` disables it. Default no-op.
    fn set_clip_plane(&mut self, _plane: Option<[f32; 4]>) {}

    /// Supplies the clustered forward+ storage buffers for this frame (the light
    /// list, per-cluster light grid, and global light-index list). Called by the
    /// window after the light-culling compute pass when clustered lighting is
    /// active; `force_rebind` is set when the light buffer was reallocated. Default
    /// no-op (materials without clustered support, or backends that fall back to
    /// the fixed-light path).
    fn set_clustered_buffers(
        &mut self,
        _lights: &wgpu::Buffer,
        _grid: &wgpu::Buffer,
        _index: &wgpu::Buffer,
        _force_rebind: bool,
    ) {
    }

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
