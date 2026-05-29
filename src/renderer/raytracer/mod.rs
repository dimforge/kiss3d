//! A progressive GPU path tracer for photorealistic rendering.
//!
//! The path tracer renders the existing scene graph (meshes, PBR materials,
//! lights, camera) by Monte-Carlo path tracing on the GPU, accumulating samples
//! across frames for a noise-free, physically-based image. It is driven through
//! [`Window::render_raytraced`](crate::window::Window::render_raytraced).
//!
//! Two backends share the same shading kernel:
//! - **Compute** (always available): traverses a CPU-built BVH in a compute
//!   shader. Runs on every wgpu backend, including Metal and the web.
//! - **Hardware ray-query** (`raytracing` feature, capable Vulkan GPUs): uses
//!   wgpu's experimental ray queries against hardware acceleration structures.
//!
//! Accumulation restarts automatically when the camera moves, the viewport is
//! resized, or the scene changes. For in-place vertex edits that the change hash
//! does not capture, call [`RayTracer::mark_dirty`].

mod accumulation;
mod bvh;
mod gpu_scene;
mod pipeline;
pub mod scene_data;
mod tonemap;

use crate::camera::Camera3d;
use crate::light::LightCollection;
use crate::scene::SceneNode3d;

use accumulation::Accumulation;
use gpu_scene::GpuScene;
use pipeline::{FrameUniforms, PathTracePipeline};
use tonemap::Tonemap;

/// Which intersection backend the path tracer uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RayBackend {
    /// Portable compute-shader BVH traversal.
    Compute,
    /// Hardware ray queries (requires the `raytracing` feature and GPU support).
    #[cfg(feature = "raytracing")]
    HardwareRayQuery,
}

/// A progressive GPU path tracer.
///
/// Construct one with [`RayTracer::new`] after a [`Window`](crate::window::Window)
/// exists (the GPU context must be initialized), keep it alive across frames so
/// accumulation can progress, and pass it to
/// [`Window::render_raytraced`](crate::window::Window::render_raytraced).
pub struct RayTracer {
    backend: RayBackend,
    pipeline: PathTracePipeline,
    tonemap: Tonemap,
    gpu_scene: Option<GpuScene>,
    accum: Accumulation,
    sample_index: u32,
    max_bounces: u32,
    samples_per_frame: u32,
    exposure: f32,
    interactive_scale: f32,
    last_camera: [f32; 16],
    dirty: bool,
    /// Maximum number of pixels the accumulation buffer may hold, derived from
    /// device buffer-size limits. Larger framebuffers are traced at a reduced
    /// resolution and upscaled by the tonemap pass.
    max_pixels: u64,
}

/// Caps `(width, height)` to at most `max_pixels` while preserving aspect ratio.
fn capped_resolution(width: u32, height: u32, max_pixels: u64) -> (u32, u32) {
    let pixels = width as u64 * height as u64;
    if pixels == 0 {
        return (1, 1);
    }
    if pixels <= max_pixels {
        return (width, height);
    }
    let scale = (max_pixels as f64 / pixels as f64).sqrt();
    let rw = ((width as f64 * scale).floor() as u32).max(1);
    let rh = ((height as f64 * scale).floor() as u32).max(1);
    (rw, rh)
}

impl RayTracer {
    /// Creates a new path tracer, selecting the best available backend.
    ///
    /// # Panics
    /// Panics if the GPU context has not been initialized (i.e. no window exists).
    pub fn new() -> RayTracer {
        let backend = Self::pick_backend();

        // The accumulation buffer is bound as a single storage buffer, so it is
        // limited by both the max buffer size and the max storage-buffer binding
        // size. Cap the traced resolution to whatever fits.
        let limits = crate::context::Context::get().device.limits();
        let max_bytes = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let max_pixels = (max_bytes / 16).max(1);

        RayTracer {
            backend,
            pipeline: PathTracePipeline::new(backend),
            tonemap: Tonemap::new(),
            gpu_scene: None,
            accum: Accumulation::new(1, 1),
            sample_index: 0,
            max_bounces: 8,
            samples_per_frame: 1,
            exposure: 1.0,
            interactive_scale: 0.5,
            last_camera: [f32::NAN; 16],
            dirty: true,
            max_pixels,
        }
    }

    fn pick_backend() -> RayBackend {
        #[cfg(feature = "raytracing")]
        {
            // The hardware path requires the ray-query device feature (which also
            // gates acceleration structures), enabled at device creation.
            let features = crate::context::Context::get().device.features();
            if features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
                return RayBackend::HardwareRayQuery;
            }
        }
        RayBackend::Compute
    }

    /// Returns the backend selected at construction.
    pub fn backend(&self) -> RayBackend {
        self.backend
    }

    /// Sets the maximum path length (number of bounces). Resets accumulation.
    pub fn set_max_bounces(&mut self, bounces: u32) {
        if self.max_bounces != bounces {
            self.max_bounces = bounces.max(1);
            self.dirty = true;
        }
    }

    /// Sets the tonemap exposure multiplier. Does not reset accumulation.
    pub fn set_exposure(&mut self, exposure: f32) {
        self.exposure = exposure;
    }

    /// Number of path-tracing samples computed per rendered frame (default 1).
    ///
    /// Higher values converge in fewer frames at the cost of a longer frame; it
    /// also amortizes per-dispatch overhead for headless/batch rendering. Resets
    /// accumulation.
    pub fn set_samples_per_frame(&mut self, samples: u32) {
        let samples = samples.max(1);
        if self.samples_per_frame != samples {
            self.samples_per_frame = samples;
            self.dirty = true;
        }
    }

    /// Resolution scale used while the camera is moving, in `(0, 1]` (default 0.5).
    ///
    /// While the camera moves the image is restarting anyway, so it is traced at
    /// this fraction of the framebuffer resolution and upscaled — making
    /// interaction much faster — then re-traced at full resolution once the camera
    /// settles. Set to `1.0` to always trace at full resolution.
    pub fn set_interactive_scale(&mut self, scale: f32) {
        self.interactive_scale = scale.clamp(0.05, 1.0);
    }

    /// Number of samples accumulated into the current image.
    pub fn samples_accumulated(&self) -> u32 {
        self.sample_index
    }

    /// Forces accumulation to restart on the next frame (e.g. after editing
    /// vertex positions in place, which the automatic change detection misses).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Renders one path-traced frame: refreshes the GPU scene, decides whether to
    /// restart accumulation, dispatches the tracer, and tonemaps to `output_view`.
    ///
    /// Called by `Window::render_raytraced`; `lights` must already be populated
    /// (which also propagates the scene's world transforms).
    pub(crate) fn render_frame(
        &mut self,
        scene: &SceneNode3d,
        camera: &mut dyn Camera3d,
        lights: &LightCollection,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        // Detect what changed this frame. A moving camera *or* a moving/changed
        // scene both invalidate the accumulated image — there is no longer a
        // consistent image to average — so both restart accumulation.
        let cam = camera.transformation().to_cols_array();
        let camera_moved = cam != self.last_camera;
        self.last_camera = cam;

        // Cheap content hash (no vertex arrays built); the expensive `gather` only
        // runs on an actual change.
        let hash = scene_data::scene_hash(scene, lights);
        let scene_changed = self.gpu_scene.as_ref().is_none_or(|g| g.hash != hash);

        // While anything is in motion the image is restarting every frame anyway,
        // so trace at a reduced resolution for responsiveness (covers both camera
        // and object/light animation); trace full-resolution once everything is
        // still so the converged result stays sharp. Clamp to the buffer limit.
        let moving = camera_moved || scene_changed;
        let scale = if moving { self.interactive_scale } else { 1.0 };
        let sw = ((width as f32 * scale).round() as u32).max(1);
        let sh = ((height as f32 * scale).round() as u32).max(1);
        let (render_width, render_height) = capped_resolution(sw, sh, self.max_pixels);

        let mut reset = camera_moved;
        if scene_changed {
            let rt_scene = scene_data::gather(scene, lights);
            self.gpu_scene = Some(GpuScene::build(&rt_scene, self.backend));
            reset = true;
        }

        // Resize the accumulation buffer if the render resolution changed.
        if self.accum.ensure(render_width, render_height) {
            reset = true;
        }

        if self.dirty {
            reset = true;
            self.dirty = false;
        }
        if reset {
            self.sample_index = 0;
        }

        let gpu_scene = self.gpu_scene.as_ref().expect("gpu scene just ensured");
        let spp = self.samples_per_frame.max(1);

        let uniforms = FrameUniforms {
            inv_view_proj: camera.inverse_transformation().to_cols_array_2d(),
            cam_eye: camera.eye().to_array(),
            width: render_width,
            height: render_height,
            sample_index: self.sample_index,
            num_triangles: gpu_scene.num_triangles,
            num_lights: gpu_scene.num_lights,
            ambient: lights.ambient,
            max_bounces: self.max_bounces,
            seed: self.sample_index,
            samples_per_frame: spp,
        };
        self.pipeline.write_uniforms(&uniforms);

        match self.backend {
            RayBackend::Compute => {
                self.pipeline.dispatch_compute(
                    encoder,
                    gpu_scene,
                    &self.accum,
                    render_width,
                    render_height,
                );
            }
            #[cfg(feature = "raytracing")]
            RayBackend::HardwareRayQuery => {
                self.pipeline.dispatch_hardware(
                    encoder,
                    gpu_scene,
                    &self.accum,
                    render_width,
                    render_height,
                );
            }
        }

        self.tonemap
            .draw(encoder, &self.accum, self.exposure, output_view, width, height);

        self.sample_index += spp;
    }
}

impl Default for RayTracer {
    fn default() -> Self {
        Self::new()
    }
}
