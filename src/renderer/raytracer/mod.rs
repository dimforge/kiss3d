//! A progressive GPU path tracer for photorealistic rendering.
//!
//! The path tracer renders the existing scene graph (meshes, PBR materials,
//! lights, camera) by Monte-Carlo path tracing on the GPU, accumulating samples
//! across frames for a noise-free, physically-based image. It is driven through
//! [`Window::raytrace_3d`](crate::window::Window::raytrace_3d).
//!
//! Two backends share the same shading kernel:
//! - **Compute** (always available): traverses a CPU-built BVH in a compute
//!   shader. Runs on every wgpu backend, including Metal and the web.
//! - **Hardware ray-query** (`hw_raytracer` feature, capable Vulkan GPUs): uses
//!   wgpu's experimental ray queries against hardware acceleration structures.
//!
//! Accumulation restarts automatically when the camera moves, the viewport is
//! resized, or the scene changes. For in-place vertex edits that the change hash
//! does not capture, call [`RayTracer::mark_dirty`].

mod accumulation;
mod bvh;
mod denoise;
pub(crate) mod environment;
mod gpu_scene;
mod pipeline;
pub mod scene_data;
mod tex_array;
mod tonemap;

use std::path::Path;

use crate::camera::Camera3d;
use crate::light::LightCollection;
use crate::scene::SceneNode3d;

use accumulation::Accumulation;
use denoise::Denoise;
use environment::Environment;
use gpu_scene::GpuScene;
use pipeline::{FrameUniforms, PathTracePipeline};
use tonemap::Tonemap;

/// Which intersection backend the path tracer uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RayBackend {
    /// Portable compute-shader BVH traversal.
    Software,
    /// Hardware ray queries (requires the `hw_raytracer` feature and GPU support).
    #[cfg(feature = "hw_raytracer")]
    Hardware,
}

/// A progressive GPU path tracer.
///
/// Construct one with [`RayTracer::new`] after a [`Window`](crate::window::Window)
/// exists (the GPU context must be initialized), keep it alive across frames so
/// accumulation can progress, and pass it to
/// [`Window::raytrace_3d`](crate::window::Window::raytrace_3d).
pub struct RayTracer {
    backend: RayBackend,
    pipeline: PathTracePipeline,
    tonemap: Tonemap,
    denoise: Denoise,
    gpu_scene: Option<GpuScene>,
    accum: Accumulation,
    sample_index: u32,
    max_bounces: u32,
    samples_per_frame: u32,
    interactive_scale: f32,
    /// Thin-lens aperture radius (0 = pinhole). See [`RayTracer::set_aperture`].
    lens_radius: f32,
    /// Focus distance for the thin-lens camera (world units).
    focus_distance: f32,
    /// HDRI environment map (a black fallback when none is set).
    environment: Environment,
    /// Environment Y-rotation in radians and its luminance scale.
    env_rotation: f32,
    env_intensity: f32,
    last_camera: [f32; 16],
    dirty: bool,
    /// Maximum number of pixels the accumulation buffer may hold, derived from
    /// device buffer-size limits. Larger framebuffers are traced at a reduced
    /// resolution and upscaled by the tonemap pass.
    max_pixels: u64,
    /// Whether the à-trous denoiser runs before tonemapping (default off).
    denoise_enabled: bool,
    /// Number of à-trous iterations when denoising is enabled.
    denoise_iterations: u32,
    /// Whether path tracing is active. When `false`,
    /// [`Window::raytrace_3d`](crate::window::Window::raytrace_3d)
    /// falls back to the rasterizer instead of tracing. Enabled by default.
    enabled: bool,
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

/// Number of à-trous denoiser iterations to run given the configured maximum and
/// how many samples have accumulated into the current image.
///
/// Path-traced noise falls off with the sample count, so full-strength filtering
/// is only worthwhile for the first handful of samples; beyond that the
/// iteration count tapers (log-linearly in the sample count) and, once the image
/// is effectively converged, drops to `0` so the denoiser is skipped entirely.
/// `sample_index` resets on any camera/scene/resolution change, so a fresh,
/// noisy image always gets full-strength filtering again.
fn effective_denoise_iterations(max_iterations: u32, samples: u32) -> u32 {
    if max_iterations == 0 {
        return 0;
    }
    /// At or below this sample count the image is still noisy: full strength.
    const FULL: u32 = 16;
    /// At or above this sample count it is effectively converged: skip it.
    const CONVERGED: u32 = 512;

    if samples <= FULL {
        return max_iterations;
    }
    if samples >= CONVERGED {
        return 0;
    }
    // Taper from `max_iterations` (at FULL samples) down towards 0 (at CONVERGED).
    let t = (samples as f32 / FULL as f32).log2() / (CONVERGED as f32 / FULL as f32).log2();
    let iterations = (max_iterations as f32 * (1.0 - t)).ceil() as u32;
    iterations.max(1)
}

impl RayTracer {
    /// Creates a new path tracer, selecting the best available backend: the
    /// hardware ray-query backend when the `hw_raytracer` feature is enabled and
    /// the GPU supports it, otherwise the portable compute backend.
    ///
    /// # Panics
    /// Panics if the GPU context has not been initialized (i.e. no window exists).
    pub fn new() -> RayTracer {
        Self::with_backend(Self::pick_backend())
    }

    /// Creates a new path tracer that is enabled or not.
    ///
    /// If it is marked as disabled, it will use the raster pipeline instead of path tracing.
    pub fn with_enabled(enabled: bool) -> RayTracer {
        let mut result = Self::default();
        result.enabled = enabled;
        result
    }

    /// Creates a path tracer using a specific intersection backend.
    fn with_backend(backend: RayBackend) -> RayTracer {
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
            denoise: Denoise::new(),
            gpu_scene: None,
            accum: Accumulation::new(1, 1),
            sample_index: 0,
            max_bounces: 8,
            samples_per_frame: 1,
            interactive_scale: 0.5,
            lens_radius: 0.0,
            focus_distance: 1.0,
            environment: Environment::fallback(),
            env_rotation: 0.0,
            env_intensity: 1.0,
            last_camera: [f32::NAN; 16],
            dirty: true,
            max_pixels,
            denoise_enabled: false,
            denoise_iterations: 5,
            enabled: true,
        }
    }

    fn pick_backend() -> RayBackend {
        #[cfg(feature = "hw_raytracer")]
        {
            // The hardware path requires the ray-query device feature (which also
            // gates acceleration structures), enabled at device creation.
            let features = crate::context::Context::get().device.features();
            if features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
                return RayBackend::Hardware;
            }
        }
        RayBackend::Software
    }

    /// Returns the backend selected at construction.
    pub fn backend(&self) -> RayBackend {
        self.backend
    }

    /// Whether path tracing is active (enabled by default).
    ///
    /// See [`set_enabled`](Self::set_enabled).
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Enables or disables path tracing.
    ///
    /// When disabled, [`Window::raytrace_3d`](crate::window::Window::raytrace_3d)
    /// renders the scene with the rasterizer instead of tracing it (the same
    /// output as [`Window::render_3d`](crate::window::Window::render_3d)), while
    /// keeping this `RayTracer` and its accumulated samples around so tracing can
    /// resume when re-enabled. Useful for cheaply A/B-ing the two renderers
    /// without restructuring the render loop. Does not reset accumulation.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Sets the maximum path length (number of bounces). Resets accumulation.
    pub fn set_max_bounces(&mut self, bounces: u32) {
        if self.max_bounces != bounces {
            self.max_bounces = bounces.max(1);
            self.dirty = true;
        }
    }

    /// Enables or disables the edge-aware à-trous denoiser (default off).
    ///
    /// When enabled, an SVGF-style wavelet filter runs over the accumulated
    /// radiance before tonemapping, smoothing Monte-Carlo noise while preserving
    /// edges using the first-hit normal and luminance guides (with albedo
    /// demodulation to keep texture detail). This lets low-sample-count frames
    /// look clean. Does not reset accumulation; when disabled the raw
    /// accumulation is tonemapped exactly as before.
    pub fn set_denoise(&mut self, enabled: bool) {
        self.denoise_enabled = enabled;
    }

    /// Whether the denoiser is currently enabled.
    pub fn denoise(&self) -> bool {
        self.denoise_enabled
    }

    /// Sets the number of à-trous wavelet iterations the denoiser performs
    /// (clamped to at least 1, default 5).
    ///
    /// Each iteration doubles the filter's tap spacing, so more iterations widen
    /// the effective denoising radius (smoother, but slower and more prone to
    /// over-blurring). Has no effect unless denoising is enabled. Does not reset
    /// accumulation.
    pub fn set_denoise_iterations(&mut self, iterations: u32) {
        self.denoise_iterations = iterations.max(1);
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

    /// Sets the thin-lens aperture radius (world units) and focus distance for
    /// depth of field. An aperture of `0` (the default) keeps the pinhole camera.
    ///
    /// Larger apertures blur objects away from the focus plane more strongly.
    /// Resets accumulation.
    pub fn set_aperture(&mut self, lens_radius: f32, focus_distance: f32) {
        let lens_radius = lens_radius.max(0.0);
        let focus_distance = focus_distance.max(1e-3);
        if self.lens_radius != lens_radius || self.focus_distance != focus_distance {
            self.lens_radius = lens_radius;
            self.focus_distance = focus_distance;
            self.dirty = true;
        }
    }

    /// Sets the aperture from a photographic f-number and focus distance.
    ///
    /// The lens radius is `focal_length / (2 * f_number)`; here we approximate the
    /// focal length with the focus distance, giving an intuitive "smaller f-number
    /// = blurrier" control. Resets accumulation.
    pub fn set_f_number(&mut self, f_number: f32, focus_distance: f32) {
        let r = if f_number > 0.0 {
            focus_distance / (2.0 * f_number)
        } else {
            0.0
        };
        self.set_aperture(r, focus_distance);
    }

    /// Loads an equirectangular HDR/LDR environment map for image-based lighting.
    ///
    /// Escaped rays and the background sample this map instead of the procedural
    /// sky. Returns `false` (and keeps the previous environment) if the file
    /// cannot be decoded. Resets accumulation.
    pub fn set_environment_from_file(&mut self, path: &Path) -> bool {
        match Environment::from_file(path) {
            Some(env) => {
                self.environment = env;
                self.dirty = true;
                true
            }
            None => false,
        }
    }

    /// Sets the environment map from an already-decoded equirectangular image.
    /// Resets accumulation.
    pub fn set_environment_image(&mut self, image: &image::DynamicImage) {
        self.environment = Environment::from_image(image);
        self.dirty = true;
    }

    /// Clears the environment map, reverting to the procedural gradient sky.
    /// Resets accumulation.
    pub fn clear_environment(&mut self) {
        self.environment = Environment::fallback();
        self.dirty = true;
    }

    /// Sets the environment rotation about the Y axis (radians) and a luminance
    /// scale multiplier. Resets accumulation.
    pub fn set_environment_orientation(&mut self, rotation_radians: f32, intensity: f32) {
        self.env_rotation = rotation_radians;
        self.env_intensity = intensity.max(0.0);
        self.dirty = true;
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

    /// Resolution `(width, height)` of the guide buffers (the traced resolution).
    pub fn guide_resolution(&self) -> (u32, u32) {
        (self.accum.width, self.accum.height)
    }

    /// The shared accumulation buffer, which also holds the denoiser guide
    /// channels. It is laid out as three contiguous regions of `width * height`
    /// pixels (`vec4<f32>` each): region 0 = radiance, region 1 = first-hit albedo
    /// guide (`rgb`), region 2 = first-hit world normal (`xyz`). Region `k` of
    /// pixel `p` is at element `k * width * height + p`. The buffer has
    /// `STORAGE | COPY_SRC` usage so a region can be copied to a readback buffer;
    /// see [`guide_resolution`](Self::guide_resolution) for `width`/`height`.
    pub fn guide_albedo_buffer(&self) -> &wgpu::Buffer {
        &self.accum.buffer
    }

    /// The shared accumulation buffer holding the guide channels; see
    /// [`guide_albedo_buffer`](Self::guide_albedo_buffer) for the region layout
    /// (the normal guide is region 2, starting at element `2 * width * height`).
    pub fn guide_normal_buffer(&self) -> &wgpu::Buffer {
        &self.accum.buffer
    }

    /// Renders one path-traced frame: refreshes the GPU scene, decides whether to
    /// restart accumulation, dispatches the tracer, and tonemaps to `output_view`.
    ///
    /// Called by `Window::raytrace_3d`; `lights` must already be populated
    /// (which also propagates the scene's world transforms).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render_frame(
        &mut self,
        scene: &SceneNode3d,
        camera: &mut dyn Camera3d,
        lights: &LightCollection,
        background: crate::color::Color,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        // Exposure and tonemap operator are shared with the rasterizer's
        // `HdrSettings` (passed in by `Window::raytrace_3d_frame`) so both
        // renderers display with the same finishing.
        exposure: f32,
        tonemap_operator: u32,
        // Records the path tracer's GPU phases (trace / denoise / tonemap) via
        // per-pass timestamp queries; see `RenderTimings`.
        gpu: &mut crate::renderer::timings::GpuTimer,
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

        let env_present = self.environment.present;
        let uniforms = FrameUniforms {
            inv_view_proj: camera.inverse_transformation().to_cols_array_2d(),
            env_rotation: [
                self.env_rotation.cos(),
                self.env_rotation.sin(),
                self.env_intensity,
                0.0,
            ],
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
            num_emitters: gpu_scene.num_emitters,
            lens_radius: self.lens_radius,
            focus_distance: self.focus_distance,
            has_env: env_present as u32,
            // background.a is otherwise unused (only .rgb is read for the miss
            // color); it carries the "scene has translucent casters" flag so the
            // kernel's shadow rays accumulate colored transmittance only when
            // needed, keeping fully-opaque scenes on the cheap binary-occlusion path.
            background: [
                background.r,
                background.g,
                background.b,
                if gpu_scene.has_translucent { 1.0 } else { 0.0 },
            ],
        };
        self.pipeline.write_uniforms(&uniforms);

        match self.backend {
            RayBackend::Software => {
                self.pipeline.dispatch_compute(
                    encoder,
                    gpu_scene,
                    &self.accum,
                    &self.environment,
                    render_width,
                    render_height,
                    gpu,
                );
            }
            #[cfg(feature = "hw_raytracer")]
            RayBackend::Hardware => {
                self.pipeline.dispatch_hardware(
                    encoder,
                    gpu_scene,
                    &self.accum,
                    &self.environment,
                    render_width,
                    render_height,
                    gpu,
                );
            }
        }

        // Run the edge-aware denoiser (operating on the accumulation/guide
        // buffers at the traced resolution) and tonemap its output; otherwise
        // tonemap the raw accumulation directly, preserving the original path.
        //
        // The number of à-trous iterations is scaled down as the image
        // converges: Monte-Carlo noise shrinks with the sample count, so heavy
        // filtering is only needed for the first few samples. Once effectively
        // converged the denoiser is skipped entirely and the raw accumulation is
        // displayed — saving several full-resolution compute passes per frame on
        // a static, converged view. `sample_index` is still the pre-frame count
        // here (it is advanced below), so add this frame's `spp`.
        let effective_iterations = if self.denoise_enabled {
            effective_denoise_iterations(self.denoise_iterations, self.sample_index + spp)
        } else {
            0
        };
        let radiance = if effective_iterations > 0 {
            self.denoise
                .run(encoder, &self.accum, effective_iterations, gpu)
        } else {
            &self.accum.buffer
        };

        self.tonemap.draw(
            encoder,
            &self.accum,
            radiance,
            exposure,
            tonemap_operator,
            output_view,
            width,
            height,
            gpu,
        );

        self.sample_index += spp;
    }
}

impl Default for RayTracer {
    fn default() -> Self {
        Self::new()
    }
}
