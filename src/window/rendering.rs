//! Rendering functionality.

#![allow(clippy::await_holding_refcell_ref)]

use crate::camera::{Camera2d, Camera3d, FixedView3d};
use crate::context::Context;
use crate::event::WindowEvent;
use crate::light::LightCollection;
use crate::post_processing::{PostProcessingContext, PostProcessingEffect};
use crate::prelude::FixedView2d;
use crate::renderer::timings::{CpuTimer, RenderTimings};
use crate::renderer::{RayTracer, Renderer3d};
use crate::resource::{
    MaterialManager2d, MaterialManager3d, RenderContext, RenderContext2d, RenderContext2dEncoder,
    RenderPhase, RenderTarget,
};
use crate::scene::{SceneNode2d, SceneNode3d};

use super::Window;

/// Grace period during which the first frame keeps retrying surface acquisition
/// before giving up. A freshly created window — particularly on macOS — may need
/// the event loop to be pumped a few times before its surface is presentable.
#[cfg(not(target_arch = "wasm32"))]
const STARTUP_SURFACE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// Delay between surface acquisition attempts while waiting for the first frame.
#[cfg(not(target_arch = "wasm32"))]
const SURFACE_RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);

impl Window {
    /// Renders one frame of a 3D scene.
    ///
    /// This is the main rendering function that should be called in your render loop.
    /// It handles events, updates the scene, renders all objects, and swaps buffers.
    ///
    /// # Arguments
    /// * `scene` - The 3D scene graph to render
    /// * `camera` - The camera used for viewing the scene
    ///
    /// # Returns
    /// `true` if rendering should continue, `false` if the window should close
    ///
    /// # Example
    /// ```no_run
    /// use kiss3d::prelude::*;
    ///
    /// #[kiss3d::main]
    /// async fn main() {
    ///     let mut window = Window::new("My Application").await;
    ///     let mut camera = OrbitCamera3d::default();
    ///     let mut scene = SceneNode3d::empty();
    ///
    ///     while window.render_3d(&mut scene, &mut camera).await {
    ///         // Your per-frame code here
    ///     }
    /// }
    /// ```
    ///
    /// # Platform-specific
    /// - **Native**: Returns immediately after rendering one frame
    /// - **WASM**: Yields to the browser's event loop and returns when the next frame is ready
    pub async fn render_3d(&mut self, scene: &mut SceneNode3d, camera: &mut impl Camera3d) -> bool {
        self.render(Some(scene), None, Some(camera), None, None, None)
            .await
    }

    pub async fn render_2d(&mut self, scene: &mut SceneNode2d, camera: &mut impl Camera2d) -> bool {
        self.render(None, Some(scene), None, Some(camera), None, None)
            .await
    }

    // `scene`/`camera` are only taken mutably (via `as_deref_mut`) by the
    // `rt_switcher` block below; without that feature the `mut` is unused.
    #[cfg_attr(not(feature = "rt_switcher"), allow(unused_mut))]
    pub async fn render(
        &mut self,
        mut scene: Option<&mut SceneNode3d>,
        scene_2d: Option<&mut SceneNode2d>,
        mut camera: Option<&mut dyn Camera3d>,
        camera_2d: Option<&mut dyn Camera2d>,
        renderer: Option<&mut dyn Renderer3d>,
        post_processing: Option<&mut dyn PostProcessingEffect>,
    ) -> bool {
        #[cfg(feature = "rt_switcher")]
        if let (Some(mut rt), Some(camera), Some(scene)) = (
            self.raytracer.0.take(),
            camera.as_deref_mut(),
            scene.as_deref_mut(),
        ) {
            // NOTE: this will skip 2D completely.
            // Indicate the raytracer was enabled before calling `raytrace_3d`.
            // This is useful so we can restore the correct state depending on
            // the switch input handling.
            self.raytracer.1 = true;
            let result = self.raytrace_3d(scene, camera, &mut rt).await;
            if self.raytracer.0.is_none() && self.raytracer.1 {
                self.raytracer.0 = Some(rt);
            }
            return result;
        }

        let mut default_cam2 = FixedView2d::default();
        let mut default_cam = FixedView3d::default();

        let camera = camera.unwrap_or(&mut default_cam);
        let camera_2d = camera_2d.unwrap_or(&mut default_cam2);
        self.handle_events(camera, camera_2d);
        self.render_single_frame(
            scene,
            scene_2d,
            camera,
            camera_2d,
            renderer,
            post_processing,
        )
        .await
    }

    async fn render_single_frame(
        &mut self,
        mut scene: Option<&mut SceneNode3d>,
        mut scene_2d: Option<&mut SceneNode2d>,
        camera: &mut dyn Camera3d,
        camera_2d: &mut dyn Camera2d,
        mut renderer: Option<&mut dyn Renderer3d>,
        mut post_processing: Option<&mut dyn PostProcessingEffect>,
    ) -> bool {
        // Frame timing: CPU wall-clock for the whole frame (and submit/present
        // below) plus per-pass GPU timestamps recorded into the GPU timer. The
        // frame-to-frame wall-clock period (true FPS) is the delta between
        // successive frames at this same point — it captures the vsync/present wait
        // and app/event time that the per-pass GPU timestamps don't.
        let frame_start = web_time::Instant::now();
        let frame_wall = self
            .last_frame_instant
            .map(|prev| frame_start.duration_since(prev))
            .unwrap_or_default();
        self.last_frame_instant = Some(frame_start);
        let cpu = CpuTimer::start();
        self.gpu_timer.begin_frame();

        // A visible window renders into its surface; a hidden window has no
        // presentable surface, so it renders into an offscreen texture that
        // `snap` and recording can still read back.
        let offscreen = self.hidden;

        // Acquire the surface texture for visible windows. A just-created
        // window may not be presentable yet, so `acquire_next_frame` retries
        // until it is.
        let frame = if offscreen {
            None
        } else {
            match self.acquire_next_frame() {
                Some(frame) => Some(frame),
                None => return !self.should_close(),
            }
        };

        // Read the size only now: while retrying, a pending resize event may
        // have been processed and the surface reconfigured.
        let w = self.width();
        let h = self.height();

        camera_2d.handle_event(&self.canvas, &WindowEvent::FramebufferSize(w, h));
        camera.handle_event(&self.canvas, &WindowEvent::FramebufferSize(w, h));
        camera_2d.update(&self.canvas);
        camera.update(&self.canvas);

        // No need to update the light position here - it's computed per-frame
        // in the material's prepare() based on the camera position

        // `OffscreenBuffers` are never multisampled, so offscreen rendering
        // always uses a single sample (a hidden window is not antialiased).
        let sample_count = if offscreen {
            1
        } else {
            self.canvas.sample_count()
        };

        let ctxt = Context::get();
        let mut encoder = ctxt.create_command_encoder(Some("kiss3d_frame_encoder"));

        // Resize the HDR film + the offscreen render targets if needed.
        //
        // The rasterizer's material/renderer pipelines are built per sample count (a
        // lazy `PipelineCache` keyed by `context.sample_count`), so the HDR film is
        // multisampled to match the canvas. The scene is rasterized into the MSAA HDR
        // attachment and resolved into the single-sample HDR texture (see
        // `resolve_view` below) before tonemapping.
        self.hdr.resize(w, h, sample_count);
        self.post_process_render_target
            .resize(w, h, self.canvas.surface_format());
        if offscreen {
            if self.offscreen_output_target.is_none() {
                self.offscreen_output_target =
                    Some(self.framebuffer_manager.new_render_target(w, h, true));
            }
            self.offscreen_output_target.as_mut().unwrap().resize(
                w,
                h,
                self.canvas.surface_format(),
            );
        }

        // The view that receives the final composited image: the surface
        // texture for a visible window, the offscreen color texture otherwise.
        let frame_view = match &frame {
            Some(frame) => frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            None => self
                .offscreen_output_target
                .as_ref()
                .expect("offscreen render target was just created")
                .color_view()
                .expect("offscreen render target is never the screen")
                .clone(),
        };

        // The rasterized scene always renders into the HDR film. `color_view` is
        // the MSAA attachment when multisampling is on, the single-sample HDR
        // texture otherwise; `resolve_view` is the MSAA resolve destination.
        // A final tonemap pass converts this HDR image to the LDR output below.
        let color_view = self.hdr.scene_render_view().clone();
        let resolve_view = self.hdr.scene_resolve_view().cloned();

        // The depth attachment must match the scene target's sample count. The
        // canvas depth texture is built MSAA-aware; offscreen rendering is always
        // single-sampled and uses the offscreen target's depth.
        let depth_view = if offscreen {
            self.offscreen_output_target
                .as_ref()
                .expect("offscreen render target was just created")
                .depth_view()
                .expect("offscreen render target is never the screen")
                .clone()
        } else {
            self.canvas.depth_view().clone()
        };

        // Clear the render target at the start of the frame
        {
            let bg = self.background;
            let clear_ts = self.gpu_timer.render_scope("clear");
            let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg.r as f64,
                            g: bg.g as f64,
                            b: bg.b as f64,
                            a: bg.a as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: clear_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // Render pass is dropped here, ending the clear pass
        }

        // Signal start of new frame to all materials (for dynamic buffer clearing)
        MaterialManager3d::get_global_manager(|mm| mm.begin_frame());

        // Supply the skybox environment to the default material for image-based
        // lighting (or clear it when no skybox is set).
        {
            let default_mat = MaterialManager3d::get_global_manager(|mm| mm.get_default());
            let mut mat = default_mat.borrow_mut();
            if let Some(env) = self.skybox.ibl_env() {
                mat.set_environment_lighting(Some(crate::resource::EnvLight {
                    view: &env.view,
                    sampler: &env.sampler,
                    mip_count: env.mip_count,
                    intensity: self.skybox.intensity(),
                    rotation: self.skybox.rotation(),
                }));
            } else {
                mat.set_environment_lighting(None);
            }
        }

        // Supply the reflection probes (parallax-corrected localized env maps) to
        // the default material, or clear them when none are registered.
        {
            let default_mat = MaterialManager3d::get_global_manager(|mm| mm.get_default());
            let mut mat = default_mat.borrow_mut();
            match self.reflection_probes.as_ref() {
                Some(probes) if !probes.is_empty() => {
                    let records: Vec<crate::resource::ProbeData> = probes
                        .probes()
                        .iter()
                        .enumerate()
                        .map(|(layer, p)| crate::resource::ProbeData {
                            center: p.center,
                            half_extents: p.half_extents,
                            falloff: p.falloff,
                            intensity: p.intensity,
                            rotation: p.rotation,
                            layer: layer as u32,
                        })
                        .collect();
                    mat.set_reflection_probes(Some(crate::resource::ProbeLighting {
                        array_view: probes.array_view(),
                        probes: &records,
                        max_lod: probes.max_lod(),
                    }));
                }
                _ => mat.set_reflection_probes(None),
            }
        }

        // SSAO / SSR share the geometry G-buffer prepass. Ensure the prepass
        // resources exist/sized whenever either effect is active, then bind the
        // (persistent) AO texture + set the enable flag on the material BEFORE
        // prepare() — the flag is baked into the frame uniform there. The AO
        // contents are computed below (after prepare propagates transforms) into
        // that same texture, before the opaque pass samples it.
        let ssr_active = self.ssr_enabled && Context::get().supports_clustered_lighting();
        // DoF only needs the view-position G-buffer (no compute/storage), so it runs
        // on every backend the prepass does.
        let dof_active = self.dof_enabled;
        // Refractive glass: needs the opaque scene resolved + the prepass depth (for
        // occlusion), even when SSAO/SSR/DoF are all off.
        let glass_active = self.transmission_enabled
            && scene
                .as_deref()
                .is_some_and(|s| s.has_refractive_surfaces());
        if self.ssao_enabled || ssr_active || dof_active || glass_active {
            let ssao = self
                .ssao
                .get_or_insert_with(|| crate::renderer::Ssao::new(w, h));
            ssao.resize(w, h);
        }
        {
            let default_mat = MaterialManager3d::get_global_manager(|mm| mm.get_default());
            let mut mat = default_mat.borrow_mut();
            if self.ssao_enabled {
                mat.set_ssao(Some(self.ssao.as_ref().unwrap().ao_view()));
            } else {
                mat.set_ssao(None);
            }
        }

        // Create a light collection for this frame
        let mut lights = LightCollection::with_ambient(self.ambient_intensity);
        lights.ambient_color = self.ambient_color;
        lights.fog = self.fog;

        // Reflection-probe runtime capture (queued via `capture_reflection_probe`).
        // For each queued probe, render the scene into six cube faces from the probe
        // center and reproject them into the probe's environment map. Runs before
        // the main passes so the captured maps are ready for the opaque pass that
        // samples them. Uses the fixed-light path (per-face views have no clustered
        // cull data) and the previous frame's shadow atlas.
        if !self.pending_probe_captures.is_empty() && scene.is_some() {
            let captures = std::mem::take(&mut self.pending_probe_captures);
            let (znear, zfar) = camera.clip_planes();
            const FACE: u32 = 256;
            if self.probe_capture.is_none() {
                self.probe_capture = Some(crate::renderer::ProbeCapture::new(FACE));
            }
            // Force the non-clustered shading path for the capture frame uniforms.
            MaterialManager3d::get_global_manager(|mm| mm.get_default())
                .borrow_mut()
                .set_capture_mode(true);

            // Each cube face must be its own queue submission. The frame uniform
            // (and the object-uniform buffer) are shared and uploaded with
            // `queue.write_buffer`, whose writes coalesce *before* any command
            // buffer in the same submission runs — so if all six faces shared the
            // frame encoder, the last face's (or the main render's) uniforms would
            // clobber every earlier face, capturing the main view with probes on
            // instead of six probe-less cube faces (a feedback loop). Submitting
            // per face makes each face's uniforms take effect.
            let ctxt = Context::get();
            let sky_set = self.skybox.is_set();
            for idx in captures {
                let center = match self.reflection_probes.as_ref() {
                    Some(p) if idx < p.len() => p.probes()[idx].center,
                    _ => continue,
                };
                for face in 0..6usize {
                    let mut cam = crate::renderer::CubeFaceCamera::new(center, face, znear, zfar);
                    // Bump the frame counter so prepare writes this face's uniforms.
                    MaterialManager3d::get_global_manager(|mm| mm.begin_frame());
                    let mut cap_lights = LightCollection::with_ambient(self.ambient_intensity);
                    cap_lights.ambient_color = self.ambient_color;
                    cap_lights.fog = self.fog;
                    if let Some(scene) = scene.as_deref_mut() {
                        scene
                            .data_mut()
                            .prepare(0, &mut cam, &mut cap_lights, FACE, FACE);
                        scene.update_deformations();
                    }
                    MaterialManager3d::get_global_manager(|mm| mm.flush());

                    let mut fenc = ctxt.create_command_encoder(Some("probe_capture_face_encoder"));
                    let cap = self.probe_capture.as_ref().unwrap();
                    if sky_set {
                        self.skybox.render(
                            &mut fenc,
                            cap.face_color_view(face),
                            1,
                            cam.inverse_transformation(),
                            None,
                        );
                    }
                    let ctx = RenderContext {
                        surface_format: crate::post_processing::HDR_FORMAT,
                        sample_count: 1,
                        viewport_width: FACE,
                        viewport_height: FACE,
                        render_layers: self.reflection_capture_layers,
                        force_no_cull: false,
                        shadow: Some(self.shadow_mapper.resources()),
                        phase: RenderPhase::Opaque,
                    };
                    {
                        let load = if sky_set {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                        };
                        let probe_ts = self.gpu_timer.render_scope("probe");
                        let mut pass = fenc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("probe_capture_face"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: cap.face_color_view(face),
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load,
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: cap.depth_view(),
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: probe_ts,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        if let Some(scene) = scene.as_deref_mut() {
                            scene
                                .data_mut()
                                .render(0, &mut cam, &cap_lights, &mut pass, &ctx);
                        }
                    }
                    ctxt.submit(std::iter::once(fenc.finish()));
                }

                // Reproject the six faces into the probe's environment map + mips
                // (its own submission, after the face submits that produced them).
                if let Some(probes) = self.reflection_probes.as_ref() {
                    let dst = probes.layer_mip0_view(idx);
                    let mut renc = ctxt.create_command_encoder(Some("probe_reproject_encoder"));
                    self.probe_capture.as_ref().unwrap().reproject(
                        &mut renc,
                        &dst,
                        &mut self.gpu_timer,
                    );
                    probes.generate_layer_mips(&mut renc, idx, Some(&mut self.gpu_timer));
                    ctxt.submit(std::iter::once(renc.finish()));
                }
            }

            // Restore clustered shading and bump the counter so the main passes
            // re-write the real camera's frame uniforms below.
            MaterialManager3d::get_global_manager(|mm| mm.get_default())
                .borrow_mut()
                .set_capture_mode(false);
            MaterialManager3d::get_global_manager(|mm| mm.begin_frame());
        }

        // === Planar reflectors (mirrors) ===
        // Discover reflector surfaces in the scene, render the scene from each one's
        // mirror camera into its own texture, and store the reflected view-proj so
        // the surface samples it during the main pass. Runs before the per-pass loop;
        // like probe capture it uses the previous frame's shadow atlas and a separate
        // queue submission per reflector (the `write_buffer` coalescing rule).
        self.render_reflectors(scene.as_deref_mut(), camera, w, h);

        // Skybox: drawn full-screen into the HDR film right after the clear, so the
        // opaque pass overwrites it wherever geometry is visible. Uses the primary
        // (pass 0) inverse view-projection; since the sky is at infinity this is
        // also correct for stereo (both eyes share the view rotation). No-op when
        // no skybox environment is set. Recorded into `encoder` here (after the
        // clear pass, before the opaque pass — command order is preserved), but
        // *issued* after the probe-capture and planar-reflector passes so its
        // `write_buffer` to the shared skybox uniform is the last one before the
        // main submit (those auxiliary passes reuse the same uniform with their own
        // matrices and would otherwise leave it holding a mirror/cube-face view).
        if self.skybox.is_set() {
            self.skybox.render(
                &mut encoder,
                &color_view,
                sample_count,
                camera.inverse_transformation(),
                Some(&mut self.gpu_timer),
            );
        }

        // Render the 3D scene using two-phase rendering
        for pass in 0usize..camera.num_passes() {
            camera.start_pass(pass, &self.canvas);

            // Phase 1: Prepare - collect uniforms in CPU memory and gather lights from scene
            if let Some(scene) = scene.as_deref_mut() {
                scene.data_mut().prepare(pass, camera, &mut lights, w, h);
                // Refresh skinned-mesh joint palettes now that world transforms
                // are propagated, before any render pass consumes them.
                scene.update_deformations();
            }

            // Phase 2: Flush - upload all batched uniforms to GPU
            MaterialManager3d::get_global_manager(|mm| mm.flush());

            // Phase 2.5: Shadow pre-pass — render scene depth from each shadow-casting
            // light into the shadow atlas before the color pass. World transforms are
            // already propagated and lights collected by `prepare`. Only meaningful
            // for the first pass; stereo passes reuse the same shadow maps.
            if let Some(scene) = scene.as_deref_mut() {
                self.shadow_mapper.render(
                    scene,
                    &*camera,
                    &lights,
                    &mut encoder,
                    &mut self.gpu_timer,
                );
            }

            // Phase 2.6: geometry G-buffer prepass (first pass only), shared by SSAO
            // and SSR. Renders view-space positions + world normal/roughness + F0/
            // metallic into the prepass MRT, then (if SSAO is on) runs the SSAO +
            // blur passes into the AO texture the opaque pass samples. SSR consumes
            // the same G-buffer after the opaque pass.
            if (self.ssao_enabled || ssr_active || dof_active || glass_active) && pass == 0 {
                if let Some(scene) = scene.as_deref_mut() {
                    let ssao = self.ssao.as_ref().unwrap();
                    let prepass_ctx = RenderContext {
                        surface_format: crate::post_processing::HDR_FORMAT,
                        sample_count: 1,
                        viewport_width: w,
                        viewport_height: h,
                        render_layers: camera.render_layers(),
                        force_no_cull: false,
                        shadow: Some(self.shadow_mapper.resources()),
                        phase: RenderPhase::Prepass,
                    };
                    {
                        let clear_color = wgpu::Operations {
                            // a = 0 marks background (no geometry).
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        };
                        let prepass_ts = self.gpu_timer.render_scope("prepass");
                        let mut pp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("gbuffer_prepass"),
                            color_attachments: &[
                                Some(wgpu::RenderPassColorAttachment {
                                    view: ssao.viewpos_view(),
                                    resolve_target: None,
                                    ops: clear_color,
                                    depth_slice: None,
                                }),
                                Some(wgpu::RenderPassColorAttachment {
                                    view: ssao.normal_view(),
                                    resolve_target: None,
                                    ops: clear_color,
                                    depth_slice: None,
                                }),
                                Some(wgpu::RenderPassColorAttachment {
                                    view: ssao.material_view(),
                                    resolve_target: None,
                                    ops: clear_color,
                                    depth_slice: None,
                                }),
                                Some(wgpu::RenderPassColorAttachment {
                                    view: ssao.ssr_params_view(),
                                    resolve_target: None,
                                    ops: clear_color,
                                    depth_slice: None,
                                }),
                            ],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: ssao.depth_view(),
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: prepass_ts,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        scene
                            .data_mut()
                            .render(0, camera, &lights, &mut pp, &prepass_ctx);
                    }
                    if self.ssao_enabled {
                        let (_, proj) = camera.view_transform_pair(0);
                        ssao.compute(&mut encoder, proj, &mut self.gpu_timer);
                    }
                }
            }

            // Phase 2.7: Clustered forward+ light culling (first pass only). Uploads
            // the overflow lights (stamped with the shadow slots the shadow pass just
            // assigned), rebuilds the cluster AABBs when the projection/viewport
            // changed, and culls lights into the clusters the object material's
            // fragment shader reads. Stereo passes reuse the result.
            if pass == 0 && Context::get().supports_clustered_lighting() {
                // Cheap copy so the clustered borrow below doesn't alias the mapper.
                let shadow_slots = self.shadow_mapper.shadow_slots().to_vec();
                let clustered = self
                    .clustered
                    .get_or_insert_with(|| crate::builtin::clustered::Clustered::new(w, h));
                let realloc = clustered.run(
                    &mut encoder,
                    &lights,
                    &shadow_slots,
                    &*camera,
                    w,
                    h,
                    &mut self.gpu_timer,
                );
                let lights_buf = clustered.lights_buffer().clone();
                let grid_buf = clustered.grid_buffer().clone();
                let index_buf = clustered.index_buffer().clone();
                let default_mat = MaterialManager3d::get_global_manager(|mm| mm.get_default());
                default_mat.borrow_mut().set_clustered_buffers(
                    &lights_buf,
                    &grid_buf,
                    &index_buf,
                    realloc,
                );
            }

            // Phase 3: Render - issue draw calls using a SINGLE render pass.
            // The scene is rasterized into the HDR film, so the render context
            // advertises the HDR format.
            {
                let render_context = RenderContext {
                    surface_format: Context::render_format(),
                    sample_count,
                    viewport_width: w,
                    viewport_height: h,
                    render_layers: camera.render_layers(),
                    force_no_cull: false,
                    shadow: Some(self.shadow_mapper.resources()),
                    phase: RenderPhase::Opaque,
                };

                // Create one render pass for all 3D scene objects
                let opaque_ts = self.gpu_timer.render_scope("opaque");
                let mut wgpu_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("scene_render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: opaque_ts,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                if let Some(scene) = scene.as_deref_mut() {
                    self.render_scene(
                        scene,
                        camera,
                        &lights,
                        pass,
                        &mut wgpu_render_pass,
                        &render_context,
                    );
                }

                // Custom renderer still needs the old interface - drop render pass first
                drop(wgpu_render_pass);

                if let Some(ref mut renderer) = renderer {
                    // Create a separate render pass for custom renderers
                    let mut custom_render_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("custom_renderer_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &color_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                    renderer.render(pass, camera, &mut custom_render_pass, &render_context);
                }
            }
        }

        // === Order-independent transparency ===
        // Transparent object surfaces are drawn in a separate weighted-blended pass
        // (McGuire & Bavoil) into the HDR pipeline's accum + revealage targets, then
        // composited over the opaque HDR scene — so transparency needs no sorting and
        // is robust to interpenetration. Points/polylines are opaque overlays already
        // drawn above. Done once (not per stereo pass).
        //
        // Skipped entirely when nothing in the scene is transparent (the common
        // case): the geometry pass clears + MSAA-resolves the accum/revealage targets
        // and the composite blends them back, all for zero draws otherwise. The
        // `has_transparent_surfaces` check uses the same per-object classification the
        // material applies, so a real transparent surface is never dropped.
        if let Some(scene) = scene
            .as_deref_mut()
            .filter(|s| s.has_transparent_surfaces())
        {
            let oit_context = RenderContext {
                surface_format: Context::render_format(),
                // The OIT geometry pass shares the (MSAA) opaque depth buffer, so its
                // targets and pipelines must use the same sample count.
                sample_count,
                viewport_width: w,
                viewport_height: h,
                render_layers: camera.render_layers(),
                force_no_cull: false,
                shadow: Some(self.shadow_mapper.resources()),
                phase: RenderPhase::Transparent,
            };
            {
                // Under MSAA the geometry pass renders into the multisampled accum/
                // revealage attachments and resolves into their single-sample copies,
                // which `composite_oit` then samples.
                let oit_accum_resolve = self.hdr.oit_accum_resolve_view();
                let oit_reveal_resolve = self.hdr.oit_reveal_resolve_view();
                let oit_ts = self.gpu_timer.render_scope("transparent");
                let mut oit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("oit_geometry_pass"),
                    color_attachments: &[
                        // accum: cleared to 0 (additive).
                        Some(wgpu::RenderPassColorAttachment {
                            view: self.hdr.oit_accum_view(),
                            resolve_target: oit_accum_resolve,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                        // revealage: cleared to 1 (nothing occluded yet).
                        Some(wgpu::RenderPassColorAttachment {
                            view: self.hdr.oit_reveal_view(),
                            resolve_target: oit_reveal_resolve,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                    ],
                    // Test against the opaque depth (the OIT pipeline does not write it).
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: oit_ts,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                scene
                    .data_mut()
                    .render(0, camera, &lights, &mut oit_pass, &oit_context);
            }
            self.hdr.composite_oit(&mut encoder, &mut self.gpu_timer);
        }

        camera.render_complete(&self.canvas);

        // Render the 2D planar scene (into the HDR film, like the 3D scene).
        {
            let context_2d = RenderContext2d {
                surface_format: Context::render_format(),
                sample_count,
                viewport_width: w,
                viewport_height: h,
            };

            // Clear material buffers for the new frame
            MaterialManager2d::get_global_manager(|mm| mm.begin_frame());

            // Prepare phase (uniform writes)
            if let Some(scene_2d) = scene_2d.as_deref_mut() {
                scene_2d.prepare(camera_2d, &context_2d);
            }

            // Flush all accumulated uniform data to GPU
            MaterialManager2d::get_global_manager(|mm| mm.flush());

            // Render phase for the 2D scene (single render pass). Skipped entirely
            // when there is no 2D scene: the pass only does a full-screen Load/Store
            // of the (MSAA) HDR film, so on a 3D-only frame it is pure wasted
            // bandwidth — and skipping it leaves the film's contents untouched,
            // exactly as a no-op Load/Store pass would.
            if let Some(scene_2d) = scene_2d {
                let scene2d_ts = self.gpu_timer.render_scope("2d");
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("2d_scene_render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: scene2d_ts,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                scene_2d
                    .data_mut()
                    .render(camera_2d, &mut render_pass, &context_2d);
            }

            // Polylines and points render on top of surfaces (into the HDR film).
            {
                let mut context_2d_encoder = RenderContext2dEncoder {
                    encoder: &mut encoder,
                    color_view: &color_view,
                    surface_format: Context::render_format(),
                    sample_count,
                    viewport_width: w,
                    viewport_height: h,
                };

                if self.polyline_renderer_2d.needs_rendering() {
                    self.polyline_renderer_2d
                        .render(camera_2d, &mut context_2d_encoder);
                }

                if self.point_renderer_2d.needs_rendering() {
                    self.point_renderer_2d
                        .render(camera_2d, &mut context_2d_encoder);
                }
            }
        }

        let (znear, zfar) = camera.clip_planes();

        // HDR resolve: the scene was rasterized into the HDR film. If MSAA is
        // active, resolve the multisampled HDR attachment into the single-sample
        // HDR texture first (a render pass resolves on End even with no draws).
        //
        // The MSAA attachment is `Discard`-stored, not `Store`d: only the resolved
        // single-sample texture is read afterwards (SSR/DoF/tonemap), so writing the
        // four multisampled samples back to memory is pure waste — and an expensive
        // one at high resolution on tile-based GPUs, where it is a full extra
        // round-trip of the 4× HDR film. The resolve_target is written regardless of
        // the store op.
        if let Some(resolve_view) = &resolve_view {
            let resolve_ts = self.gpu_timer.render_scope("resolve");
            let _resolve_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_msaa_resolve_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: Some(resolve_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Discard,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: resolve_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        // Refractive transmission (glass): snapshot the resolved opaque scene into a
        // blurred mip chain, then draw the refractive objects into the resolved scene
        // sampling that snapshot (offset by their IOR/thickness, mip by roughness).
        // Runs before SSR/DoF so glass shares their depth and gets blurred with them.
        if glass_active {
            {
                let t = self
                    .transmission
                    .get_or_insert_with(|| crate::renderer::Transmission::new(w, h));
                t.resize(w, h);
            }
            let steps = self.transmission.as_ref().unwrap().steps() as usize;
            // Collect the glass objects and sort them back-to-front (farthest first).
            let mut glass_nodes: Vec<SceneNode3d> = Vec::new();
            if let Some(s) = scene.as_deref() {
                s.collect_refractive(&mut glass_nodes);
            }
            if let Some(ssao) = self.ssao.as_ref() {
                if !glass_nodes.is_empty() {
                    let (view_pose, _proj) = camera.view_transform_pair(0);
                    let view = view_pose.to_mat4();
                    // View-space z is most negative for the farthest objects; ascending
                    // z sorts farthest-first.
                    glass_nodes.sort_by(|a, b| {
                        let za = view.transform_point3(a.world_position()).z;
                        let zb = view.transform_point3(b.world_position()).z;
                        za.partial_cmp(&zb).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let num = glass_nodes.len();
                    // Split into `min(steps, num)` depth layers, each preceded by a fresh
                    // snapshot. The farthest objects share the first layer; each of the
                    // nearest `groups-1` objects gets its own layer, so a front object
                    // refracts a snapshot that already contains the glass behind it —
                    // revealing glass through glass. `steps == 1` => one layer (refracts
                    // only the opaque scene), matching the cheap default.
                    let groups = steps.clamp(1, num);
                    let first_len = num - groups + 1;

                    let scene_resolved = self.hdr.scene_resolved_view();
                    let t = self.transmission.as_ref().unwrap();
                    let glass_ctx = RenderContext {
                        surface_format: crate::post_processing::HDR_FORMAT,
                        sample_count: 1,
                        viewport_width: w,
                        viewport_height: h,
                        render_layers: camera.render_layers(),
                        force_no_cull: false,
                        shadow: Some(self.shadow_mapper.resources()),
                        phase: RenderPhase::Transmission,
                    };
                    for g in 0..groups {
                        let lo = if g == 0 { 0 } else { first_len + (g - 1) };
                        let hi = if g == 0 { first_len } else { first_len + g };
                        // Snapshot the resolved scene as it stands (opaque + the glass
                        // layers already drawn), then draw this layer sampling it.
                        t.build(&mut encoder, scene_resolved, &mut self.gpu_timer);
                        {
                            let default_mat =
                                MaterialManager3d::get_global_manager(|mm| mm.get_default());
                            default_mat
                                .borrow_mut()
                                .set_transmission_background(Some(t.view()));
                        }
                        let glass_ts = self.gpu_timer.render_scope("transmission");
                        let mut glass_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("transmission_glass_pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: scene_resolved,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachment {
                                        view: ssao.depth_view(),
                                        depth_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Load,
                                            store: wgpu::StoreOp::Store,
                                        }),
                                        stencil_ops: None,
                                    },
                                ),
                                timestamp_writes: glass_ts,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            });
                        for node in glass_nodes[lo..hi].iter_mut() {
                            node.data_mut().render_object_only(
                                0,
                                camera,
                                &lights,
                                &mut glass_pass,
                                &glass_ctx,
                            );
                        }
                    }
                }
            }
        }

        // Screen-space reflections: now that the opaque scene is resolved into the
        // single-sample HDR texture, march the G-buffer and additively composite
        // reflections into it (before bloom/tonemap). Uses the pass-0 matrices and
        // the global skybox environment as the off-screen fallback. Skipped on
        // backends without the required support (WebGL2).
        if ssr_active {
            {
                let ssr = self
                    .ssr
                    .get_or_insert_with(|| crate::renderer::Ssr::new(w, h));
                ssr.resize(w, h);
            }
            if let Some(ssao) = self.ssao.as_ref() {
                let (view_pose, proj) = camera.view_transform_pair(0);
                let env = self.skybox.ibl_env().map(|e| crate::resource::EnvLight {
                    view: &e.view,
                    sampler: &e.sampler,
                    mip_count: e.mip_count,
                    intensity: self.skybox.intensity(),
                    rotation: self.skybox.rotation(),
                });
                let scene_resolved = self.hdr.scene_resolved_view();
                self.ssr.as_ref().unwrap().compute(
                    &mut encoder,
                    scene_resolved,
                    ssao.viewpos_view(),
                    ssao.normal_view(),
                    ssao.material_view(),
                    ssao.ssr_params_view(),
                    view_pose.to_mat4(),
                    proj,
                    env,
                    &mut self.gpu_timer,
                );
            }
        }

        // Depth of field: blur the resolved HDR scene by per-pixel circle of
        // confusion (focal distance + aperture), reading the prepass view-position
        // for depth and writing the composite back into the scene before tonemap.
        // Background pixels (no opaque surface) use the far clip plane as their
        // depth so the sky blurs like a distant surface.
        if dof_active {
            {
                let dof = self
                    .dof
                    .get_or_insert_with(|| crate::renderer::Dof::new(w, h));
                dof.resize(w, h);
            }
            if let Some(ssao) = self.ssao.as_ref() {
                let (_, proj) = camera.view_transform_pair(0);
                let scene_resolved = self.hdr.scene_resolved_view();
                self.dof.as_ref().unwrap().compute(
                    &mut encoder,
                    scene_resolved,
                    ssao.viewpos_view(),
                    proj,
                    zfar,
                    &mut self.gpu_timer,
                );
            }
        }

        // Tonemap + bloom resolve. Existing post-processing effects run on the
        // tonemapped LDR image: the HDR film is tonemapped into the LDR
        // post-process target, then the effect composites it into `frame_view`.
        // Without an effect, the HDR film is tonemapped straight into `frame_view`.
        if let Some(ref mut p) = post_processing {
            let pp_ldr_view = match &self.post_process_render_target {
                RenderTarget::Offscreen(o) => o.color_view.clone(),
                RenderTarget::Screen => frame_view.clone(),
            };
            self.hdr
                .resolve(&mut encoder, &pp_ldr_view, &mut self.gpu_timer);

            // TODO: use the real time value instead of 0.016!
            p.update(0.016, w as f32, h as f32, znear, zfar);

            let mut pp_context = PostProcessingContext {
                encoder: &mut encoder,
                output_view: &frame_view,
            };

            p.draw(&self.post_process_render_target, &mut pp_context);
        } else {
            self.hdr
                .resolve(&mut encoder, &frame_view, &mut self.gpu_timer);
        }

        // Render text
        {
            let mut context_2d_encoder = RenderContext2dEncoder {
                encoder: &mut encoder,
                color_view: &frame_view,
                surface_format: self.canvas.surface_format(),
                sample_count,
                viewport_width: w,
                viewport_height: h,
            };
            self.text_renderer
                .render(w as f32, h as f32, &mut context_2d_encoder);
        }

        // Resolve the GPU timestamp queries into a readback buffer before submit.
        self.gpu_timer.resolve(&mut encoder);

        // Submit the main command buffer (CPU-timed) and kick off the async
        // timestamp readback.
        let (_, cpu_submit) = CpuTimer::time(|| ctxt.submit(std::iter::once(encoder.finish())));
        self.gpu_timer.after_submit();

        // Render egui if enabled (uses its own command encoder and submits it)
        #[cfg(feature = "egui")]
        {
            // Close the pass opened by any draw_ui/draw_inspector calls this
            // frame so all their shapes are tessellated together.
            self.finish_egui_pass();
            self.egui_context.renderer.render(
                &frame_view,
                &depth_view,
                w,
                h,
                self.canvas.scale_factor() as f32,
            );
        }

        // Copy the rendered image into the readback texture so `snap`,
        // `snap_rect` and recording can read it back.
        match &frame {
            Some(frame) => self.canvas.copy_frame_to_readback(frame),
            None => {
                let color = self
                    .offscreen_output_target
                    .as_ref()
                    .expect("offscreen render target was just created")
                    .color_texture()
                    .expect("offscreen render target is never the screen")
                    .clone();
                self.canvas.copy_texture_to_readback(&color);
            }
        }

        // Capture frame for video recording if enabled
        #[cfg(feature = "recording")]
        self.capture_frame_if_recording();

        // Present the frame (visible windows only; a hidden window has no
        // presentable surface).
        let (_, cpu_present) = CpuTimer::time(|| {
            if let Some(frame) = frame {
                self.canvas.present(frame);
            }
        });

        // Stored before the wasm frame-pacing wait below, so `total` reflects the
        // render work and not the idle wait for the next animation frame.
        self.last_timings = Some(RenderTimings {
            renderer: "Rasterizer",
            frame_wall,
            total: cpu.elapsed(),
            cpu_submit,
            cpu_present,
            gpu_steps: self.gpu_timer.last(),
        });

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use web_sys::wasm_bindgen::closure::Closure;

            if let Some(window) = web_sys::window() {
                let (s, r) = oneshot::channel();

                let closure = Closure::once(move || s.send(()).unwrap());

                window
                    .request_animation_frame(closure.as_ref().unchecked_ref())
                    .unwrap();

                r.await.unwrap();
            }
        }

        !self.should_close()
    }

    /// Renders one path-traced frame.
    ///
    /// This bypasses the rasterizer entirely: it collects lights and propagates
    /// world transforms (via the scene's `prepare` pass), then drives the
    /// [`RayTracer`] to dispatch the path-tracing pass into its HDR accumulation
    /// buffer and tonemap the result into the frame's output view. Text overlays
    /// are still rendered on top.
    pub(super) async fn raytrace_3d_frame(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut dyn Camera3d,
        raytracer: &mut RayTracer,
    ) -> bool {
        // Wall-clock frame-to-frame period (true FPS), the metric the per-pass GPU
        // timestamps don't capture. See `render_single_frame`.
        let frame_start = web_time::Instant::now();
        let frame_wall = self
            .last_frame_instant
            .map(|prev| frame_start.duration_since(prev))
            .unwrap_or_default();
        self.last_frame_instant = Some(frame_start);
        let cpu = CpuTimer::start();
        self.gpu_timer.begin_frame();
        let offscreen = self.hidden;

        let frame = if offscreen {
            None
        } else {
            match self.acquire_next_frame() {
                Some(frame) => Some(frame),
                None => return !self.should_close(),
            }
        };

        let w = self.width();
        let h = self.height();

        camera.handle_event(&self.canvas, &WindowEvent::FramebufferSize(w, h));
        camera.update(&self.canvas);

        let sample_count = if offscreen {
            1
        } else {
            self.canvas.sample_count()
        };

        let ctxt = Context::get();
        let mut encoder = ctxt.create_command_encoder(Some("kiss3d_raytrace_encoder"));

        // The path tracer renders single-sampled into an offscreen color texture
        // when the window is hidden; otherwise straight into the surface.
        if offscreen {
            if self.offscreen_output_target.is_none() {
                self.offscreen_output_target =
                    Some(self.framebuffer_manager.new_render_target(w, h, true));
            }
            self.offscreen_output_target.as_mut().unwrap().resize(
                w,
                h,
                self.canvas.surface_format(),
            );
        }

        let frame_view = match &frame {
            Some(frame) => frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            None => self
                .offscreen_output_target
                .as_ref()
                .expect("offscreen render target was just created")
                .color_view()
                .expect("offscreen render target is never the screen")
                .clone(),
        };

        // Collect lights and propagate world transforms for the path tracer.
        // (`prepare` does both; the path tracer reads geometry off the CPU side.)
        let mut lights = LightCollection::with_ambient(self.ambient_intensity);
        lights.ambient_color = self.ambient_color;
        lights.fog = self.fog;
        scene.data_mut().prepare(0, camera, &mut lights, w, h);
        // Refresh skinned-mesh joint palettes so the path tracer gathers the
        // animated (CPU-skinned) geometry, not the bind pose.
        scene.update_deformations();

        // Exposure and tonemap operator are shared with the rasterizer, so the
        // path tracer finishes the image with the window's `HdrSettings`.
        let hdr = self.hdr_settings();
        let exposure = hdr.exposure;
        let tonemap_operator = hdr.tonemap.as_u32();
        // Feed the window's skybox to the path tracer as its environment, so a
        // skybox set on the window also lights and backgrounds the ray-traced
        // view (the tracer's own environment, if set, still takes precedence).
        let skybox = if self.skybox.is_set() {
            Some((
                self.skybox.environment(),
                self.skybox.rotation(),
                self.skybox.intensity(),
                self.skybox.generation(),
            ))
        } else {
            None
        };

        // Depth of field: the path tracer shares the rasterizer's DoF settings rather
        // than a separate aperture. Convert the window's DoF (focal distance, aperture
        // f-stops, sensor height + the camera's vertical FOV) into the tracer's
        // thin-lens radius + focus distance, matching the rasterizer's
        // circle-of-confusion model (see `circle_of_confusion` in dof.wgsl). Disabled
        // DoF gives a pinhole camera. Re-applied every frame; `set_aperture` only
        // resets accumulation when the derived values actually change.
        let (lens_radius, focus_distance) = if self.dof_enabled {
            let dof = self.dof.as_ref().map(|d| *d.settings()).unwrap_or_default();
            // proj[1][1] = cot(fov_y / 2): the rasterizer derives the focal length
            // from the sensor height and this term.
            let proj11 = camera.view_transform_pair(0).1.to_cols_array()[5];
            let focal_length = 0.5 * dof.sensor_height * proj11;
            let aperture_d = focal_length / dof.aperture_f_stops.max(1e-3);
            let denom = (dof.focal_distance - focal_length).max(1e-4);
            (
                0.5 * aperture_d * dof.focal_distance / denom,
                dof.focal_distance,
            )
        } else {
            (0.0, 1.0)
        };
        raytracer.set_aperture(lens_radius, focus_distance);

        // The path tracer records its own GPU phases (trace / denoise / tonemap)
        // into the GPU timer.
        raytracer.render_frame(
            scene,
            camera,
            &lights,
            self.background,
            skybox,
            &mut encoder,
            &frame_view,
            w,
            h,
            exposure,
            tonemap_operator,
            &mut self.gpu_timer,
        );

        // Render text on top of the path-traced image.
        {
            let mut context_2d_encoder = RenderContext2dEncoder {
                encoder: &mut encoder,
                color_view: &frame_view,
                surface_format: self.canvas.surface_format(),
                sample_count,
                viewport_width: w,
                viewport_height: h,
            };
            self.text_renderer
                .render(w as f32, h as f32, &mut context_2d_encoder);
        }

        // Resolve GPU timestamps before submit, then submit (CPU-timed) and kick
        // off the async readback.
        self.gpu_timer.resolve(&mut encoder);
        let (_, cpu_submit) = CpuTimer::time(|| ctxt.submit(std::iter::once(encoder.finish())));
        self.gpu_timer.after_submit();

        // Render egui on top of the path-traced image (uses its own encoder).
        // The depth view is unused by egui, so the color view is passed twice.
        #[cfg(feature = "egui")]
        {
            // Close the pass opened by any draw_ui/draw_inspector calls this
            // frame so all their shapes are tessellated together.
            self.finish_egui_pass();
            self.egui_context.renderer.render(
                &frame_view,
                &frame_view,
                w,
                h,
                self.canvas.scale_factor() as f32,
            );
        }

        match &frame {
            Some(frame) => self.canvas.copy_frame_to_readback(frame),
            None => {
                let color = self
                    .offscreen_output_target
                    .as_ref()
                    .expect("offscreen render target was just created")
                    .color_texture()
                    .expect("offscreen render target is never the screen")
                    .clone();
                self.canvas.copy_texture_to_readback(&color);
            }
        }

        #[cfg(feature = "recording")]
        self.capture_frame_if_recording();

        let (_, cpu_present) = CpuTimer::time(|| {
            if let Some(frame) = frame {
                self.canvas.present(frame);
            }
        });

        self.last_timings = Some(RenderTimings {
            renderer: "Path tracer",
            frame_wall,
            total: cpu.elapsed(),
            cpu_submit,
            cpu_present,
            gpu_steps: self.gpu_timer.last(),
        });

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use web_sys::wasm_bindgen::closure::Closure;

            if let Some(window) = web_sys::window() {
                let (s, r) = oneshot::channel();
                let closure = Closure::once(move || s.send(()).unwrap());
                window
                    .request_animation_frame(closure.as_ref().unchecked_ref())
                    .unwrap();
                r.await.unwrap();
            }
        }

        !self.should_close()
    }

    /// Acquires the surface texture for the next frame.
    ///
    /// Returns `None` when no frame is available and the caller should skip
    /// rendering. Until the first frame has been rendered, this retries —
    /// pumping window events between attempts — for up to a couple of seconds,
    /// because a freshly created window may need the event loop to run a few
    /// times before its surface becomes presentable. Once a frame has been
    /// acquired, a later failure (e.g. a minimized window) skips the frame
    /// immediately instead of stalling.
    fn acquire_next_frame(&mut self) -> Option<wgpu::SurfaceTexture> {
        if let Some(frame) = self.canvas.get_current_texture() {
            self.first_frame = false;
            return Some(frame);
        }

        // The window has rendered before: treat this as a transient failure
        // and skip the frame without stalling.
        if !self.first_frame {
            return None;
        }

        #[cfg(target_arch = "wasm32")]
        return None;

        #[cfg(not(target_arch = "wasm32"))]
        {
            let deadline = std::time::Instant::now() + STARTUP_SURFACE_TIMEOUT;
            loop {
                std::thread::sleep(SURFACE_RETRY_INTERVAL);
                self.canvas.poll_events();

                if let Some(frame) = self.canvas.get_current_texture() {
                    self.first_frame = false;
                    return Some(frame);
                }

                if std::time::Instant::now() >= deadline {
                    log::warn!(
                        "could not acquire a surface texture within \
                         {STARTUP_SURFACE_TIMEOUT:?}; the window failed to become ready"
                    );
                    return None;
                }
            }
        }
    }

    fn render_scene(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut dyn Camera3d,
        lights: &LightCollection,
        pass: usize,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext,
    ) {
        // Render points
        self.point_renderer
            .render(pass, camera, render_pass, context);

        // Render polylines (lines with configurable width)
        self.polyline_renderer
            .render(pass, camera, render_pass, context);

        // Render scene graph (surfaces and wireframes are handled by ObjectMaterial)
        scene
            .data_mut()
            .render(pass, camera, lights, render_pass, context);
    }

    /// Renders every planar reflector (mirror) in the scene: for each reflector
    /// surface, render the scene from a mirror camera into the reflector's own
    /// texture and store the reflected view-projection so the surface samples it
    /// during the main pass. See [`crate::renderer::reflector`].
    fn render_reflectors(
        &mut self,
        scene: Option<&mut SceneNode3d>,
        camera: &mut dyn Camera3d,
        w: u32,
        h: u32,
    ) {
        let scene = match scene {
            Some(s) => s,
            None => return,
        };

        // Cheap detection first — skip the extra prepare when there are no mirrors.
        let mut any = false;
        scene.apply_to_objects_recursive(&mut |obj| {
            if obj.reflector().is_some() {
                any = true;
            }
        });
        if !any {
            return;
        }

        let ctxt = Context::get();
        let (znear, zfar) = camera.clip_planes();
        let (mview, mproj) = camera.view_transform_pair(0);
        let eye = camera.eye();
        let sky_set = self.skybox.is_set();

        // Propagate world transforms so we can read each reflector's world plane.
        MaterialManager3d::get_global_manager(|mm| mm.begin_frame());
        let mut lights = LightCollection::with_ambient(self.ambient_intensity);
        lights.ambient_color = self.ambient_color;
        lights.fog = self.fog;
        scene.data_mut().prepare(0, camera, &mut lights, w, h);
        scene.update_deformations();

        // Build a mirror camera per reflector and (in the same walk) resize its
        // target + store the reflected view-proj on it. Collect the views + clip
        // plane so the renders below don't need to re-borrow the scene's reflectors.
        let mut jobs: Vec<(
            crate::renderer::MirrorCamera,
            wgpu::TextureView,
            wgpu::TextureView,
            [f32; 4],
        )> = Vec::new();
        scene.apply_to_objects_with_world_mut_recursive(&mut |pose, _scale, obj| {
            let local_n = match obj.reflector() {
                Some(r) => r.local_normal(),
                None => return,
            };
            // World plane: origin = node position; normal = node rotation * local
            // normal, oriented toward the camera so we keep the front half-space.
            let point = pose.translation;
            let mut normal = (pose.rotation * local_n).normalize();
            if normal.dot(eye - point) < 0.0 {
                normal = -normal;
            }
            let mcam =
                crate::renderer::MirrorCamera::new(mview, mproj, eye, znear, zfar, point, normal);
            let view_proj = mcam.reflector_view_proj();
            let clip = [normal.x, normal.y, normal.z, -normal.dot(point)];

            let r = obj.reflector_mut().unwrap();
            r.resize(w, h);
            r.set_view_proj(view_proj);
            jobs.push((mcam, r.color_view().clone(), r.depth_view().clone(), clip));
        });

        // Fixed-light path for the capture frames (the mirror camera has no clustered
        // cull data). Reflector surfaces skip themselves during capture (handled in
        // the material via `capture_mode`).
        MaterialManager3d::get_global_manager(|mm| mm.get_default())
            .borrow_mut()
            .set_capture_mode(true);

        for (mut mcam, color_view, depth_view, clip) in jobs {
            // Clip geometry behind this mirror's plane.
            MaterialManager3d::get_global_manager(|mm| mm.get_default())
                .borrow_mut()
                .set_clip_plane(Some(clip));

            // Prepare + flush the scene for the mirror camera.
            MaterialManager3d::get_global_manager(|mm| mm.begin_frame());
            let mut mlights = LightCollection::with_ambient(self.ambient_intensity);
            mlights.ambient_color = self.ambient_color;
            mlights.fog = self.fog;
            scene.data_mut().prepare(0, &mut mcam, &mut mlights, w, h);
            scene.update_deformations();
            MaterialManager3d::get_global_manager(|mm| mm.flush());

            // Render into this reflector's target (its own queue submission).
            let mut menc = ctxt.create_command_encoder(Some("reflector_encoder"));
            if sky_set {
                self.skybox.render(
                    &mut menc,
                    &color_view,
                    1,
                    mcam.inverse_transformation(),
                    None,
                );
            }
            let ctx = RenderContext {
                surface_format: crate::post_processing::HDR_FORMAT,
                sample_count: 1,
                viewport_width: w,
                viewport_height: h,
                render_layers: camera.render_layers(),
                // The reflected projection flips winding, so disable back-face cull.
                force_no_cull: true,
                shadow: Some(self.shadow_mapper.resources()),
                phase: RenderPhase::Opaque,
            };
            {
                let load = if sky_set {
                    wgpu::LoadOp::Load
                } else {
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                };
                let reflector_ts = self.gpu_timer.render_scope("reflector");
                let mut pass = menc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("reflector_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: reflector_ts,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                scene
                    .data_mut()
                    .render(0, &mut mcam, &mlights, &mut pass, &ctx);
            }
            ctxt.submit(std::iter::once(menc.finish()));
        }

        // Restore the default material's state.
        {
            let default_mat = MaterialManager3d::get_global_manager(|mm| mm.get_default());
            let mut m = default_mat.borrow_mut();
            m.set_capture_mode(false);
            m.set_clip_plane(None);
        }

        // Bump the frame counter so the per-pass loop re-prepares the main camera
        // (reflector objects then pick up the view-proj set on their Reflector above).
        MaterialManager3d::get_global_manager(|mm| mm.begin_frame());
    }
}
