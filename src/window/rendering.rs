//! Rendering functionality.

#![allow(clippy::await_holding_refcell_ref)]

use crate::camera::{Camera2d, Camera3d, FixedView3d};
use crate::context::Context;
use crate::event::WindowEvent;
use crate::light::LightCollection;
use crate::post_processing::{PostProcessingContext, PostProcessingEffect};
use crate::prelude::FixedView2d;
use crate::renderer::{RayTracer, Renderer3d};
use crate::resource::{
    MaterialManager2d, MaterialManager3d, RenderContext, RenderContext2d, RenderContext2dEncoder,
    RenderTarget,
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

    pub async fn render(
        &mut self,
        scene: Option<&mut SceneNode3d>,
        scene_2d: Option<&mut SceneNode2d>,
        camera: Option<&mut dyn Camera3d>,
        camera_2d: Option<&mut dyn Camera2d>,
        renderer: Option<&mut dyn Renderer3d>,
        post_processing: Option<&mut dyn PostProcessingEffect>,
    ) -> bool {
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

        // Resize the offscreen render targets if needed.
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

        // Determine which views the scene is rendered into.
        let (color_view, depth_view) = if post_processing.is_some() {
            // Render to the offscreen buffer for post-processing.
            match &self.post_process_render_target {
                RenderTarget::Offscreen(o) => (o.color_view.clone(), o.depth_view.clone()),
                // Shouldn't happen, but fall back to the final view.
                RenderTarget::Screen => (frame_view.clone(), self.canvas.depth_view().clone()),
            }
        } else if offscreen {
            let o = self
                .offscreen_output_target
                .as_ref()
                .expect("offscreen render target was just created");
            (
                frame_view.clone(),
                o.depth_view()
                    .expect("offscreen render target is never the screen")
                    .clone(),
            )
        } else {
            (frame_view.clone(), self.canvas.depth_view().clone())
        };

        // Clear the render target at the start of the frame
        {
            let bg = self.background;
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
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // Render pass is dropped here, ending the clear pass
        }

        // Signal start of new frame to all materials (for dynamic buffer clearing)
        MaterialManager3d::get_global_manager(|mm| mm.begin_frame());

        // Create a light collection for this frame
        let mut lights = LightCollection::with_ambient(self.ambient_intensity);

        // Render the 3D scene using two-phase rendering
        for pass in 0usize..camera.num_passes() {
            camera.start_pass(pass, &self.canvas);

            // Phase 1: Prepare - collect uniforms in CPU memory and gather lights from scene
            if let Some(scene) = scene.as_deref_mut() {
                scene.data_mut().prepare(pass, camera, &mut lights, w, h);
            }

            // Phase 2: Flush - upload all batched uniforms to GPU
            MaterialManager3d::get_global_manager(|mm| mm.flush());

            // Phase 3: Render - issue draw calls using a SINGLE render pass
            {
                let render_context = RenderContext {
                    surface_format: self.canvas.surface_format(),
                    sample_count,
                    viewport_width: w,
                    viewport_height: h,
                };

                // Create one render pass for all 3D scene objects
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
                    timestamp_writes: None,
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

        camera.render_complete(&self.canvas);

        // Render the 2D planar scene
        {
            let context_2d = RenderContext2d {
                surface_format: self.canvas.surface_format(),
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

            // Render phase for scene (single render pass)
            {
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
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                if let Some(scene_2d) = scene_2d {
                    scene_2d
                        .data_mut()
                        .render(camera_2d, &mut render_pass, &context_2d);
                }
            }

            // Polylines and points render on top of surfaces
            {
                let mut context_2d_encoder = RenderContext2dEncoder {
                    encoder: &mut encoder,
                    color_view: &color_view,
                    surface_format: self.canvas.surface_format(),
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

        // Apply post-processing if enabled
        if let Some(ref mut p) = post_processing {
            // TODO: use the real time value instead of 0.016!
            p.update(0.016, w as f32, h as f32, znear, zfar);

            let mut pp_context = PostProcessingContext {
                encoder: &mut encoder,
                output_view: &frame_view,
            };

            p.draw(&self.post_process_render_target, &mut pp_context);
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

        // Submit the main command buffer
        ctxt.submit(std::iter::once(encoder.finish()));

        // Render egui if enabled (uses its own command encoder and submits it)
        #[cfg(feature = "egui")]
        {
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
        if let Some(frame) = frame {
            self.canvas.present(frame);
        }
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
    pub(super) async fn render_raytraced_frame(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut dyn Camera3d,
        raytracer: &mut RayTracer,
    ) -> bool {
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

        let sample_count = if offscreen { 1 } else { self.canvas.sample_count() };

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
        scene.data_mut().prepare(0, camera, &mut lights, w, h);

        raytracer.render_frame(scene, camera, &lights, &mut encoder, &frame_view, w, h);

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

        ctxt.submit(std::iter::once(encoder.finish()));

        // Render egui on top of the path-traced image (uses its own encoder).
        // The depth view is unused by egui, so the color view is passed twice.
        #[cfg(feature = "egui")]
        {
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

        if let Some(frame) = frame {
            self.canvas.present(frame);
        }
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
}
