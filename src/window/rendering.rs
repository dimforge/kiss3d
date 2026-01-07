//! Rendering functionality.

#![allow(clippy::await_holding_refcell_ref)]

use crate::camera::{Camera2d, Camera3d, FixedView3d};
use crate::context::Context;
use crate::event::WindowEvent;
use crate::light::LightCollection;
use crate::post_processing::{PostProcessingContext, PostProcessingEffect};
use crate::prelude::FixedView2d;
use crate::renderer::Renderer3d;
use crate::resource::{
    MaterialManager2d, MaterialManager3d, RenderContext, RenderContext2d, RenderContext2dEncoder,
    RenderTarget,
};
use crate::scene::{SceneNode2d, SceneNode3d};

use super::Window;

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
        let w = self.width();
        let h = self.height();

        camera_2d.handle_event(&self.canvas, &WindowEvent::FramebufferSize(w, h));
        camera.handle_event(&self.canvas, &WindowEvent::FramebufferSize(w, h));
        camera_2d.update(&self.canvas);
        camera.update(&self.canvas);

        // No need to update the light position here - it's computed per-frame
        // in the material's prepare() based on the camera position

        // Get the surface texture
        let frame = match self.canvas.get_current_texture() {
            Ok(frame) => frame,
            Err(e) => {
                eprintln!("Failed to acquire surface texture: {:?}", e);
                return !self.should_close();
            }
        };
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let ctxt = Context::get();
        let mut encoder = ctxt.create_command_encoder(Some("kiss3d_frame_encoder"));

        // Resize post-process render target if needed
        self.post_process_render_target
            .resize(w, h, self.canvas.surface_format());

        // Determine which views to render to
        let (color_view, depth_view) = if post_processing.is_some() {
            // Render to offscreen buffer for post-processing
            match &self.post_process_render_target {
                RenderTarget::Offscreen(o) => (&o.color_view, &o.depth_view),
                RenderTarget::Screen => {
                    // Shouldn't happen, but fallback to main view
                    (&frame_view, self.canvas.depth_view())
                }
            }
        } else {
            (&frame_view, self.canvas.depth_view())
        };
        let (color_view, depth_view) = (color_view.clone(), depth_view.clone());

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
                    depth_slice: None
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
                    sample_count: self.canvas.sample_count(),
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
                        depth_slice: None
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
                                depth_slice: None
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
                            occlusion_query_set: None
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
                sample_count: self.canvas.sample_count(),
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
                        depth_slice: None
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
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
                    sample_count: self.canvas.sample_count(),
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
                sample_count: self.canvas.sample_count(),
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

        // Copy frame to readback texture for snap/snap_rect functionality
        self.canvas.copy_frame_to_readback(&frame);

        // Capture frame for video recording if enabled
        #[cfg(feature = "recording")]
        self.capture_frame_if_recording();

        // Present the frame
        self.canvas.present(frame);
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
