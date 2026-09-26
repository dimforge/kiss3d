//! A renderer for egui UI using wgpu.

use crate::context::Context;
use egui::{Context as EguiContext, RawInput};
use egui_wgpu::RendererOptions;

/// Structure which manages the egui UI rendering.
pub struct EguiRenderer {
    egui_ctx: EguiContext,
    renderer: egui_wgpu::Renderer,
    shapes: Vec<egui::epaint::ClippedShape>,
    textures_delta: egui::TexturesDelta,
    cursor: egui::CursorIcon,
    retain_shapes: bool,
}

impl EguiRenderer {
    /// Creates a new egui renderer.
    pub fn new() -> EguiRenderer {
        let egui_ctx = EguiContext::default();

        // Load fonts manually - use kiss3d's embedded font
        let mut fonts = egui::FontDefinitions::default();

        // Add WorkSans font from kiss3d
        fonts.font_data.insert(
            "WorkSans".to_owned(),
            egui::FontData::from_static(include_bytes!("../text/WorkSans-Regular.ttf")).into(),
        );

        // Set it as the proportional font
        fonts
            .families
            .get_mut(&egui::FontFamily::Proportional)
            .unwrap()
            .insert(0, "WorkSans".to_owned());

        // Set it as the monospace font too
        fonts
            .families
            .get_mut(&egui::FontFamily::Monospace)
            .unwrap()
            .insert(0, "WorkSans".to_owned());

        egui_ctx.set_fonts(fonts);

        // Set default pixels_per_point to avoid DPI warnings.
        // Not using 1.0 exactly so that draw_ui() gets a chance
        // to initialize it to the actual value (which might be 1)
        // and trigger a redraw.
        egui_ctx.set_pixels_per_point(0.987);

        // Run a dummy frame to initialize fonts with correct DPI
        let dummy_input = RawInput::default();
        egui_ctx.begin_pass(dummy_input);
        let mut dummy_output = egui_ctx.end_pass();

        let ctxt = Context::get();

        // Create the egui-wgpu renderer
        let mut renderer = egui_wgpu::Renderer::new(
            &ctxt.device,
            ctxt.surface_format,
            RendererOptions {
                msaa_samples: 1,
                dithering: true,
                ..Default::default()
            },
        );

        // Apply textures from the dummy pass (font textures, etc.)
        for (id, image_deltas) in &dummy_output.textures_delta.set {
            for image_delta in image_deltas {
                renderer.update_texture(&ctxt.device, &ctxt.queue, *id, image_delta);
            }
        }
        // `TexturesDelta` asserts on drop that every delta was handled.
        dummy_output.textures_delta.clear();

        EguiRenderer {
            egui_ctx,
            renderer,
            shapes: Vec::new(),
            textures_delta: Default::default(),
            cursor: egui::CursorIcon::Default,
            retain_shapes: false,
        }
    }

    /// Get a mutable reference to the egui Context.
    pub fn context_mut(&mut self) -> &mut EguiContext {
        &mut self.egui_ctx
    }

    /// Get a reference to the egui Context.
    pub fn context(&self) -> &EguiContext {
        &self.egui_ctx
    }

    /// Begin a new frame with the given raw input.
    pub fn begin_frame(&mut self, raw_input: RawInput) {
        // The pass about to run replaces last pass's shapes, so this is where
        // they are dropped once they have been rendered too (see
        // `set_retain_shapes`).
        self.shapes.clear();
        self.egui_ctx.begin_pass(raw_input);
    }

    /// Whether a pass's shapes survive being rendered, so a frame that runs no
    /// pass at all draws the last one again. Off by default.
    ///
    /// Turning it on lets a host run its UI only on the frames something
    /// changed. It also means a UI stops being drawn only when the next pass
    /// draws something else: a caller that shows its UI conditionally wants
    /// this off, or has to call [`Self::clear_shapes`] when it hides it.
    pub fn set_retain_shapes(&mut self, retain: bool) {
        self.retain_shapes = retain;
    }

    /// Whether rendered shapes are kept for the frames that run no pass.
    pub fn retains_shapes(&self) -> bool {
        self.retain_shapes
    }

    /// Drops the shapes of the last pass, so a retaining host stops drawing
    /// the UI it last built. A no-op unless [`Self::set_retain_shapes`] is on,
    /// since otherwise rendering them already dropped them.
    pub fn clear_shapes(&mut self) {
        self.shapes.clear();
    }

    /// What the last pass asked the pointer to look like over the widget it
    /// was on. `Default` whenever nothing under it asked for anything else.
    pub fn cursor(&self) -> egui::CursorIcon {
        self.cursor
    }

    /// End the current frame and prepare for rendering.
    pub fn end_frame(&mut self) {
        let output = self.egui_ctx.end_pass();
        self.cursor = output.platform_output.cursor_icon;
        self.shapes = output.shapes;
        // Append rather than replace: if a previous frame's render was skipped
        // (e.g. failed to acquire surface texture), we must not lose its texture
        // deltas (such as the font atlas glyph upload).
        self.textures_delta.append(output.textures_delta);
    }

    /// Throw the open pass away and begin it again with `raw_input`, as
    /// `Context::run` does when a pass asks to be discarded: its shapes are
    /// dropped, its texture uploads kept (the font atlas may have grown), and
    /// its pass count carried over so egui knows how many it has run.
    pub fn rerun_frame(&mut self, raw_input: RawInput) {
        let output = self.egui_ctx.end_pass();
        self.textures_delta.append(output.textures_delta);
        let passes = output.platform_output.num_completed_passes;
        // `end_pass` took the viewport's output; `will_discard` reads the
        // count back off the fresh one.
        self.egui_ctx
            .output_mut(|output| output.num_completed_passes = passes);
        self.egui_ctx.begin_pass(raw_input);
    }

    /// Registers a native wgpu texture view with egui, returning a
    /// [`egui::TextureId`] that `ui.image((id, size))` can draw — no CPU copy
    /// involved. The texture stays registered until
    /// [`Self::unregister_native_texture`].
    pub fn register_native_texture(
        &mut self,
        view: &wgpu::TextureView,
        filter: wgpu::FilterMode,
    ) -> egui::TextureId {
        let ctxt = Context::get();
        self.renderer
            .register_native_texture(&ctxt.device, view, filter)
    }

    /// Frees a texture id previously returned by
    /// [`Self::register_native_texture`].
    pub fn unregister_native_texture(&mut self, id: egui::TextureId) {
        self.renderer.free_texture(&id);
    }

    /// Returns true if egui wants to capture the mouse (e.g., hovering over a widget).
    ///
    /// This deliberately does not use egui's own `egui_wants_pointer_input`. That
    /// one treats the whole background layer as egui's whenever
    /// `root_ui_available_rect` is unset, and egui only fills that rect in from
    /// its closure-based `Context::run_ui`; a `begin_pass`/`end_pass` integration
    /// like ours always leaves it unset. Up to egui 0.35 a legacy fallback still
    /// gave the right answer, but 0.36 dropped it, so egui started claiming every
    /// hover and scroll over the empty 3D viewport and the camera stopped seeing
    /// them.
    pub fn wants_pointer_input(&self) -> bool {
        egui_captures_pointer(&self.egui_ctx)
    }

    /// Returns true if egui wants to capture keyboard input (e.g., text input focused).
    pub fn wants_keyboard_input(&self) -> bool {
        self.egui_ctx.egui_wants_keyboard_input()
    }

    /// Actually renders the UI.
    pub fn render(
        &mut self,
        color_view: &wgpu::TextureView,
        _depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        scale_factor: f32,
    ) {
        let ctxt = Context::get();

        // Update textures
        for (id, image_deltas) in &self.textures_delta.set {
            for image_delta in image_deltas {
                self.renderer
                    .update_texture(&ctxt.device, &ctxt.queue, *id, image_delta);
            }
        }

        // Prepare clipped primitives
        let clipped_primitives = self.egui_ctx.tessellate(self.shapes.clone(), scale_factor);

        // Create screen descriptor
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: scale_factor,
        };

        // Create our own encoder for egui rendering to avoid lifetime issues
        let mut encoder = ctxt.create_command_encoder(Some("egui_command_encoder"));

        // Update buffers
        self.renderer.update_buffers(
            &ctxt.device,
            &ctxt.queue,
            &mut encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        // Render
        {
            // egui doesn't need depth testing - it renders 2D overlays
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                multiview_mask: None,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // egui-wgpu requires 'static lifetime, so we use forget_lifetime
            // SAFETY: The render pass will be dropped before the encoder is finished,
            // and we don't use the encoder for anything else after this.
            let mut render_pass = render_pass.forget_lifetime();

            self.renderer
                .render(&mut render_pass, &clipped_primitives, &screen_descriptor);
        }

        // Submit the egui commands
        ctxt.submit(std::iter::once(encoder.finish()));

        // Free textures
        for id in &self.textures_delta.free {
            self.renderer.free_texture(id);
        }

        self.textures_delta.clear();
        if !self.retain_shapes {
            self.shapes.clear();
        }
    }
}

impl Default for EguiRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for EguiRenderer {
    fn drop(&mut self) {
        // `TexturesDelta` asserts on drop that every delta was handled. Deltas can
        // still be pending here if the last frame ended without being rendered.
        self.textures_delta.clear();
    }
}

/// Whether egui should consume pointer events, given the current state of `ctx`.
fn egui_captures_pointer(ctx: &EguiContext) -> bool {
    // Actively driving a widget (dragging a slider, holding a button down): egui
    // owns the pointer wherever it goes.
    if ctx.egui_is_using_pointer() {
        return true;
    }

    // A drag that began outside egui stays with the app for its whole duration,
    // even if it wanders over a window.
    if ctx.input(|i| i.pointer.any_down()) {
        return false;
    }

    // Merely hovering: capture only over an actual egui area. Everything egui
    // draws on its own layers (windows, popups, menus, tooltips) sits above
    // `Order::Background`, which is the catch-all covering the whole viewport.
    ctx.input(|i| i.pointer.interact_pos())
        .and_then(|pos| ctx.layer_id_at(pos))
        .is_some_and(|layer| layer.order != egui::Order::Background)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCREEN: egui::Vec2 = egui::vec2(800.0, 600.0);
    const WINDOW_POS: egui::Pos2 = egui::pos2(10.0, 10.0);
    const OVER_WINDOW: egui::Pos2 = egui::pos2(60.0, 60.0);
    const OVER_VIEWPORT: egui::Pos2 = egui::pos2(600.0, 500.0);

    /// Drives a context the way `EguiRenderer` does (manual `begin_pass` /
    /// `end_pass`, UI drawn as an `egui::Window`) and reports whether egui would
    /// swallow the pointer.
    fn captures_at(pointer: egui::Pos2, button_down: bool) -> bool {
        let ctx = EguiContext::default();
        ctx.set_pixels_per_point(1.0);

        // A few passes, so the layer rects from an earlier pass are available.
        for _ in 0..3 {
            let mut events = vec![egui::Event::PointerMoved(pointer)];
            if button_down {
                events.push(egui::Event::PointerButton {
                    pos: pointer,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                });
            }

            ctx.begin_pass(RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, SCREEN)),
                events,
                ..Default::default()
            });
            egui::Window::new("panel")
                .fixed_pos(WINDOW_POS)
                .fixed_size(egui::vec2(200.0, 120.0))
                .show(&ctx, |ui| {
                    ui.label("hello");
                    let _ = ui.button("click me");
                });
            ctx.end_pass().textures_delta.clear();
        }

        egui_captures_pointer(&ctx)
    }

    // egui 0.36 dropped the legacy fallback in `egui_wants_pointer_input`, which
    // made it claim the pointer anywhere over the background layer. That starved
    // the cameras of every hover and scroll event over the 3D viewport: zoom went
    // dead and `last_cursor_pos` only advanced during drags, so each new click
    // teleported the camera.
    #[test]
    fn viewport_pointer_is_left_to_the_camera() {
        assert!(!captures_at(OVER_VIEWPORT, false));
        assert!(!captures_at(OVER_VIEWPORT, true));
    }

    #[test]
    fn pointer_over_a_widget_goes_to_egui() {
        assert!(captures_at(OVER_WINDOW, false));
        assert!(captures_at(OVER_WINDOW, true));
    }
}
