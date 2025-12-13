//! A renderer for egui UI using wgpu.

use crate::context::Context;
use egui::{Context as EguiContext, RawInput};

/// Structure which manages the egui UI rendering.
pub struct EguiRenderer {
    egui_ctx: EguiContext,
    renderer: egui_wgpu::Renderer,
    shapes: Vec<egui::epaint::ClippedShape>,
    textures_delta: egui::TexturesDelta,
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
        let _ = egui_ctx.end_pass();

        let ctxt = Context::get();

        // Create the egui-wgpu renderer
        let renderer = egui_wgpu::Renderer::new(
            &ctxt.device,
            ctxt.surface_format,
            Some(Context::depth_format()),
            1, // sample count
            true, // dithering
        );

        EguiRenderer {
            egui_ctx,
            renderer,
            shapes: Vec::new(),
            textures_delta: Default::default(),
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
        self.egui_ctx.begin_pass(raw_input);
    }

    /// End the current frame and prepare for rendering.
    pub fn end_frame(&mut self) {
        let output = self.egui_ctx.end_pass();
        self.shapes = output.shapes;
        self.textures_delta = output.textures_delta;
    }

    /// Returns true if egui wants to capture the mouse (e.g., hovering over a widget).
    pub fn wants_pointer_input(&self) -> bool {
        self.egui_ctx.wants_pointer_input()
    }

    /// Returns true if egui wants to capture keyboard input (e.g., text input focused).
    pub fn wants_keyboard_input(&self) -> bool {
        self.egui_ctx.wants_keyboard_input()
    }

    /// Actually renders the UI.
    pub fn render(
        &mut self,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        scale_factor: f32,
    ) {
        let ctxt = Context::get();

        // Update textures
        for (id, image_delta) in &self.textures_delta.set {
            self.renderer
                .update_texture(&ctxt.device, &ctxt.queue, *id, image_delta);
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
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
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
    }
}

impl Default for EguiRenderer {
    fn default() -> Self {
        Self::new()
    }
}
