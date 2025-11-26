//! A renderer for egui UI.

use crate::context::{Context, VertexArray};
use egui::{Context as EguiContext, RawInput};
use egui_glow::Painter;
use std::sync::Arc;

/// Structure which manages the egui UI rendering.
pub struct EguiRenderer {
    egui_ctx: EguiContext,
    painter: Painter,
    shapes: Vec<egui::epaint::ClippedShape>,
    textures_delta: egui::TexturesDelta,
    kiss3d_vao: Option<VertexArray>,
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
        // TODO: could we avoid this dummy pass completely instead?
        egui_ctx.set_pixels_per_point(0.987);

        // Run a dummy frame to initialize fonts with correct DPI
        let dummy_input = RawInput::default();
        egui_ctx.begin_pass(dummy_input);
        let _ = egui_ctx.end_pass(); // This will initialize fonts

        let ctxt = Context::get();

        // Get the Arc<glow::Context> from the kiss3d Context
        // kiss3d Context wraps a GLContext which has an Arc<glow::Context>
        let glow_ctx = Arc::clone(&ctxt.ctxt.context);

        // Use proper shader version for both desktop and web
        #[cfg(target_arch = "wasm32")]
        let shader_version = egui_glow::ShaderVersion::Es100;
        #[cfg(not(target_arch = "wasm32"))]
        let shader_version = egui_glow::ShaderVersion::Gl140; // OpenGL 3.1+, works on macOS

        let painter = Painter::new(glow_ctx, "", Some(shader_version), false)
            .expect("Failed to create egui painter");

        // // Clear any GL errors that might have been generated during painter initialization
        // let ctxt = Context::get();
        // while ctxt.get_error() != 0 {}  // Clear all pending errors

        // Create a VAO for kiss3d to use. We'll rebind this after egui renders.
        let kiss3d_vao = ctxt.create_vertex_array();
        ctxt.bind_vertex_array(kiss3d_vao.as_ref());

        EguiRenderer {
            egui_ctx,
            painter,
            shapes: Vec::new(),
            textures_delta: Default::default(),
            kiss3d_vao,
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
    pub fn render(&mut self, width: f32, height: f32, scale_factor: f32) {
        use crate::verify;

        let ctxt = Context::get();

        // Save and setup GL state for egui rendering
        // egui_glow's painter should handle this, but we'll ensure proper state
        verify!(ctxt.disable(Context::DEPTH_TEST));
        verify!(ctxt.enable(Context::BLEND));
        verify!(ctxt.blend_func_separate(
            Context::ONE,
            Context::ONE_MINUS_SRC_ALPHA,
            Context::ONE,
            Context::ONE_MINUS_SRC_ALPHA,
        ));
        verify!(ctxt.disable(Context::CULL_FACE));
        verify!(ctxt.enable(Context::SCISSOR_TEST));

        // Update textures
        for (id, image_delta) in &self.textures_delta.set {
            self.painter.set_texture(*id, image_delta);
        }

        // Prepare clipped primitives
        let clipped_primitives = self.egui_ctx.tessellate(self.shapes.clone(), scale_factor);

        // Render
        self.painter.paint_primitives(
            [width as u32, height as u32],
            scale_factor,
            &clipped_primitives,
        );

        // Free textures
        for id in &self.textures_delta.free {
            self.painter.free_texture(*id);
        }

        self.textures_delta.clear();

        // Restore kiss3d's GL state after egui rendering
        verify!(ctxt.enable(Context::DEPTH_TEST));
        verify!(ctxt.disable(Context::BLEND));
        verify!(ctxt.enable(Context::CULL_FACE));

        // Restore kiss3d's VAO after egui rendering
        // This is crucial because egui binds its own VAO, and kiss3d expects its VAO to be bound
        ctxt.bind_vertex_array(self.kiss3d_vao.as_ref());
    }
}

impl Drop for EguiRenderer {
    fn drop(&mut self) {
        self.painter.destroy();
    }
}
