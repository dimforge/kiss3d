use crate::camera::Camera3d;
use crate::resource::RenderContext;

/// Trait for implementing custom rendering logic.
///
/// Implement this trait to create custom renderers that can draw additional
/// geometry or effects during the render pipeline. Custom renderers are invoked
/// during each rendering pass.
pub trait Renderer3d {
    /// Performs a custom rendering pass.
    ///
    /// This method is called during each rendering pass, after the main scene
    /// has been rendered but before post-processing effects are applied.
    ///
    /// # Arguments
    /// * `pass` - The current rendering pass index (0 for single-pass rendering)
    /// * `camera` - The camera being used for rendering
    /// * `render_pass` - The active wgpu render pass to draw into
    /// * `context` - The render context with viewport info
    fn render(
        &mut self,
        pass: usize,
        camera: &mut dyn Camera3d,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext,
    );
}
