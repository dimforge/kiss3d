//! Trait implemented by every post-processing effect.

use crate::resource::RenderTarget;

/// Context passed to post-processing effects during draw.
pub struct PostProcessingContext<'a> {
    /// The command encoder for this frame.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The output color view to render to.
    pub output_view: &'a wgpu::TextureView,
}

/// Trait for implementing custom post-processing effects.
///
/// Post-processing effects are applied after the 3D scene has been rendered to a texture.
/// Only one post-processing effect can be active at a time. Implement this trait to create
/// custom effects like bloom, blur, edge detection, etc.
pub trait PostProcessingEffect {
    /// Updates the post-processing effect state.
    ///
    /// Called once per frame to update effect parameters based on time and viewport settings.
    ///
    /// # Arguments
    /// * `dt` - Delta time since last frame in seconds
    /// * `w` - Screen width in pixels
    /// * `h` - Screen height in pixels
    /// * `znear` - Near clipping plane distance
    /// * `zfar` - Far clipping plane distance
    fn update(&mut self, dt: f32, w: f32, h: f32, znear: f32, zfar: f32);

    /// Renders the post-processing effect.
    ///
    /// This method is called after the scene has been rendered to a texture.
    /// The effect should read from the render target and apply its processing.
    ///
    /// # Arguments
    /// * `target` - The render target containing the rendered scene (color and depth textures)
    /// * `context` - The post-processing context with encoder and output view
    fn draw(&mut self, target: &RenderTarget, context: &mut PostProcessingContext);
}
