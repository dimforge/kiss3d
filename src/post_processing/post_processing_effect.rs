//! Trait implemented by every post-processing effect.

use crate::resource::RenderTarget;
use std::collections::HashMap;

/// Context passed to post-processing effects during draw.
pub struct PostProcessingContext<'a> {
    /// The command encoder for this frame.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The output color view to render to.
    pub output_view: &'a wgpu::TextureView,
    /// The colour format of `output_view`. A pipeline is tied to the format of
    /// the attachment it writes, so an effect that runs on both the LDR chain
    /// (the surface format) and the film chain
    /// ([`HDR_FORMAT`](crate::post_processing::HDR_FORMAT)) needs one per
    /// format; [`FormatPipelines`] keeps them.
    pub output_format: wgpu::TextureFormat,
}

/// An effect's render pipeline, built once per colour format it is asked to
/// write into.
///
/// The built-in effects all keep theirs here rather than building one pipeline
/// up front, so the same effect can be handed to either post-processing chain.
pub struct FormatPipelines {
    build: Box<dyn Fn(wgpu::TextureFormat) -> wgpu::RenderPipeline>,
    built: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

impl FormatPipelines {
    /// Defers pipeline creation to `build`, which is called once per format.
    pub fn new(build: impl Fn(wgpu::TextureFormat) -> wgpu::RenderPipeline + 'static) -> Self {
        FormatPipelines {
            build: Box::new(build),
            built: HashMap::new(),
        }
    }

    /// The pipeline writing `format`, built the first time it is asked for.
    /// The returned pipeline is a handle, so cloning it out costs nothing and
    /// leaves the caller free to borrow the rest of its effect.
    pub fn get(&mut self, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        if let Some(pipeline) = self.built.get(&format) {
            return pipeline.clone();
        }
        let pipeline = (self.build)(format);
        self.built.insert(format, pipeline.clone());
        pipeline
    }
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
