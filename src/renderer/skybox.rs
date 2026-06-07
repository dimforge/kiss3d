//! Equirectangular skybox for the rasterizer.
//!
//! Renders an HDR environment map as the scene background, drawn as a full-screen
//! pass into the HDR film before the opaque geometry (which then overwrites it
//! wherever something is visible). The direction→UV mapping matches the path
//! tracer's HDRI lookup, so the same image produces the same sky in both
//! backends.

use crate::context::Context;
use crate::renderer::ibl::EnvironmentMap;
use crate::renderer::raytracer::environment::Environment;
use crate::resource::{multisample_state, PipelineCache};
use bytemuck::{Pod, Zeroable};
use std::path::Path;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SkyUniforms {
    inv_view_proj: [[f32; 4]; 4],
    // (cos(rotation), sin(rotation), intensity, unused)
    params: [f32; 4],
}

/// Owns the equirectangular environment map and the full-screen pipeline used to
/// draw it as the scene background. One instance lives on each
/// [`Window`](crate::window::Window).
pub struct Skybox {
    environment: Environment,
    /// Mip-chained copy of the environment used as the image-based-lighting
    /// source (built alongside the background environment).
    ibl_env: Option<EnvironmentMap>,
    rotation: f32,
    intensity: f32,
    layout: wgpu::BindGroupLayout,
    pipeline: PipelineCache,
    uniform: wgpu::Buffer,
}

impl Default for Skybox {
    fn default() -> Self {
        Self::new()
    }
}

impl Skybox {
    /// Creates a skybox with no environment set (renders nothing until one is).
    pub fn new() -> Skybox {
        let ctxt = Context::get();

        let layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skybox_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skybox_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let shader = ctxt
            .create_shader_module(Some("skybox_shader"), include_str!("../builtin/skybox.wgsl"));

        // Built lazily per MSAA sample count to match the HDR scene attachment.
        let pipeline = PipelineCache::new(move |sample_count| {
            let ctxt = Context::get();
            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("skybox_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: Context::render_format(),
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                // No depth attachment: the sky always fills the background and is
                // overwritten by the opaque pass wherever geometry is visible.
                depth_stencil: None,
                multisample: multisample_state(sample_count),
                multiview_mask: None,
                cache: None,
            })
        });

        let uniform = ctxt.create_buffer_simple(
            Some("skybox_uniform"),
            std::mem::size_of::<SkyUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        Skybox {
            environment: Environment::fallback(),
            ibl_env: None,
            rotation: 0.0,
            intensity: 1.0,
            layout,
            pipeline,
            uniform,
        }
    }

    /// Whether an environment map is currently set (otherwise rendering is a no-op).
    pub fn is_set(&self) -> bool {
        self.environment.present
    }

    /// Sets the skybox from an equirectangular image file (HDR `.hdr`, EXR, or any
    /// format the `image` crate decodes). Returns `false` if it can't be decoded.
    pub fn set_from_file(&mut self, path: &Path) -> bool {
        match image::open(path) {
            Ok(img) => {
                self.set_image(&img);
                true
            }
            Err(_) => false,
        }
    }

    /// Sets the skybox from an already-decoded equirectangular image.
    pub fn set_image(&mut self, image: &image::DynamicImage) {
        self.environment = Environment::from_image(image);
        self.ibl_env = Some(EnvironmentMap::from_image(image));
    }

    /// Clears the skybox (subsequent frames render no background or IBL).
    pub fn clear(&mut self) {
        self.environment = Environment::fallback();
        self.ibl_env = None;
    }

    /// The mip-chained environment map used for image-based lighting, if set.
    pub fn ibl_env(&self) -> Option<&EnvironmentMap> {
        self.ibl_env.as_ref()
    }

    /// The environment Y-rotation in radians.
    pub fn rotation(&self) -> f32 {
        self.rotation
    }

    /// The environment luminance multiplier.
    pub fn intensity(&self) -> f32 {
        self.intensity
    }

    /// Sets the skybox Y-axis rotation (radians) and a luminance multiplier.
    pub fn set_orientation(&mut self, rotation_radians: f32, intensity: f32) {
        self.rotation = rotation_radians;
        self.intensity = intensity.max(0.0);
    }

    /// Draws the skybox into `color_view` (the HDR scene attachment) using the
    /// camera's inverse view-projection. A no-op when no environment is set.
    pub(crate) fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        sample_count: u32,
        inverse_view_proj: glamx::Mat4,
    ) {
        if !self.environment.present {
            return;
        }
        let ctxt = Context::get();

        ctxt.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&SkyUniforms {
                inv_view_proj: inverse_view_proj.to_cols_array_2d(),
                params: [self.rotation.cos(), self.rotation.sin(), self.intensity, 0.0],
            }),
        );

        let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skybox_bind_group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.environment.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.environment.sampler),
                },
            ],
        });

        let pipeline = self.pipeline.get(sample_count);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("skybox_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Load: the clear pass already ran; the sky simply overwrites
                    // the cleared background everywhere it draws.
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
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
