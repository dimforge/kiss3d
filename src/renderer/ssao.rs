//! Screen-space ambient occlusion for the rasterizer.
//!
//! Runs after a depth + view-position prepass (rendered by the window using the
//! material's prepass pipeline): a hemisphere-sampling pass produces a raw AO
//! buffer, a box blur smooths it, and the result is handed to the material to
//! darken ambient lighting. Single-sampled regardless of the scene's MSAA.

use crate::context::Context;
use bytemuck::{Pod, Zeroable};
use glamx::Mat4;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SsaoUniforms {
    proj: [[f32; 4]; 4],
    inv_resolution: [f32; 2],
    radius: f32,
    bias: f32,
    intensity: f32,
    power: f32,
    _pad: [f32; 2],
}

/// Tunable SSAO parameters.
#[derive(Copy, Clone, Debug)]
pub struct SsaoSettings {
    /// World-space sampling radius.
    pub radius: f32,
    /// Depth bias to avoid self-occlusion.
    pub bias: f32,
    /// Occlusion strength multiplier.
    pub intensity: f32,
    /// Contrast exponent applied to the AO.
    pub power: f32,
}

impl Default for SsaoSettings {
    fn default() -> Self {
        SsaoSettings {
            radius: 0.5,
            bias: 0.025,
            intensity: 1.2,
            power: 1.5,
        }
    }
}

struct Target {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// Owns the SSAO prepass targets, AO buffers and pipelines for one window.
pub struct Ssao {
    settings: SsaoSettings,
    width: u32,
    height: u32,

    viewpos: Target,
    depth: Target,
    // G-buffer targets shared with SSR: world normal + roughness, F0 + metallic, and
    // per-object SSR params. Written by the same prepass; unused by SSAO itself.
    normal: Target,
    material: Target,
    ssr_params: Target,
    ao: Target,
    ao_blur: Target,

    sampler: wgpu::Sampler,
    ssao_layout: wgpu::BindGroupLayout,
    ssao_pipeline: wgpu::RenderPipeline,
    ssao_uniform: wgpu::Buffer,
    blur_layout: wgpu::BindGroupLayout,
    blur_pipeline: wgpu::RenderPipeline,
    blur_uniform: wgpu::Buffer,
}

impl Ssao {
    /// Creates the SSAO resources for the given size.
    pub fn new(width: u32, height: u32) -> Ssao {
        let ctxt = Context::get();
        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssao_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let make_layout = |label: &str| {
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
        };
        let ssao_layout = make_layout("ssao_layout");
        let blur_layout = make_layout("ssao_blur_layout");

        let make_pipeline = |label: &str, src: &str, layout: &wgpu::BindGroupLayout| {
            let shader = ctxt.create_shader_module(Some(label), src);
            let pl = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
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
                        format: wgpu::TextureFormat::R16Float,
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
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let ssao_pipeline = make_pipeline(
            "ssao",
            &crate::builtin::compile_shader_with_common(
                "package::ssao",
                include_str!("../builtin/ssao.wgsl"),
            ),
            &ssao_layout,
        );
        let blur_pipeline = make_pipeline(
            "ssao_blur",
            include_str!("../builtin/ssao_blur.wgsl"),
            &blur_layout,
        );

        let ssao_uniform = ctxt.create_buffer_simple(
            Some("ssao_uniform"),
            std::mem::size_of::<SsaoUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let blur_uniform = ctxt.create_buffer_simple(
            Some("ssao_blur_uniform"),
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let (viewpos, depth, normal, material, ssr_params, ao, ao_blur) =
            Self::make_targets(width, height);
        Ssao {
            settings: SsaoSettings::default(),
            width,
            height,
            viewpos,
            depth,
            normal,
            material,
            ssr_params,
            ao,
            ao_blur,
            sampler,
            ssao_layout,
            ssao_pipeline,
            ssao_uniform,
            blur_layout,
            blur_pipeline,
            blur_uniform,
        }
    }

    #[allow(clippy::type_complexity)]
    #[allow(clippy::type_complexity)]
    fn make_targets(
        width: u32,
        height: u32,
    ) -> (Target, Target, Target, Target, Target, Target, Target) {
        let ctxt = Context::get();
        let w = width.max(1);
        let h = height.max(1);
        let color = |label: &str, format: wgpu::TextureFormat| {
            let tex = ctxt.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            Target {
                _texture: tex,
                view,
            }
        };
        let viewpos = color("ssao_viewpos", wgpu::TextureFormat::Rgba16Float);
        let depth = {
            let tex = ctxt.create_texture(&wgpu::TextureDescriptor {
                label: Some("ssao_prepass_depth"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Context::depth_format(),
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            Target {
                _texture: tex,
                view,
            }
        };
        let normal = color("gbuffer_normal_roughness", wgpu::TextureFormat::Rgba16Float);
        let material = color("gbuffer_f0_metallic", wgpu::TextureFormat::Rgba16Float);
        let ssr_params = color("gbuffer_ssr_params", wgpu::TextureFormat::Rgba16Float);
        let ao = color("ssao_ao", wgpu::TextureFormat::R16Float);
        let ao_blur = color("ssao_ao_blur", wgpu::TextureFormat::R16Float);
        (viewpos, depth, normal, material, ssr_params, ao, ao_blur)
    }

    /// Resizes the SSAO targets if needed.
    pub fn resize(&mut self, width: u32, height: u32) {
        if self.width == width.max(1) && self.height == height.max(1) {
            return;
        }
        let (viewpos, depth, normal, material, ssr_params, ao, ao_blur) =
            Self::make_targets(width, height);
        self.viewpos = viewpos;
        self.depth = depth;
        self.normal = normal;
        self.material = material;
        self.ssr_params = ssr_params;
        self.ao = ao;
        self.ao_blur = ao_blur;
        self.width = width.max(1);
        self.height = height.max(1);
    }

    /// Mutable access to the SSAO settings.
    pub fn settings_mut(&mut self) -> &mut SsaoSettings {
        &mut self.settings
    }

    /// The view-position prepass color attachment (window renders the scene here).
    pub fn viewpos_view(&self) -> &wgpu::TextureView {
        &self.viewpos.view
    }

    /// The prepass depth attachment.
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth.view
    }

    /// The G-buffer world-normal (xyz) + linear-roughness (a) attachment.
    pub fn normal_view(&self) -> &wgpu::TextureView {
        &self.normal.view
    }

    /// The G-buffer F0 (rgb) + metallic (a) attachment.
    pub fn material_view(&self) -> &wgpu::TextureView {
        &self.material.view
    }

    /// The G-buffer per-object SSR-params attachment (intensity, infinite_thick,
    /// distance_attenuation, fresnel).
    pub fn ssr_params_view(&self) -> &wgpu::TextureView {
        &self.ssr_params.view
    }

    /// The final (blurred) AO texture, sampled by the material.
    pub fn ao_view(&self) -> &wgpu::TextureView {
        &self.ao_blur.view
    }

    /// Runs the SSAO + blur passes. The view-position prepass must already have
    /// been rendered into [`viewpos_view`](Self::viewpos_view). `proj` is the
    /// camera projection used to project samples back to screen space.
    pub(crate) fn compute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        proj: Mat4,
        gpu: &mut crate::renderer::timings::GpuTimer,
    ) {
        let ctxt = Context::get();
        let inv_res = [1.0 / self.width as f32, 1.0 / self.height as f32];

        ctxt.write_buffer(
            &self.ssao_uniform,
            0,
            bytemuck::bytes_of(&SsaoUniforms {
                proj: proj.to_cols_array_2d(),
                inv_resolution: inv_res,
                radius: self.settings.radius,
                bias: self.settings.bias,
                intensity: self.settings.intensity,
                power: self.settings.power,
                _pad: [0.0; 2],
            }),
        );
        ctxt.write_buffer(
            &self.blur_uniform,
            0,
            bytemuck::bytes_of(&[inv_res[0], inv_res[1], 0.0, 0.0]),
        );

        // SSAO pass: view-position -> raw AO.
        let ssao_bg = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_bg"),
            layout: &self.ssao_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.viewpos.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.ssao_uniform.as_entire_binding(),
                },
            ],
        });
        Self::fullscreen(
            encoder,
            &self.ssao_pipeline,
            &ssao_bg,
            &self.ao.view,
            "ssao_pass",
            gpu,
        );

        // Blur pass: raw AO -> blurred AO.
        let blur_bg = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_blur_bg"),
            layout: &self.blur_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.ao.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.blur_uniform.as_entire_binding(),
                },
            ],
        });
        Self::fullscreen(
            encoder,
            &self.blur_pipeline,
            &blur_bg,
            &self.ao_blur.view,
            "ssao_blur_pass",
            gpu,
        );
    }

    fn fullscreen(
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        bind_group: &wgpu::BindGroup,
        target: &wgpu::TextureView,
        label: &str,
        gpu: &mut crate::renderer::timings::GpuTimer,
    ) {
        let ssao_ts = gpu.render_scope("ssao");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: ssao_ts,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
