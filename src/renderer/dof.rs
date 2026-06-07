//! Depth of field (DoF) for the rasterizer.
//!
//! Runs after the opaque pass + MSAA resolve (and after SSR), reading the resolved
//! HDR scene color and the view-position G-buffer produced by the shared geometry
//! prepass. A per-pixel signed circle-of-confusion (CoC) is computed from a thin-
//! lens model (focal distance + aperture), packed alongside the scene color into a
//! mip chain, and a single spiral-gather pass reconstructs the blurred image and
//! writes it back over the scene before tonemapping. Single-frame.

use crate::context::Context;
use bytemuck::{Pod, Zeroable};
use glamx::Mat4;

/// How the out-of-focus regions are blurred.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DepthOfFieldMode {
    /// Circular (uniform-disk) bokeh — sharper highlight discs, higher contrast.
    Bokeh,
    /// Smooth gaussian falloff — softer, cheaper-looking blur.
    Gaussian,
}

/// Tunable depth-of-field parameters. The defaults model a 50mm-ish lens on a
/// full-frame sensor focused 10 units away.
#[derive(Copy, Clone, Debug)]
pub struct DofSettings {
    /// Blur kernel shape.
    pub mode: DepthOfFieldMode,
    /// Distance (world units) to the plane that stays perfectly in focus.
    pub focal_distance: f32,
    /// Aperture in f-stops: smaller values open the aperture and blur more.
    pub aperture_f_stops: f32,
    /// Sensor height in world units (default 18.66mm ≈ a full-frame 35mm sensor).
    /// Together with the camera's vertical FOV this fixes the lens focal length.
    pub sensor_height: f32,
    /// Maximum circle-of-confusion *diameter*, in pixels. Caps the blur radius
    /// (and hence the cost/quality) of strongly out-of-focus regions.
    pub max_coc_diameter: f32,
    /// Surfaces farther than this (world units) are clamped to this depth, so the
    /// far background stops getting blurrier. Default is effectively infinite.
    pub max_depth: f32,
    /// Number of taps in the gather spiral. More taps = smoother bokeh, higher cost.
    pub num_taps: u32,
}

impl Default for DofSettings {
    fn default() -> Self {
        // Defaults: a full-frame sensor with a deliberately wide
        // `f/0.125` aperture so simply enabling DoF produces a visible blur (a
        // physically realistic aperture has a very deep depth of field at typical
        // scene scales). Raise `aperture_f_stops` for a deeper, subtler effect.
        DofSettings {
            mode: DepthOfFieldMode::Bokeh,
            focal_distance: 10.0,
            aperture_f_stops: 1.0 / 8.0,
            sensor_height: 0.018_66,
            max_coc_diameter: 64.0,
            max_depth: 1.0e6,
            num_taps: 48,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DofUniforms {
    proj: [[f32; 4]; 4],
    params0: [f32; 4],
    params1: [f32; 4],
    params2: [f32; 4],
}

/// Owns the DoF color+CoC mip chain, pipelines and uniform for one window.
pub struct Dof {
    settings: DofSettings,
    width: u32,
    height: u32,

    // Scene color (rgb) + signed CoC (a) mip chain, rebuilt each frame.
    _chain_texture: wgpu::Texture,
    chain_view: wgpu::TextureView,
    chain_mips: u32,

    sampler: wgpu::Sampler,

    // CoC pass: scene + view-position -> color+CoC (mip 0 of the chain).
    coc_layout: wgpu::BindGroupLayout,
    coc_pipeline: wgpu::RenderPipeline,
    // Mip-chain downsample (reuses env_downsample.wgsl, averages color + CoC).
    downsample_layout: wgpu::BindGroupLayout,
    downsample_pipeline: wgpu::RenderPipeline,
    // Gather pass: chain -> composited DoF, written back over the scene.
    gather_layout: wgpu::BindGroupLayout,
    gather_pipeline: wgpu::RenderPipeline,

    uniform: wgpu::Buffer,
}

impl Dof {
    /// Creates the DoF resources for the given size.
    pub fn new(width: u32, height: u32) -> Dof {
        let ctxt = Context::get();
        let w = width.max(1);
        let h = height.max(1);

        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("dof_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let (chain_texture, chain_view, chain_mips) = Self::make_chain(w, h);

        let shader = ctxt.create_shader_module(
            Some("dof"),
            &crate::builtin::compile_shader_with_common(
                "package::dof",
                include_str!("../builtin/dof.wgsl"),
            ),
        );

        // CoC + gather share the same two-texture + sampler + uniform layout.
        let coc_layout = make_layout("dof_coc_layout");
        let gather_layout = make_layout("dof_gather_layout");
        let coc_pipeline =
            make_fullscreen_pipeline("dof_coc", &shader, "fs_coc", &coc_layout, None);
        let gather_pipeline =
            make_fullscreen_pipeline("dof_gather", &shader, "fs_gather", &gather_layout, None);

        // Mip-chain downsample reuses the env box-downsample shader (texture+sampler).
        let downsample_shader = ctxt.create_shader_module(
            Some("dof_downsample"),
            &crate::builtin::compile_shader_with_common(
                "package::env_downsample",
                crate::builtin::ENV_DOWNSAMPLE_WESL,
            ),
        );
        let downsample_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof_downsample_layout"),
            entries: &[
                tex_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let downsample_pipeline = make_downsample_pipeline(&downsample_shader, &downsample_layout);

        let uniform = ctxt.create_buffer_simple(
            Some("dof_uniform"),
            std::mem::size_of::<DofUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        Dof {
            settings: DofSettings::default(),
            width: w,
            height: h,
            _chain_texture: chain_texture,
            chain_view,
            chain_mips,
            sampler,
            coc_layout,
            coc_pipeline,
            downsample_layout,
            downsample_pipeline,
            gather_layout,
            gather_pipeline,
            uniform,
        }
    }

    fn make_chain(w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView, u32) {
        let ctxt = Context::get();
        // Cap the chain so the coarsest mip stays a few texels wide.
        let mips = (32 - w.max(h).leading_zeros()).clamp(1, 7);
        let texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("dof_chain"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::post_processing::HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view, mips)
    }

    /// Resizes the DoF targets if needed.
    pub fn resize(&mut self, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.width == w && self.height == h {
            return;
        }
        let (tex, view, mips) = Self::make_chain(w, h);
        self._chain_texture = tex;
        self.chain_view = view;
        self.chain_mips = mips;
        self.width = w;
        self.height = h;
    }

    /// Mutable access to the DoF settings.
    pub fn settings_mut(&mut self) -> &mut DofSettings {
        &mut self.settings
    }

    /// The current DoF settings.
    pub fn settings(&self) -> &DofSettings {
        &self.settings
    }

    /// Runs DoF: computes per-pixel color+CoC from `scene_view` and `viewpos`,
    /// builds a mip chain, then gathers the blurred result back into `scene_view`.
    /// `proj` is the (pass-0) projection matrix; `background_depth` is the depth
    /// used for sky / background pixels (typically the far clip plane).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_view: &wgpu::TextureView,
        viewpos: &wgpu::TextureView,
        proj: Mat4,
        background_depth: f32,
        gpu: &mut crate::renderer::timings::GpuTimer,
    ) {
        let ctxt = Context::get();
        let s = &self.settings;

        ctxt.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&DofUniforms {
                proj: proj.to_cols_array_2d(),
                params0: [
                    1.0 / self.width as f32,
                    1.0 / self.height as f32,
                    self.height as f32,
                    (self.chain_mips.max(1) - 1) as f32,
                ],
                params1: [
                    s.focal_distance,
                    s.aperture_f_stops,
                    s.sensor_height,
                    s.max_coc_diameter,
                ],
                params2: [
                    s.max_depth,
                    background_depth,
                    if s.mode == DepthOfFieldMode::Gaussian {
                        1.0
                    } else {
                        0.0
                    },
                    s.num_taps.max(1) as f32,
                ],
            }),
        );

        // 1. CoC pass -> chain mip 0 (color + signed CoC).
        let mip0 = self
            ._chain_texture
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("dof_chain_mip0"),
                base_mip_level: 0,
                mip_level_count: Some(1),
                ..Default::default()
            });
        {
            let bg = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dof_coc_bg"),
                layout: &self.coc_layout,
                entries: &[
                    tex_bind(0, scene_view),
                    tex_bind(1, viewpos),
                    samp_bind(2, &self.sampler),
                    uniform_bind(3, &self.uniform),
                ],
            });
            let dof_ts = gpu.render_scope("dof");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dof_coc_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &mip0,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: dof_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.coc_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 2. Box-downsample the chain so coarser mips pre-blur color and CoC.
        for mip in 1..self.chain_mips {
            let src = self
                ._chain_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("dof_chain_src"),
                    base_mip_level: mip - 1,
                    mip_level_count: Some(1),
                    ..Default::default()
                });
            let dst = self
                ._chain_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("dof_chain_dst"),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                });
            let bg = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dof_downsample_bg"),
                layout: &self.downsample_layout,
                entries: &[tex_bind(0, &src), samp_bind(1, &self.sampler)],
            });
            let dof_ts = gpu.render_scope("dof");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dof_downsample_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: dof_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.downsample_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 3. Gather pass: composite the blurred result back over the scene. The
        //    gather samples the chain (not `scene_view`) so reading and writing the
        //    same texture never overlaps.
        {
            let bg = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dof_gather_bg"),
                layout: &self.gather_layout,
                entries: &[
                    tex_bind(0, &self.chain_view),
                    tex_bind(1, &self.chain_view),
                    samp_bind(2, &self.sampler),
                    uniform_bind(3, &self.uniform),
                ],
            });
            let dof_ts = gpu.render_scope("dof");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dof_gather_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: dof_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.gather_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// Two filtered textures + a sampler + a uniform buffer (all fragment-stage).
fn make_layout(label: &str) -> wgpu::BindGroupLayout {
    let ctxt = Context::get();
    ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            tex_entry(0),
            tex_entry(1),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
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
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn tex_bind(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn samp_bind(binding: u32, sampler: &wgpu::Sampler) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Sampler(sampler),
    }
}

fn uniform_bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn make_fullscreen_pipeline(
    label: &str,
    shader: &wgpu::ShaderModule,
    fs_entry: &str,
    layout: &wgpu::BindGroupLayout,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    let ctxt = Context::get();
    let pl = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: crate::post_processing::HDR_FORMAT,
                blend,
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
}

fn make_downsample_pipeline(
    shader: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let ctxt = Context::get();
    let pl = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("dof_downsample"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("dof_downsample"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: crate::post_processing::HDR_FORMAT,
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
}
