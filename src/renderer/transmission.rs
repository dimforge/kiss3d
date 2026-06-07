//! Background snapshot for screen-space refractive transmission (glass).
//!
//! After the opaque pass is resolved into the single-sample HDR scene texture,
//! this module copies it into a mip-chained "background" texture (mip 0 = the
//! resolved scene, coarser mips = progressively blurrier). Refractive (glass)
//! objects then sample that texture in screen space — offset by their refracted
//! view ray — to show the scene behind them, picking a mip by surface roughness
//! for frosted glass.
//!
//! The geometry pass that actually draws the glass lives in the window renderer
//! (it reuses the default object material); this module only owns the background.
//! It is a plain fullscreen downsample (no compute/storage), so it runs on every
//! backend, including WebGL2.

use crate::context::Context;

/// Quality of the refraction (transmission) background blur — how the blur mip
/// chain is downsampled. Higher quality uses wider, smoother kernels so frosted
/// (rough) glass stays smooth instead of blocky, at the cost of more texture taps.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TransmissionBlurQuality {
    /// 2x2 box downsample (1 bilinear tap). Cheapest; can look blocky at high roughness.
    Low,
    /// ≈4x4 tent downsample (4 bilinear taps). A balanced middle ground.
    Medium,
    /// 13-tap near-Gaussian downsample. Smoothest frosted blur.
    #[default]
    High,
}

impl TransmissionBlurQuality {
    fn pipeline_index(self) -> usize {
        match self {
            TransmissionBlurQuality::Low => 0,
            TransmissionBlurQuality::Medium => 1,
            TransmissionBlurQuality::High => 2,
        }
    }
}

/// Tunable refractive-transmission (glass) settings.
#[derive(Copy, Clone, Debug)]
pub struct TransmissionSettings {
    /// Quality of the roughness blur applied to the refracted background.
    pub blur_quality: TransmissionBlurQuality,
    /// Number of screen-space transmission passes. Each pass re-snapshots the scene
    /// (now including the glass drawn by the previous pass) and redraws the
    /// transmissive objects, so glass becomes visible *through* other glass — one
    /// extra layer of glass-behind-glass per step. `1` (the default) only refracts
    /// the opaque scene; higher values cost one more snapshot + glass draw each.
    pub steps: u32,
}

impl Default for TransmissionSettings {
    fn default() -> Self {
        TransmissionSettings {
            blur_quality: TransmissionBlurQuality::default(),
            steps: 1,
        }
    }
}

/// Owns the refraction-background mip chain for one window.
pub struct Transmission {
    settings: TransmissionSettings,
    width: u32,
    height: u32,
    _texture: wgpu::Texture,
    /// View over the whole mip chain (sampled by glass at `roughness * max_lod`).
    view: wgpu::TextureView,
    mips: u32,
    sampler: wgpu::Sampler,
    downsample_layout: wgpu::BindGroupLayout,
    /// One downsample pipeline per [`TransmissionBlurQuality`] (low / medium / high).
    downsample_pipelines: [wgpu::RenderPipeline; 3],
}

impl Transmission {
    /// Creates the refraction-background resources for the given size.
    pub fn new(width: u32, height: u32) -> Transmission {
        let ctxt = Context::get();
        let w = width.max(1);
        let h = height.max(1);

        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("transmission_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let (texture, view, mips) = Self::make_chain(w, h);

        // High-quality 13-tap downsample (near-Gaussian) so the blurred mips stay
        // smooth instead of blocky when magnified across rough glass.
        let downsample_shader = ctxt.create_shader_module(
            Some("transmission_downsample"),
            &crate::builtin::compile_shader_with_common(
                "package::transmission_downsample",
                include_str!("../builtin/transmission_downsample.wgsl"),
            ),
        );
        let downsample_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transmission_downsample_layout"),
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
            ],
        });
        let pl = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("transmission_downsample"),
            bind_group_layouts: &[Some(&downsample_layout)],
            immediate_size: 0,
        });
        // One pipeline per quality preset (same vertex shader, different filter).
        let make = |entry: &str| {
            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("transmission_downsample"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &downsample_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &downsample_shader,
                    entry_point: Some(entry),
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
        };
        let downsample_pipelines = [make("fs_low"), make("fs_medium"), make("fs_high")];

        Transmission {
            settings: TransmissionSettings::default(),
            width: w,
            height: h,
            _texture: texture,
            view,
            mips,
            sampler,
            downsample_layout,
            downsample_pipelines,
        }
    }

    /// Mutable access to the transmission settings (e.g. the blur quality).
    pub fn settings_mut(&mut self) -> &mut TransmissionSettings {
        &mut self.settings
    }

    /// Number of transmission passes to run this frame (always at least 1).
    pub fn steps(&self) -> u32 {
        self.settings.steps.max(1)
    }

    fn make_chain(w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView, u32) {
        let ctxt = Context::get();
        // Cap the chain so the coarsest mip stays a few texels wide.
        let mips = (32 - w.max(h).leading_zeros()).clamp(1, 7);
        let texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("transmission_background_chain"),
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

    /// Resizes the background targets if needed.
    pub fn resize(&mut self, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.width == w && self.height == h {
            return;
        }
        let (tex, view, mips) = Self::make_chain(w, h);
        self._texture = tex;
        self.view = view;
        self.mips = mips;
        self.width = w;
        self.height = h;
    }

    /// View over the whole background mip chain (bound to the object material's
    /// transmission-background slot).
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Maximum mip LOD (for selecting the blur level by roughness).
    pub fn max_lod(&self) -> f32 {
        (self.mips.max(1) - 1) as f32
    }

    /// Snapshots `scene_view` into the background chain: mip 0 is a copy of the
    /// resolved opaque scene, coarser mips are progressively blurrier (frosted).
    pub(crate) fn build(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_view: &wgpu::TextureView,
        gpu: &mut crate::renderer::timings::GpuTimer,
    ) {
        let ctxt = Context::get();
        for mip in 0..self.mips {
            let prev_view = if mip == 0 {
                None
            } else {
                Some(self._texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("transmission_bg_src"),
                    base_mip_level: mip - 1,
                    mip_level_count: Some(1),
                    ..Default::default()
                }))
            };
            let src_ref: &wgpu::TextureView = prev_view.as_ref().unwrap_or(scene_view);
            let dst = self._texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("transmission_bg_dst"),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let bg = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("transmission_downsample_bg"),
                layout: &self.downsample_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src_ref),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            let ts = gpu.render_scope("transmission");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("transmission_downsample_pass"),
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
                timestamp_writes: ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.downsample_pipelines[self.settings.blur_quality.pipeline_index()]);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
