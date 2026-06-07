//! Image-based lighting support: a mip-chained equirectangular environment map.
//!
//! The rasterizer approximates IBL with the "mip-as-prefilter" technique: a
//! single equirectangular HDR texture is given a full mip chain (each level a
//! box-downsample of the previous), so the coarse mips stand in for
//! roughness-prefiltered reflections, and the coarsest mip stands in for the
//! diffuse irradiance. Combined with Karis' analytic environment BRDF (in the
//! shader) this needs no separate irradiance / prefilter / BRDF-LUT passes while
//! still giving plausible ambient reflections and fill light.

use crate::context::Context;

/// A GPU-resident equirectangular environment map with a full mip chain, used as
/// the rasterizer's image-based-lighting source.
pub struct EnvironmentMap {
    // Kept alive alongside the view.
    _texture: wgpu::Texture,
    /// View over all mip levels (sampled by the lighting shader with an explicit LOD).
    pub view: wgpu::TextureView,
    /// Linear/trilinear sampler (repeat in U, clamp in V).
    pub sampler: wgpu::Sampler,
    /// Number of mip levels (the max sampleable LOD is `mip_count - 1`).
    pub mip_count: u32,
}

impl EnvironmentMap {
    /// Builds a mip-chained environment map from an equirectangular image.
    pub fn from_image(img: &image::DynamicImage) -> EnvironmentMap {
        use image::GenericImageView;
        let (w, h) = img.dimensions();
        let rgba = img.to_rgba32f();
        Self::from_rgba_f32(w, h, rgba.as_raw())
    }

    /// Builds a mip-chained environment map from RGBA-f32 pixels.
    pub fn from_rgba_f32(width: u32, height: u32, rgba: &[f32]) -> EnvironmentMap {
        let ctxt = Context::get();
        let width = width.max(1);
        let height = height.max(1);
        let mip_count = (32 - width.max(height).leading_zeros()).max(1);

        let texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl_environment"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        // Upload mip 0 (f32 -> f16).
        let halves: Vec<u16> = rgba.iter().map(|&v| f32_to_f16(v)).collect();
        ctxt.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&halves),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 8),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ibl_environment_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        if mip_count > 1 {
            Self::generate_mips(&texture, &sampler, mip_count);
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        EnvironmentMap {
            _texture: texture,
            view,
            sampler,
            mip_count,
        }
    }

    /// Renders each mip from the previous one with a box (bilinear) downsample.
    fn generate_mips(texture: &wgpu::Texture, sampler: &wgpu::Sampler, mip_count: u32) {
        let ctxt = Context::get();

        let layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ibl_downsample_layout"),
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
        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ibl_downsample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = ctxt.create_shader_module(
            Some("ibl_downsample"),
            include_str!("../builtin/env_downsample.wgsl"),
        );
        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ibl_downsample_pipeline"),
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
                    format: wgpu::TextureFormat::Rgba16Float,
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
        });

        let mut encoder = ctxt.create_command_encoder(Some("ibl_mipgen_encoder"));
        for mip in 1..mip_count {
            let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("ibl_mip_src"),
                base_mip_level: mip - 1,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("ibl_mip_dst"),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ibl_downsample_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ibl_downsample_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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
        ctxt.submit(std::iter::once(encoder.finish()));
    }
}

/// Converts an `f32` to IEEE-754 half-precision bits (truncating mantissa).
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    if exp <= 0 {
        sign
    } else if exp >= 0x1f {
        sign | 0x7c00
    } else {
        sign | ((exp as u16) << 10) | (mant as u16)
    }
}
