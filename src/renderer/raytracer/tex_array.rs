//! Packs the scene's per-object PBR textures into one `texture_2d_array` for the
//! path tracer.
//!
//! The path tracer samples object albedo / normal / metallic-roughness / emissive
//! maps through a single array texture indexed by a per-material layer (see the
//! `*_tex` fields of [`RtMaterial`](super::scene_data::RtMaterial)). Source
//! textures may have any resolution, so each is resampled into a fixed-size layer
//! with a small fullscreen blit. A 1×1 white fallback layer is always present so
//! the binding is valid even when the scene has no maps. A 2D-array texture is
//! used (rather than bindless arrays) because it is broadly supported, including
//! on the compute / Metal backend.

use std::sync::Arc;

use crate::context::Context;
use crate::resource::Texture;

/// Fixed resolution every layer is resampled to. A power of two keeps sampling
/// cheap and is well within array-layer limits on every backend.
const LAYER_SIZE: u32 = 1024;

/// The packed texture array plus its sampler.
pub struct TexArray {
    /// The array texture view bound at group 1, binding 6.
    pub view: wgpu::TextureView,
    /// The sampler bound at group 1, binding 7.
    pub sampler: wgpu::Sampler,
}

impl TexArray {
    /// Builds the array from the scene's source textures (one layer each), plus a
    /// trailing fallback layer. An empty `sources` list yields a single white
    /// layer so the binding stays valid.
    pub fn build(sources: &[Arc<Texture>]) -> TexArray {
        let ctxt = Context::get();
        let layers = (sources.len() as u32) + 1; // +1 fallback white layer

        let texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("rt_tex_array"),
            size: wgpu::Extent3d {
                width: LAYER_SIZE,
                height: LAYER_SIZE,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Linear (non-sRGB) so the path tracer controls color conversion; the
            // blit just resamples the source texels.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        // Blit pipeline (created on demand; cheap and only runs on scene rebuild).
        let bgl = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt_blit_bgl"),
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
        let layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rt_blit_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let shader = ctxt.create_shader_module(
            Some("rt_blit_shader"),
            include_str!("../../builtin/raytrace/blit.wgsl"),
        );
        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rt_blit_pipeline"),
            layout: Some(&layout),
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
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
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

        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rt_blit_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Render each source into its layer. The fallback white layer is left as
        // a cleared white attachment.
        let mut encoder = ctxt.create_command_encoder(Some("rt_tex_array_blit"));
        for (i, src) in sources.iter().enumerate() {
            let layer_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("rt_tex_array_layer"),
                base_array_layer: i as u32,
                array_layer_count: Some(1),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });
            let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rt_blit_bind_group"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rt_blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &layer_view,
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
            pass.draw(0..4, 0..1);
        }
        // Clear the fallback layer to white.
        {
            let layer_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("rt_tex_array_fallback"),
                base_array_layer: sources.len() as u32,
                array_layer_count: Some(1),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rt_blit_fallback_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &layer_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        ctxt.submit(std::iter::once(encoder.finish()));

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("rt_tex_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        TexArray { view, sampler }
    }
}
