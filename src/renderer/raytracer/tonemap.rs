//! Fullscreen tonemap pass that turns the HDR accumulation buffer into the
//! final LDR image, written to the frame's output view.

use bytemuck::{Pod, Zeroable};

use crate::context::Context;

use super::accumulation::Accumulation;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TonemapUniforms {
    /// Resolution of the accumulation buffer (the traced resolution).
    src_width: u32,
    src_height: u32,
    /// Resolution of the output framebuffer (may be larger; image is upscaled).
    dst_width: u32,
    dst_height: u32,
    exposure: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QuadVertex {
    position: [f32; 2],
}

/// Owns the tonemap render pipeline and its fullscreen quad.
pub struct Tonemap {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: wgpu::Buffer,
    uniform: wgpu::Buffer,
}

impl Tonemap {
    /// Creates the tonemap pipeline targeting the surface format.
    pub fn new() -> Tonemap {
        let ctxt = Context::get();

        let bind_group_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt_tonemap_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rt_tonemap_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let shader = ctxt.create_shader_module(
            Some("rt_tonemap_shader"),
            include_str!("../../builtin/raytrace/tonemap.wgsl"),
        );

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };

        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rt_tonemap_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ctxt.surface_format,
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

        let vertices = [
            QuadVertex { position: [-1.0, -1.0] },
            QuadVertex { position: [1.0, -1.0] },
            QuadVertex { position: [-1.0, 1.0] },
            QuadVertex { position: [1.0, 1.0] },
        ];
        let vertex_buffer = ctxt.create_buffer_init(
            Some("rt_tonemap_vertex_buffer"),
            bytemuck::cast_slice(&vertices),
            wgpu::BufferUsages::VERTEX,
        );

        let uniform = ctxt.create_buffer_simple(
            Some("rt_tonemap_uniform"),
            std::mem::size_of::<TonemapUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        Tonemap {
            pipeline,
            bind_group_layout,
            vertex_buffer,
            uniform,
        }
    }

    /// Tonemaps the accumulation buffer into `output_view`, upscaling from the
    /// traced resolution (`accum`) to `dst_width` x `dst_height` if they differ.
    pub fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        accum: &Accumulation,
        exposure: f32,
        output_view: &wgpu::TextureView,
        dst_width: u32,
        dst_height: u32,
    ) {
        let ctxt = Context::get();

        ctxt.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&TonemapUniforms {
                src_width: accum.width,
                src_height: accum.height,
                dst_width,
                dst_height,
                exposure,
                _pad: [0.0; 3],
            }),
        );

        let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_tonemap_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: accum.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.uniform.as_entire_binding(),
                },
            ],
        });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("rt_tonemap_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
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

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..4, 0..1);
    }
}
