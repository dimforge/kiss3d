// This whole file is inspired by: https://github.com/jeaye/q3/blob/master/src/client/ui/ttf/renderer.rs
// available under the BSD-3 licence.
// It has been modified to work with wgpu, nalgebra, and rusttype

use crate::context::Context;
use crate::resource::RenderContext2dEncoder;
use crate::text::Font;
use bytemuck::{Pod, Zeroable};
use glamx::Vec2;
use rusttype;
use rusttype::gpu_cache::Cache;
use std::sync::Arc;
use crate::color::Color;

/// Vertex data for a text quad.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TextVertex {
    position: [f32; 2],
    tex_coord: [f32; 2],
    color: [f32; 4],
}

/// Uniforms for text rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TextUniforms {
    inv_size: [f32; 2],
    _padding: [f32; 2],
}

struct TextRenderContext {
    len: usize,
    scale: f32,
    color: [f32; 4],
    pos: Vec2,
    font: Arc<Font>,
}

/// A ttf text renderer.
pub struct TextRenderer {
    text: String,
    cache: Cache<'static>,
    glyph_texture: wgpu::Texture,
    glyph_texture_view: wgpu::TextureView,
    glyph_sampler: wgpu::Sampler,
    pipeline: wgpu::RenderPipeline,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    contexts: Vec<TextRenderContext>,
    vertices: Vec<TextVertex>,
    #[allow(dead_code)]
    atlas_width: u32,
    #[allow(dead_code)]
    atlas_height: u32,
}

impl Default for TextRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl TextRenderer {
    /// Creates a new text renderer with `font` as the default font.
    pub fn new() -> TextRenderer {
        let ctxt = Context::get();

        //
        // Create cache.
        //
        let atlas_width = 1024;
        let atlas_height = 1024;
        let cache = Cache::builder()
            .dimensions(atlas_width, atlas_height)
            .build();

        //
        // Create glyph texture (single channel R8).
        //
        let glyph_texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("text_renderer_glyph_texture"),
            size: wgpu::Extent3d {
                width: atlas_width,
                height: atlas_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let glyph_texture_view = glyph_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let glyph_sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("text_renderer_glyph_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        //
        // Create bind group layouts.
        //
        let uniform_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("text_renderer_uniform_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let texture_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("text_renderer_texture_bind_group_layout"),
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
            label: Some("text_renderer_pipeline_layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        //
        // Create shader.
        //
        let shader = ctxt.create_shader_module(
            Some("text_renderer_shader"),
            include_str!("../builtin/text.wgsl"),
        );

        // Vertex buffer layout - interleaved position, UV, and color
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TextVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2, // position
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2, // tex_coord
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4, // color (RGBA)
                },
            ],
        };

        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("text_renderer_pipeline"),
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            depth_stencil: None, // Text rendering doesn't use depth
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create uniform buffer
        let uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text_renderer_uniform_buffer"),
            size: std::mem::size_of::<TextUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create initial vertex buffer
        let vertex_capacity = 1024;
        let vertex_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text_renderer_vertex_buffer"),
            size: (std::mem::size_of::<TextVertex>() * vertex_capacity) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        TextRenderer {
            text: String::new(),
            cache,
            glyph_texture,
            glyph_texture_view,
            glyph_sampler,
            pipeline,
            uniform_bind_group_layout,
            texture_bind_group_layout,
            uniform_buffer,
            vertex_buffer,
            vertex_capacity,
            contexts: Vec::new(),
            vertices: Vec::new(),
            atlas_width,
            atlas_height,
        }
    }

    /// Adds a piece of text to be drawn during the next frame. The text is not persistent between
    /// frames. This method must be called for each text to draw, and at each update loop
    /// iteration.
    pub fn draw_text(
        &mut self,
        text: &str,
        pos: Vec2,
        scale: f32,
        font: &Arc<Font>,
        color: Color,
    ) {
        self.text.push_str(text);
        self.contexts.push(TextRenderContext {
            len: text.len(),
            scale,
            color: [color.r, color.g, color.b, color.a],
            pos,
            font: font.clone(),
        })
    }

    /// Actually draws the text.
    pub fn render(&mut self, width: f32, height: f32, context: &mut RenderContext2dEncoder) {
        if self.contexts.is_empty() {
            return;
        }

        let ctxt = Context::get();

        // Collect all glyphs with their metadata first, then process them
        // This avoids re-creating glyph objects which might not match cache entries
        struct GlyphData {
            glyph: rusttype::PositionedGlyph<'static>,
            font_uid: usize,
            color: [f32; 4],
        }
        let mut all_glyphs: Vec<GlyphData> = Vec::new();

        // First pass: collect all glyphs and queue them for caching
        let mut pos = 0;
        for text_context in self.contexts.iter() {
            let scale = rusttype::Scale::uniform(text_context.scale);
            let vmetrics = text_context.font.font().v_metrics(scale);
            let line_height = vmetrics.ascent - vmetrics.descent;
            let text = &self.text[pos..pos + text_context.len];
            let font_uid = Font::uid(&text_context.font);
            let color = text_context.color;
            let mut vshift = 0.0;

            for line in text.lines() {
                let orig = rusttype::Point {
                    x: text_context.pos.x,
                    y: text_context.pos.y + vmetrics.ascent + vshift,
                };

                vshift += line_height;
                let layout = text_context.font.font().layout(line, scale, orig);

                for glyph in layout {
                    self.cache.queue_glyph(font_uid, glyph.clone());
                    all_glyphs.push(GlyphData {
                        glyph,
                        font_uid,
                        color,
                    });
                }
            }
            pos += text_context.len;
        }

        // Update glyph cache texture with all queued glyphs
        let glyph_texture = &self.glyph_texture;
        let _ = self.cache.cache_queued(|rect, data| {
            ctxt.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: glyph_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: rect.min.x,
                        y: rect.min.y,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(rect.width()),
                    rows_per_image: Some(rect.height()),
                },
                wgpu::Extent3d {
                    width: rect.width(),
                    height: rect.height(),
                    depth_or_array_layers: 1,
                },
            );
        });

        // Second pass: generate vertices using the same glyph objects
        for glyph_data in &all_glyphs {
            if let Ok(Some((tex, px_rect))) =
                self.cache.rect_for(glyph_data.font_uid, &glyph_data.glyph)
            {
                let min_px = px_rect.min.x as f32;
                let min_py = px_rect.min.y as f32;
                let max_px = px_rect.max.x as f32;
                let max_py = px_rect.max.y as f32;
                let color = glyph_data.color;

                // Two triangles per glyph quad
                self.vertices.push(TextVertex {
                    position: [min_px, min_py],
                    tex_coord: [tex.min.x, tex.min.y],
                    color,
                });
                self.vertices.push(TextVertex {
                    position: [min_px, max_py],
                    tex_coord: [tex.min.x, tex.max.y],
                    color,
                });
                self.vertices.push(TextVertex {
                    position: [max_px, min_py],
                    tex_coord: [tex.max.x, tex.min.y],
                    color,
                });

                self.vertices.push(TextVertex {
                    position: [max_px, min_py],
                    tex_coord: [tex.max.x, tex.min.y],
                    color,
                });
                self.vertices.push(TextVertex {
                    position: [min_px, max_py],
                    tex_coord: [tex.min.x, tex.max.y],
                    color,
                });
                self.vertices.push(TextVertex {
                    position: [max_px, max_py],
                    tex_coord: [tex.max.x, tex.max.y],
                    color,
                });
            }
        }

        if self.vertices.is_empty() {
            self.contexts.clear();
            self.text.clear();
            return;
        }

        // Update uniforms
        let uniforms = TextUniforms {
            inv_size: [2.0 / width, -2.0 / height],
            _padding: [0.0, 0.0],
        };
        ctxt.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Ensure vertex buffer is large enough
        let needed = self.vertices.len();
        if needed > self.vertex_capacity {
            let new_capacity = needed.next_power_of_two();
            self.vertex_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
                label: Some("text_renderer_vertex_buffer"),
                size: (std::mem::size_of::<TextVertex>() * new_capacity) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_capacity = new_capacity;
        }

        // Upload vertex data
        ctxt.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));

        // Create bind groups
        let uniform_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("text_renderer_uniform_bind_group"),
            layout: &self.uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniform_buffer.as_entire_binding(),
            }],
        });

        let texture_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("text_renderer_texture_bind_group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.glyph_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.glyph_sampler),
                },
            ],
        });

        // Create render pass and draw all text
        {
            let mut render_pass = context
                .encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("text_renderer_render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: context.color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &uniform_bind_group, &[]);
            render_pass.set_bind_group(1, &texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.vertices.len() as u32, 0..1);
        }

        self.vertices.clear();
        self.contexts.clear();
        self.text.clear();
    }

    #[allow(dead_code)]
    fn ensure_vertex_buffer_capacity(&mut self, _needed: usize) {
        // TODO: Implement dynamic buffer resizing if needed
    }
}

/// Vertex shader used by the material to display text.
#[allow(dead_code)]
pub static TEXT_VERTEX_SRC: &str = include_str!("../builtin/text.wgsl");
/// Fragment shader used by the material to display text.
#[allow(dead_code)]
pub static TEXT_FRAGMENT_SRC: &str = include_str!("../builtin/text.wgsl");
