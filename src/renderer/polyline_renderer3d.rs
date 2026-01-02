//! A polyline renderer with configurable line width.
//!
//! Based on [bevy_polyline](https://github.com/ForesightMiningSoftwareCorporation/bevy_polyline)
//! which uses instanced rendering to draw thick lines efficiently.

use crate::camera::Camera3d;
use crate::context::Context;
use crate::renderer::Renderer3d;
use crate::resource::RenderContext;
use bytemuck::{Pod, Zeroable};
use glamx::{Pose3, Vec3};
use crate::color::Color;

/// A line segment with two endpoints and per-segment material properties.
/// This allows rendering all segments in a single draw call.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct LineSegment {
    point_a: [f32; 3],
    width: f32,
    point_b: [f32; 3],
    depth_bias: f32,
    color: [f32; 4],
    perspective: u32,
    _padding: [u32; 3],
}

/// View uniforms for polyline rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ViewUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    viewport: [f32; 4], // x, y, width, height
}

/// A polyline is a series of connected line segments.
#[derive(Clone, Debug)]
pub struct Polyline3d {
    /// The vertices of the polyline.
    pub vertices: Vec<Vec3>,
    /// The color of the polyline (RGBA, 0-1).
    pub color: Color,
    /// The width of the line in pixels.
    pub width: f32,
    /// Whether to use perspective-correct line width.
    pub perspective: bool,
    /// Depth bias for z-fighting prevention. Range [-1, 1].
    /// Negative values bring the line closer to the camera.
    pub depth_bias: f32,
    /// The model transform for this polyline.
    pub transform: Pose3,
}

impl Default for Polyline3d {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            color: crate::color::WHITE,
            width: 2.0,
            perspective: false,
            depth_bias: 0.0,
            transform: Pose3::IDENTITY,
        }
    }
}

impl Polyline3d {
    /// Creates a new polyline with the given vertices.
    pub fn new(vertices: Vec<Vec3>) -> Self {
        Self {
            vertices,
            ..Default::default()
        }
    }

    /// Sets the color of the polyline.
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Sets the width of the line in pixels.
    pub fn with_width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    /// Enables or disables perspective-correct line width.
    pub fn with_perspective(mut self, perspective: bool) -> Self {
        self.perspective = perspective;
        self
    }

    /// Sets the depth bias for z-fighting prevention.
    pub fn with_depth_bias(mut self, depth_bias: f32) -> Self {
        self.depth_bias = depth_bias;
        self
    }

    /// Sets the model transform for this polyline.
    pub fn with_transform(mut self, transform: Pose3) -> Self {
        self.transform = transform;
        self
    }
}

/// Structure which manages the display of polylines with configurable width.
pub struct PolylineRenderer3d {
    pipeline: wgpu::RenderPipeline,
    view_bind_group_layout: wgpu::BindGroupLayout,
    view_uniform_buffer: wgpu::Buffer,
    segment_buffer: wgpu::Buffer,
    segment_capacity: usize,
    /// Pre-built segments ready for rendering (avoids reallocations)
    segments: Vec<LineSegment>,
}

impl Default for PolylineRenderer3d {
    fn default() -> Self {
        Self::new()
    }
}

impl PolylineRenderer3d {
    /// Creates a new polyline renderer.
    pub fn new() -> PolylineRenderer3d {
        let ctxt = Context::get();

        // Create view bind group layout (group 0)
        let view_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("polyline_view_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("polyline_pipeline_layout"),
            bind_group_layouts: &[&view_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Load shader
        let shader = ctxt.create_shader_module(
            Some("polyline_shader"),
            include_str!("../builtin/polyline3d.wgsl"),
        );

        // Vertex buffer layout - each instance is a line segment with material data
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineSegment>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // point_a (vec3)
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // width (f32)
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32,
                },
                // point_b (vec3)
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // depth_bias (f32)
                wgpu::VertexAttribute {
                    offset: 28,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
                // color (vec4)
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // perspective (u32)
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        };

        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("polyline_pipeline"),
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Context::depth_format(),
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create view uniform buffer
        let view_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("polyline_view_uniform_buffer"),
            size: std::mem::size_of::<ViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create initial segment buffer
        let segment_capacity = 1024;
        let segment_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("polyline_segment_buffer"),
            size: (std::mem::size_of::<LineSegment>() * segment_capacity) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        PolylineRenderer3d {
            pipeline,
            view_bind_group_layout,
            view_uniform_buffer,
            segment_buffer,
            segment_capacity,
            segments: Vec::new(),
        }
    }

    /// Indicates whether some polylines need to be rendered.
    pub fn needs_rendering(&self) -> bool {
        !self.segments.is_empty()
    }

    /// Adds a polyline to be drawn during the next frame.
    /// Takes a reference to avoid allocations - segments are built immediately.
    /// Polylines are not persistent between frames.
    pub fn draw_polyline(&mut self, polyline: &Polyline3d) {
        if polyline.vertices.len() < 2 {
            return;
        }

        let transform = polyline.transform;
        let color = [polyline.color.r, polyline.color.g, polyline.color.b, polyline.color.a];
        let width = polyline.width;
        let depth_bias = polyline.depth_bias;
        let perspective = if polyline.perspective { 1 } else { 0 };

        for pair in polyline.vertices.windows(2) {
            let a = transform * pair[0];
            let b = transform * pair[1];
            self.segments.push(LineSegment {
                point_a: a.into(),
                width,
                point_b: b.into(),
                depth_bias,
                color,
                perspective,
                _padding: [0; 3],
            });
        }
    }

    /// Draws a simple line segment with the given width.
    pub fn draw_line(
        &mut self,
        a: Vec3,
        b: Vec3,
        color: Color,
        width: f32,
        perspective: bool,
    ) {
        self.segments.push(LineSegment {
            point_a: a.into(),
            width,
            point_b: b.into(),
            depth_bias: 0.0,
            color: [color.r, color.g, color.b, color.a],
            perspective: perspective as u32,
            _padding: [0; 3],
        });
    }

    fn ensure_segment_buffer_capacity(&mut self, needed: usize) {
        if needed > self.segment_capacity {
            let ctxt = Context::get();
            let new_capacity = needed.next_power_of_two();
            self.segment_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
                label: Some("polyline_segment_buffer"),
                size: (std::mem::size_of::<LineSegment>() * new_capacity) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.segment_capacity = new_capacity;
        }
    }

    fn create_view_bind_group(&self) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("polyline_view_bind_group"),
            layout: &self.view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.view_uniform_buffer.as_entire_binding(),
            }],
        })
    }
}

impl Renderer3d for PolylineRenderer3d {
    /// Renders all polylines in a single draw call.
    fn render(
        &mut self,
        pass: usize,
        camera: &mut dyn Camera3d,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext,
    ) {
        if self.segments.is_empty() {
            return;
        }

        let ctxt = Context::get();

        // Get camera matrices and viewport
        let (view, proj) = camera.view_transform_pair(pass);

        // Update view uniforms
        let view_uniforms = ViewUniforms {
            view: view.to_mat4().to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            viewport: [
                0.0,
                0.0,
                context.viewport_width as f32,
                context.viewport_height as f32,
            ],
        };
        ctxt.write_buffer(
            &self.view_uniform_buffer,
            0,
            bytemuck::bytes_of(&view_uniforms),
        );

        // Ensure buffer capacity for all segments
        self.ensure_segment_buffer_capacity(self.segments.len());

        // Upload all segment data at once
        ctxt.write_buffer(
            &self.segment_buffer,
            0,
            bytemuck::cast_slice(&self.segments),
        );

        // Create view bind group
        let view_bind_group = self.create_view_bind_group();

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &view_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.segment_buffer.slice(..));

        // Draw all polylines in a single call
        let num_segments = self.segments.len() as u32;
        render_pass.draw(0..6, 0..num_segments);

        // Clear segments for next frame
        self.segments.clear();
    }
}

/// Vertex shader source for polylines.
pub static POLYLINE_SHADER_SRC: &str = include_str!("../builtin/polyline3d.wgsl");
