//! A 2D polyline renderer with configurable line width.
//!
//! Similar to the 3D polyline renderer but for 2D planar scenes.

use crate::context::Context;
use crate::planar_camera::PlanarCamera;
use crate::resource::PlanarRenderContext;
use bytemuck::{Pod, Zeroable};
use na::{Isometry2, Matrix3, Point2, Point3};

/// A 2D line segment with endpoints and per-segment material properties.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct LineSegment2D {
    point_a: [f32; 2],
    width: f32,
    _pad1: f32,
    point_b: [f32; 2],
    _pad2: [f32; 2],
    color: [f32; 4],
}

/// View uniforms for 2D polyline rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ViewUniforms2D {
    // mat3x3 stored as 3x vec4 for alignment
    view: [[f32; 4]; 3],
    proj: [[f32; 4]; 3],
    viewport: [f32; 4], // x, y, width, height
}

/// A 2D polyline is a series of connected line segments.
#[derive(Clone, Debug)]
pub struct PlanarPolyline {
    /// The vertices of the polyline.
    pub vertices: Vec<Point2<f32>>,
    /// The color of the polyline (RGB, 0-1).
    pub color: Point3<f32>,
    /// The width of the line in pixels.
    pub width: f32,
    /// The model transform for this polyline.
    pub transform: Isometry2<f32>,
}

impl Default for PlanarPolyline {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            color: Point3::new(1.0, 1.0, 1.0),
            width: 2.0,
            transform: Isometry2::identity(),
        }
    }
}

impl PlanarPolyline {
    /// Creates a new 2D polyline with the given vertices.
    pub fn new(vertices: Vec<Point2<f32>>) -> Self {
        Self {
            vertices,
            ..Default::default()
        }
    }

    /// Sets the color of the polyline.
    pub fn with_color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.color = Point3::new(r, g, b);
        self
    }

    /// Sets the width of the line in pixels.
    pub fn with_width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    /// Sets the model transform for this polyline.
    pub fn with_transform(mut self, transform: Isometry2<f32>) -> Self {
        self.transform = transform;
        self
    }
}

/// Structure which manages the display of 2D polylines with configurable width.
pub struct PlanarPolylineRenderer {
    pipeline: wgpu::RenderPipeline,
    view_bind_group_layout: wgpu::BindGroupLayout,
    view_uniform_buffer: wgpu::Buffer,
    segment_buffer: wgpu::Buffer,
    segment_capacity: usize,
    /// Pre-built segments ready for rendering
    segments: Vec<LineSegment2D>,
}

impl Default for PlanarPolylineRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanarPolylineRenderer {
    /// Creates a new 2D polyline renderer.
    pub fn new() -> PlanarPolylineRenderer {
        let ctxt = Context::get();

        // Create bind group layout
        let view_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("planar_polyline_view_bind_group_layout"),
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
            label: Some("planar_polyline_pipeline_layout"),
            bind_group_layouts: &[&view_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Load shader
        let shader = ctxt.create_shader_module(
            Some("planar_polyline_shader"),
            include_str!("builtin/planar_polyline.wgsl"),
        );

        // Vertex buffer layout - each instance is a line segment with material data
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineSegment2D>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // point_a (vec2)
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // width (f32)
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32,
                },
                // point_b (vec2) - offset 16 (after point_a[2] + width + _pad1)
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // color (vec4) - offset 32 (after point_b[2] + _pad2[2])
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("planar_polyline_pipeline"),
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
            depth_stencil: None, // 2D rendering doesn't use depth
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
            label: Some("planar_polyline_view_uniform_buffer"),
            size: std::mem::size_of::<ViewUniforms2D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create initial segment buffer
        let segment_capacity = 1024;
        let segment_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planar_polyline_segment_buffer"),
            size: (std::mem::size_of::<LineSegment2D>() * segment_capacity) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        PlanarPolylineRenderer {
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

    /// Adds a 2D polyline to be drawn during the next frame.
    /// Takes a reference to avoid allocations - segments are built immediately.
    /// Polylines are not persistent between frames.
    pub fn draw_polyline(&mut self, polyline: &PlanarPolyline) {
        if polyline.vertices.len() < 2 {
            return;
        }

        let transform = polyline.transform;
        let color = [polyline.color.x, polyline.color.y, polyline.color.z, 1.0];
        let width = polyline.width;

        for pair in polyline.vertices.windows(2) {
            let a = transform * pair[0];
            let b = transform * pair[1];
            self.segments.push(LineSegment2D {
                point_a: a.coords.into(),
                width,
                _pad1: 0.0,
                point_b: b.coords.into(),
                _pad2: [0.0; 2],
                color,
            });
        }
    }

    /// Draws a simple 2D line segment with the given width.
    pub fn draw_line(&mut self, a: Point2<f32>, b: Point2<f32>, color: Point3<f32>, width: f32) {
        self.segments.push(LineSegment2D {
            point_a: a.coords.into(),
            width,
            _pad1: 0.0,
            point_b: b.coords.into(),
            _pad2: [0.0; 2],
            color: [color.x, color.y, color.z, 1.0],
        });
    }

    /// Renders all 2D polylines in a single draw call.
    pub fn render(&mut self, camera: &mut dyn PlanarCamera, context: &mut PlanarRenderContext) {
        if self.segments.is_empty() {
            return;
        }

        let ctxt = Context::get();

        // Get camera matrices
        let (view, proj) = camera.view_transform_pair();

        // Update view uniforms
        let view_uniforms = ViewUniforms2D {
            view: Self::mat3_to_padded(&view),
            proj: Self::mat3_to_padded(&proj),
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

        // Create bind group
        let view_bind_group = self.create_view_bind_group();

        // Create render pass (no depth for 2D)
        {
            let mut render_pass = context
                .encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("planar_polyline_render_pass"),
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
            render_pass.set_bind_group(0, &view_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.segment_buffer.slice(..));

            // Single draw call: 6 vertices per segment (2 triangles), all segments as instances
            render_pass.draw(0..6, 0..self.segments.len() as u32);
        }

        // Clear segments for next frame
        self.segments.clear();
    }

    fn ensure_segment_buffer_capacity(&mut self, needed: usize) {
        if needed > self.segment_capacity {
            let ctxt = Context::get();
            let new_capacity = needed.next_power_of_two();
            self.segment_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
                label: Some("planar_polyline_segment_buffer"),
                size: (std::mem::size_of::<LineSegment2D>() * new_capacity) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.segment_capacity = new_capacity;
        }
    }

    fn create_view_bind_group(&self) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planar_polyline_view_bind_group"),
            layout: &self.view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.view_uniform_buffer.as_entire_binding(),
            }],
        })
    }

    /// Helper to convert mat3x3 to padded array for uniforms.
    fn mat3_to_padded(m: &Matrix3<f32>) -> [[f32; 4]; 3] {
        [
            [m[(0, 0)], m[(1, 0)], m[(2, 0)], 0.0],
            [m[(0, 1)], m[(1, 1)], m[(2, 1)], 0.0],
            [m[(0, 2)], m[(1, 2)], m[(2, 2)], 0.0],
        ]
    }
}
