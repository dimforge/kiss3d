//! A batched point renderer.

use crate::camera::Camera3d;
use crate::color::Color;
use crate::context::Context;
use crate::renderer::Renderer3d;
use crate::resource::RenderContext;
use bytemuck::{Pod, Zeroable};
use glamx::Vec3;

/// Point data for storage buffer (position + size + color).
/// Layout must match points.wgsl PointData struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PointData {
    position: [f32; 3],
    size: f32, // Per-point size (uses default if <= 0)
    color: [f32; 4],
}

/// Frame uniforms for point rendering.
/// Layout must match points.wgsl FrameUniforms struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FrameUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    viewport: [f32; 4],
}

/// Structure which manages the display of short-living points.
pub struct PointRenderer3d {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    frame_uniform_buffer: wgpu::Buffer,
    point_storage_buffer: wgpu::Buffer,
    point_capacity: usize,
    points: Vec<PointData>,
}

impl Default for PointRenderer3d {
    fn default() -> Self {
        Self::new()
    }
}

impl PointRenderer3d {
    /// Creates a new points manager.
    pub fn new() -> PointRenderer3d {
        let ctxt = Context::get();

        // Create bind group layout with uniform buffer and storage buffer
        let bind_group_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point_renderer_bind_group_layout"),
            entries: &[
                // Frame uniforms (binding 0)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Point data storage buffer (binding 1)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("point_renderer_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Load shader
        let shader = ctxt.create_shader_module(
            Some("point_renderer_shader"),
            include_str!("../builtin/points3d.wgsl"),
        );

        // No vertex buffers - using storage buffer and vertex_index
        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point_renderer_pipeline"),
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
                depth_compare: wgpu::CompareFunction::Less,
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

        // Create uniform buffer
        let frame_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("point_renderer_frame_uniform_buffer"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create initial storage buffer for point data
        let point_capacity = 1024;
        let point_storage_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("point_renderer_storage_buffer"),
            size: (std::mem::size_of::<PointData>() * point_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        PointRenderer3d {
            pipeline,
            bind_group_layout,
            frame_uniform_buffer,
            point_storage_buffer,
            point_capacity,
            points: Vec::new(),
        }
    }

    /// Indicates whether some points have to be drawn.
    pub fn needs_rendering(&self) -> bool {
        !self.points.is_empty()
    }

    /// Adds a point to be drawn during the next frame. Points are not persistent between frames.
    /// This method must be called for each point to draw, and at each update loop iteration.
    pub fn draw_point(&mut self, pt: Vec3, color: Color, size: f32) {
        self.points.push(PointData {
            position: pt.into(),
            size,
            color: [color.r, color.g, color.b, color.a],
        });
    }

    fn ensure_storage_buffer_capacity(&mut self, needed: usize) {
        if needed > self.point_capacity {
            let ctxt = Context::get();
            let new_capacity = needed.next_power_of_two();
            self.point_storage_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
                label: Some("point_renderer_storage_buffer"),
                size: (std::mem::size_of::<PointData>() * new_capacity) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.point_capacity = new_capacity;
        }
    }

    fn create_bind_group(&self) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point_renderer_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.frame_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.point_storage_buffer.as_entire_binding(),
                },
            ],
        })
    }
}

impl Renderer3d for PointRenderer3d {
    /// Actually draws the points.
    fn render(
        &mut self,
        pass: usize,
        camera: &mut dyn Camera3d,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext,
    ) {
        if self.points.is_empty() {
            return;
        }

        let ctxt = Context::get();

        // Get camera matrices
        let (view, proj) = camera.view_transform_pair(pass);

        // Update frame uniforms
        let frame_uniforms = FrameUniforms {
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
            &self.frame_uniform_buffer,
            0,
            bytemuck::bytes_of(&frame_uniforms),
        );

        // Ensure storage buffer is large enough
        self.ensure_storage_buffer_capacity(self.points.len());

        // Upload point data
        ctxt.write_buffer(
            &self.point_storage_buffer,
            0,
            bytemuck::cast_slice(&self.points),
        );

        // Create bind group
        let bind_group = self.create_bind_group();

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);

        // Draw 6 vertices per point (2 triangles forming a quad)
        let num_vertices = (self.points.len() * 6) as u32;
        render_pass.draw(0..num_vertices, 0..1);

        // Clear points for next frame
        self.points.clear();
    }
}

/// Vertex shader used by the material to display point.
pub static POINTS_VERTEX_SRC: &str = include_str!("../builtin/points3d.wgsl");
/// Fragment shader used by the material to display point.
pub static POINTS_FRAGMENT_SRC: &str = include_str!("../builtin/points3d.wgsl");
