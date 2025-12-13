use crate::context::Context;
use crate::planar_camera::PlanarCamera;
use crate::resource::vertex_index::VERTEX_INDEX_FORMAT;
use crate::resource::{GpuData, PlanarMaterial, PlanarMesh, PlanarRenderContext, Texture};
use crate::scene::{PlanarInstancesBuffer, PlanarObjectData};
use bytemuck::{Pod, Zeroable};
use na::{Isometry2, Matrix2, Matrix3, Point2, Vector2};
use std::any::Any;

/// Frame-level uniforms (view, projection) for 2D rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FrameUniforms {
    // mat3x3 stored as 3x vec4 for alignment (each column padded to vec4)
    view: [[f32; 4]; 3],
    proj: [[f32; 4]; 3],
}

/// Object-level uniforms (model, scale, color) for 2D rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ObjectUniforms {
    // mat3x3 stored as 3x vec4 for alignment
    model: [[f32; 4]; 3],
    // mat2x2 stored as 2x vec4 for alignment
    scale: [[f32; 4]; 2],
    color: [f32; 3],
    _padding: f32,
}

/// View uniforms for wireframe rendering (includes viewport).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WireframeViewUniforms {
    // mat3x3 stored as 3x vec4 for alignment
    view: [[f32; 4]; 3],
    proj: [[f32; 4]; 3],
    viewport: [f32; 4], // x, y, width, height
}

/// Model uniforms for wireframe rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WireframeModelUniforms {
    // mat3x3 stored as 3x vec4 for alignment
    model: [[f32; 4]; 3],
    // mat2x2 stored as 2x vec4 for alignment
    scale: [[f32; 4]; 2],
    num_edges: u32,
    default_width: f32,
    use_perspective: u32,
    _padding1: f32,
    default_color: [f32; 4],
}

/// GPU format for 2D edges.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuEdge2D {
    point_a: [f32; 2],
    point_b: [f32; 2],
}

/// Model uniforms for point rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PointsModelUniforms {
    // mat3x3 stored as 3x vec4 for alignment
    model: [[f32; 4]; 3],
    // mat2x2 stored as 2x vec4 for alignment
    scale: [[f32; 4]; 2],
    num_vertices: u32,
    default_size: f32,
    use_perspective: u32,
    _padding1: f32,
    default_color: [f32; 4],
}

/// GPU format for 2D vertex.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuVertex2D {
    position: [f32; 2],
}

/// Per-object GPU data for PlanarObjectMaterial.
pub struct PlanarObjectMaterialGpuData {
    frame_uniform_buffer: wgpu::Buffer,
    object_uniform_buffer: wgpu::Buffer,
    // Cached bind groups for surface rendering
    frame_bind_group: Option<wgpu::BindGroup>,
    object_bind_group: Option<wgpu::BindGroup>,
    texture_bind_group: Option<wgpu::BindGroup>,
    cached_texture_ptr: usize,
    // Wireframe rendering data
    wireframe_view_uniform_buffer: wgpu::Buffer,
    wireframe_model_uniform_buffer: wgpu::Buffer,
    wireframe_edge_buffer: wgpu::Buffer,
    wireframe_edge_capacity: usize,
    wireframe_view_bind_group: Option<wgpu::BindGroup>,
    wireframe_model_bind_group: Option<wgpu::BindGroup>,
    /// Cached wireframe edges in local coordinates (built lazily from mesh).
    wireframe_edges: Option<Vec<(Point2<f32>, Point2<f32>)>>,
    /// Hash of mesh faces to detect when edges need rebuilding.
    wireframe_edges_mesh_hash: u64,
    // Point rendering data
    points_view_uniform_buffer: wgpu::Buffer,
    points_model_uniform_buffer: wgpu::Buffer,
    points_vertex_buffer: wgpu::Buffer,
    points_vertex_capacity: usize,
    points_view_bind_group: Option<wgpu::BindGroup>,
    points_model_bind_group: Option<wgpu::BindGroup>,
    /// Cached vertices for point rendering (built lazily from mesh).
    points_vertices: Option<Vec<Point2<f32>>>,
    /// Hash of mesh coords to detect when vertices need rebuilding.
    points_vertices_mesh_hash: u64,
}

impl PlanarObjectMaterialGpuData {
    pub fn new() -> Self {
        let ctxt = Context::get();

        let frame_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planar_material_frame_uniform_buffer"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let object_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planar_material_object_uniform_buffer"),
            size: std::mem::size_of::<ObjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Wireframe buffers
        let wireframe_view_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planar_wireframe_view_uniform_buffer"),
            size: std::mem::size_of::<WireframeViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let wireframe_model_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planar_wireframe_model_uniform_buffer"),
            size: std::mem::size_of::<WireframeModelUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let wireframe_edge_capacity = 256;
        let wireframe_edge_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planar_wireframe_edge_buffer"),
            size: (std::mem::size_of::<GpuEdge2D>() * wireframe_edge_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Point rendering buffers (reuse same view uniform format as wireframe)
        let points_view_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planar_points_view_uniform_buffer"),
            size: std::mem::size_of::<WireframeViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let points_model_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planar_points_model_uniform_buffer"),
            size: std::mem::size_of::<PointsModelUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let points_vertex_capacity = 256;
        let points_vertex_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planar_points_vertex_buffer"),
            size: (std::mem::size_of::<GpuVertex2D>() * points_vertex_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            frame_uniform_buffer,
            object_uniform_buffer,
            frame_bind_group: None,
            object_bind_group: None,
            texture_bind_group: None,
            cached_texture_ptr: 0,
            wireframe_view_uniform_buffer,
            wireframe_model_uniform_buffer,
            wireframe_edge_buffer,
            wireframe_edge_capacity,
            wireframe_view_bind_group: None,
            wireframe_model_bind_group: None,
            wireframe_edges: None,
            wireframe_edges_mesh_hash: 0,
            points_view_uniform_buffer,
            points_model_uniform_buffer,
            points_vertex_buffer,
            points_vertex_capacity,
            points_view_bind_group: None,
            points_model_bind_group: None,
            points_vertices: None,
            points_vertices_mesh_hash: 0,
        }
    }

    /// Ensure edge buffer has enough capacity, reallocating if needed.
    fn ensure_edge_buffer_capacity(&mut self, needed: usize) {
        if needed > self.wireframe_edge_capacity {
            let ctxt = Context::get();
            let new_capacity = needed.next_power_of_two();
            self.wireframe_edge_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
                label: Some("planar_wireframe_edge_buffer"),
                size: (std::mem::size_of::<GpuEdge2D>() * new_capacity) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.wireframe_edge_capacity = new_capacity;
            // Invalidate bind group since buffer changed
            self.wireframe_model_bind_group = None;
        }
    }

    /// Ensure vertex buffer for points has enough capacity, reallocating if needed.
    fn ensure_vertex_buffer_capacity(&mut self, needed: usize) {
        if needed > self.points_vertex_capacity {
            let ctxt = Context::get();
            let new_capacity = needed.next_power_of_two();
            self.points_vertex_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
                label: Some("planar_points_vertex_buffer"),
                size: (std::mem::size_of::<GpuVertex2D>() * new_capacity) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.points_vertex_capacity = new_capacity;
            // Invalidate bind group since buffer changed
            self.points_model_bind_group = None;
        }
    }
}

impl Default for PlanarObjectMaterialGpuData {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuData for PlanarObjectMaterialGpuData {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// The default material used to draw 2D objects.
pub struct PlanarObjectMaterial {
    pipeline: wgpu::RenderPipeline,
    frame_bind_group_layout: wgpu::BindGroupLayout,
    object_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    // Wireframe pipeline and layouts
    wireframe_pipeline: wgpu::RenderPipeline,
    wireframe_view_bind_group_layout: wgpu::BindGroupLayout,
    wireframe_model_bind_group_layout: wgpu::BindGroupLayout,
    // Points pipeline and layouts
    points_pipeline: wgpu::RenderPipeline,
    points_view_bind_group_layout: wgpu::BindGroupLayout,
    points_model_bind_group_layout: wgpu::BindGroupLayout,
}

impl Default for PlanarObjectMaterial {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanarObjectMaterial {
    /// Creates a new `PlanarObjectMaterial`.
    pub fn new() -> PlanarObjectMaterial {
        let ctxt = Context::get();

        // Create bind group layouts
        let frame_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("planar_material_frame_bind_group_layout"),
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

        let object_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("planar_material_object_bind_group_layout"),
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
                label: Some("planar_material_texture_bind_group_layout"),
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
            label: Some("planar_material_pipeline_layout"),
            bind_group_layouts: &[
                &frame_bind_group_layout,
                &object_bind_group_layout,
                &texture_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Load shader
        let shader =
            ctxt.create_shader_module(Some("planar_material_shader"), include_str!("planar.wgsl"));

        // Vertex buffer layouts
        // Note: We use separate buffers for instance data (positions, colors, deformations)
        // instead of interleaving them, to avoid per-frame data conversion overhead.
        let vertex_buffer_layouts = [
            // Buffer 0: Vertex positions (vec2)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                }],
            },
            // Buffer 1: Texture coordinates (vec2)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                }],
            },
            // Buffer 2: Instance positions (Point2<f32>)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2, // inst_tra
                    format: wgpu::VertexFormat::Float32x2,
                }],
            },
            // Buffer 3: Instance colors ([f32; 4])
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3, // inst_color
                    format: wgpu::VertexFormat::Float32x4,
                }],
            },
            // Buffer 4: Instance deformations (2x Vector2<f32> = 2 columns of 2x2 matrix)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress, // 2 vec2s
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    // inst_def_0 (column 0)
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    // inst_def_1 (column 1)
                    wgpu::VertexAttribute {
                        offset: 8, // 2 * sizeof(f32)
                        shader_location: 5,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                ],
            },
        ];

        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("planar_material_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffer_layouts,
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
                cull_mode: None, // 2D objects typically don't need culling
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // 2D rendering typically doesn't use depth
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create wireframe bind group layouts
        let wireframe_view_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("planar_wireframe_view_bind_group_layout"),
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

        let wireframe_model_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("planar_wireframe_model_bind_group_layout"),
                entries: &[
                    // Model uniforms
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
                    // Edge storage buffer
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

        let wireframe_pipeline_layout =
            ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("planar_wireframe_pipeline_layout"),
                bind_group_layouts: &[
                    &wireframe_view_bind_group_layout,
                    &wireframe_model_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        // Load wireframe shader
        let wireframe_shader = ctxt.create_shader_module(
            Some("planar_wireframe_shader"),
            include_str!("wireframe_planar_polyline.wgsl"),
        );

        // Wireframe instance vertex buffer layouts
        let wireframe_instance_buffer_layouts = [
            // Buffer 0: positions (Point2<f32>)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                }],
            },
            // Buffer 1: colors ([f32; 4]) - not used for wireframe but needed for consistency
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                }],
            },
            // Buffer 2: deformations - both columns from same buffer with stride = 2*vec2
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress, // 2 vec2s = 16 bytes
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    // Column 0 at offset 0
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    // Column 1 at offset 8
                    wgpu::VertexAttribute {
                        offset: 8,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                ],
            },
            // Buffer 3: lines_colors ([f32; 4])
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                }],
            },
            // Buffer 4: lines_widths (f32)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32,
                }],
            },
        ];

        let wireframe_pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("planar_wireframe_pipeline"),
            layout: Some(&wireframe_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &wireframe_shader,
                entry_point: Some("vs_main"),
                buffers: &wireframe_instance_buffer_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &wireframe_shader,
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create points bind group layouts (same view layout as wireframe)
        let points_view_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("planar_points_view_bind_group_layout"),
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

        let points_model_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("planar_points_model_bind_group_layout"),
                entries: &[
                    // Model uniforms
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
                    // Vertex storage buffer
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

        let points_pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planar_points_pipeline_layout"),
            bind_group_layouts: &[
                &points_view_bind_group_layout,
                &points_model_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Load points shader
        let points_shader = ctxt.create_shader_module(
            Some("planar_points_shader"),
            include_str!("wireframe_planar_points.wgsl"),
        );

        // Points instance vertex buffer layouts (same as wireframe but with points_colors/sizes)
        let points_instance_buffer_layouts = [
            // Buffer 0: positions (Point2<f32>)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                }],
            },
            // Buffer 1: colors ([f32; 4]) - not used for points but needed for consistency
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                }],
            },
            // Buffer 2: deformations - both columns from same buffer with stride = 2*vec2
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    // Column 0 at offset 0
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    // Column 1 at offset 8
                    wgpu::VertexAttribute {
                        offset: 8,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                ],
            },
            // Buffer 3: points_colors ([f32; 4])
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                }],
            },
            // Buffer 4: points_sizes (f32)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32,
                }],
            },
        ];

        let points_pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("planar_points_pipeline"),
            layout: Some(&points_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &points_shader,
                entry_point: Some("vs_main"),
                buffers: &points_instance_buffer_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &points_shader,
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        PlanarObjectMaterial {
            pipeline,
            frame_bind_group_layout,
            object_bind_group_layout,
            texture_bind_group_layout,
            wireframe_pipeline,
            wireframe_view_bind_group_layout,
            wireframe_model_bind_group_layout,
            points_pipeline,
            points_view_bind_group_layout,
            points_model_bind_group_layout,
        }
    }

    fn create_frame_bind_group(&self, buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planar_material_frame_bind_group"),
            layout: &self.frame_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }

    fn create_object_bind_group(&self, buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planar_material_object_bind_group"),
            layout: &self.object_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }

    fn create_texture_bind_group(&self, texture: &Texture) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planar_material_texture_bind_group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ],
        })
    }

    fn create_wireframe_view_bind_group(&self, buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planar_wireframe_view_bind_group"),
            layout: &self.wireframe_view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }

    fn create_wireframe_model_bind_group(
        &self,
        uniform_buffer: &wgpu::Buffer,
        edge_buffer: &wgpu::Buffer,
        edge_size: u64,
    ) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planar_wireframe_model_bind_group"),
            layout: &self.wireframe_model_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: edge_buffer,
                        offset: 0,
                        size: Some(std::num::NonZeroU64::new(edge_size).unwrap()),
                    }),
                },
            ],
        })
    }

    fn create_points_view_bind_group(&self, buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planar_points_view_bind_group"),
            layout: &self.points_view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }

    fn create_points_model_bind_group(
        &self,
        uniform_buffer: &wgpu::Buffer,
        vertex_buffer: &wgpu::Buffer,
        vertex_size: u64,
    ) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planar_points_model_bind_group"),
            layout: &self.points_model_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: vertex_buffer,
                        offset: 0,
                        size: Some(std::num::NonZeroU64::new(vertex_size).unwrap()),
                    }),
                },
            ],
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

    /// Helper to convert mat2x2 to padded array for uniforms.
    fn mat2_to_padded(m: &Matrix2<f32>) -> [[f32; 4]; 2] {
        [
            [m[(0, 0)], m[(1, 0)], 0.0, 0.0],
            [m[(0, 1)], m[(1, 1)], 0.0, 0.0],
        ]
    }
}

impl PlanarMaterial for PlanarObjectMaterial {
    fn create_gpu_data(&self) -> Box<dyn GpuData> {
        Box::new(PlanarObjectMaterialGpuData::new())
    }

    fn render(
        &mut self,
        transform: &Isometry2<f32>,
        scale: &Vector2<f32>,
        camera: &mut dyn PlanarCamera,
        data: &PlanarObjectData,
        mesh: &mut PlanarMesh,
        instances: &mut PlanarInstancesBuffer,
        gpu_data: &mut dyn GpuData,
        context: &mut PlanarRenderContext,
    ) {
        let ctxt = Context::get();

        // Downcast gpu_data to our specific type
        let gpu_data = gpu_data
            .as_any_mut()
            .downcast_mut::<PlanarObjectMaterialGpuData>()
            .expect("PlanarObjectMaterial requires PlanarObjectMaterialGpuData");

        // Get camera matrices
        let (view, proj) = camera.view_transform_pair();

        // Load instance data directly to GPU without conversion
        let num_instances = instances.len();
        instances.positions.load_to_gpu();
        instances.colors.load_to_gpu();
        instances.deformations.load_to_gpu();
        instances.lines_colors.load_to_gpu();
        instances.lines_widths.load_to_gpu();
        instances.points_colors.load_to_gpu();
        instances.points_sizes.load_to_gpu();

        // Ensure mesh buffers are on GPU
        mesh.load_to_gpu();

        // Get instance buffers
        let inst_positions_buf = match instances.positions.buffer() {
            Some(b) => b,
            None => return,
        };
        let inst_colors_buf = match instances.colors.buffer() {
            Some(b) => b,
            None => return,
        };
        let inst_deformations_buf = match instances.deformations.buffer() {
            Some(b) => b,
            None => return,
        };
        let inst_lines_colors_buf = match instances.lines_colors.buffer() {
            Some(b) => b,
            None => return,
        };
        let inst_lines_widths_buf = match instances.lines_widths.buffer() {
            Some(b) => b,
            None => return,
        };
        let inst_points_colors_buf = match instances.points_colors.buffer() {
            Some(b) => b,
            None => return,
        };
        let inst_points_sizes_buf = match instances.points_sizes.buffer() {
            Some(b) => b,
            None => return,
        };

        // Surface rendering
        if data.surface_rendering_active() {
            // Update frame uniforms
            let frame_uniforms = FrameUniforms {
                view: Self::mat3_to_padded(&view),
                proj: Self::mat3_to_padded(&proj),
            };
            ctxt.write_buffer(
                &gpu_data.frame_uniform_buffer,
                0,
                bytemuck::bytes_of(&frame_uniforms),
            );

            // Update object uniforms
            let formatted_transform = transform.to_homogeneous();
            let formatted_scale = Matrix2::from_diagonal(&Vector2::new(scale.x, scale.y));

            let object_uniforms = ObjectUniforms {
                model: Self::mat3_to_padded(&formatted_transform),
                scale: Self::mat2_to_padded(&formatted_scale),
                color: (*data.color()).into(),
                _padding: 0.0,
            };
            ctxt.write_buffer(
                &gpu_data.object_uniform_buffer,
                0,
                bytemuck::bytes_of(&object_uniforms),
            );

            let coords_buffer = mesh.coords().read().unwrap();
            let uvs_buffer = mesh.uvs().read().unwrap();
            let faces_buffer = mesh.faces().read().unwrap();

            let coords_buf = match coords_buffer.buffer() {
                Some(b) => b,
                None => return,
            };
            let uvs_buf = match uvs_buffer.buffer() {
                Some(b) => b,
                None => return,
            };
            let faces_buf = match faces_buffer.buffer() {
                Some(b) => b,
                None => return,
            };

            // Get or create cached bind groups
            if gpu_data.frame_bind_group.is_none() {
                gpu_data.frame_bind_group =
                    Some(self.create_frame_bind_group(&gpu_data.frame_uniform_buffer));
            }
            if gpu_data.object_bind_group.is_none() {
                gpu_data.object_bind_group =
                    Some(self.create_object_bind_group(&gpu_data.object_uniform_buffer));
            }

            // Cache texture bind group, invalidate if texture changed
            let texture_ptr = std::sync::Arc::as_ptr(data.texture()) as usize;
            if gpu_data.texture_bind_group.is_none() || gpu_data.cached_texture_ptr != texture_ptr {
                gpu_data.texture_bind_group = Some(self.create_texture_bind_group(data.texture()));
                gpu_data.cached_texture_ptr = texture_ptr;
            }

            let frame_bind_group = gpu_data.frame_bind_group.as_ref().unwrap();
            let object_bind_group = gpu_data.object_bind_group.as_ref().unwrap();
            let texture_bind_group = gpu_data.texture_bind_group.as_ref().unwrap();

            // Create render pass (no depth for 2D)
            {
                let mut render_pass =
                    context
                        .encoder
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("planar_material_render_pass"),
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
                render_pass.set_bind_group(0, frame_bind_group, &[]);
                render_pass.set_bind_group(1, object_bind_group, &[]);
                render_pass.set_bind_group(2, texture_bind_group, &[]);

                // Set vertex buffers for mesh data
                render_pass.set_vertex_buffer(0, coords_buf.slice(..));
                render_pass.set_vertex_buffer(1, uvs_buf.slice(..));

                // Set instance buffers directly (no per-frame conversion needed)
                render_pass.set_vertex_buffer(2, inst_positions_buf.slice(..));
                render_pass.set_vertex_buffer(3, inst_colors_buf.slice(..));
                render_pass.set_vertex_buffer(4, inst_deformations_buf.slice(..));

                render_pass.set_index_buffer(faces_buf.slice(..), VERTEX_INDEX_FORMAT);

                render_pass.draw_indexed(0..mesh.num_indices(), 0, 0..num_instances as u32);
            }
        }

        // Wireframe rendering
        if data.lines_width() > 0.0 {
            // Build edges from mesh if needed
            let faces_len = mesh.faces().read().unwrap().len();
            let faces_hash = faces_len as u64;

            if gpu_data.wireframe_edges.is_none()
                || gpu_data.wireframe_edges_mesh_hash != faces_hash
            {
                let coords_guard = mesh.coords().read().unwrap();
                let faces_guard = mesh.faces().read().unwrap();

                if let (Some(coords), Some(faces)) = (coords_guard.data(), faces_guard.data()) {
                    let mut edges = Vec::new();
                    for face in faces.iter() {
                        let idx_a = face.x as usize;
                        let idx_b = face.y as usize;
                        let idx_c = face.z as usize;
                        if idx_a < coords.len() && idx_b < coords.len() && idx_c < coords.len() {
                            edges.push((coords[idx_a], coords[idx_b]));
                            edges.push((coords[idx_b], coords[idx_c]));
                            edges.push((coords[idx_c], coords[idx_a]));
                        }
                    }
                    gpu_data.wireframe_edges = Some(edges);
                    gpu_data.wireframe_edges_mesh_hash = faces_hash;
                    // Invalidate model bind group since edges changed
                    gpu_data.wireframe_model_bind_group = None;
                }
            }

            // Get edges info and convert to GPU format
            let (num_edges, gpu_edges) = {
                let edges = match &gpu_data.wireframe_edges {
                    Some(e) => e,
                    None => return,
                };
                let num = edges.len();
                if num == 0 {
                    return;
                }
                let gpu_e: Vec<GpuEdge2D> = edges
                    .iter()
                    .map(|(a, b)| GpuEdge2D {
                        point_a: a.coords.into(),
                        point_b: b.coords.into(),
                    })
                    .collect();
                (num, gpu_e)
            };

            // Ensure edge buffer capacity
            gpu_data.ensure_edge_buffer_capacity(num_edges);

            // Upload edges to GPU
            ctxt.write_buffer(
                &gpu_data.wireframe_edge_buffer,
                0,
                bytemuck::cast_slice(&gpu_edges),
            );

            // Update wireframe view uniforms
            let wireframe_view_uniforms = WireframeViewUniforms {
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
                &gpu_data.wireframe_view_uniform_buffer,
                0,
                bytemuck::bytes_of(&wireframe_view_uniforms),
            );

            // Update wireframe model uniforms
            let formatted_transform = transform.to_homogeneous();
            let formatted_scale = Matrix2::from_diagonal(&Vector2::new(scale.x, scale.y));

            // Get default color from object or use white
            let default_color = data
                .lines_color()
                .map(|c| [c.x, c.y, c.z, 1.0])
                .unwrap_or([1.0, 1.0, 1.0, 1.0]);

            let wireframe_model_uniforms = WireframeModelUniforms {
                model: Self::mat3_to_padded(&formatted_transform),
                scale: Self::mat2_to_padded(&formatted_scale),
                num_edges: num_edges as u32,
                default_width: data.lines_width(),
                use_perspective: if data.lines_use_perspective() { 1 } else { 0 },
                _padding1: 0.0,
                default_color,
            };
            ctxt.write_buffer(
                &gpu_data.wireframe_model_uniform_buffer,
                0,
                bytemuck::bytes_of(&wireframe_model_uniforms),
            );

            // Get or create cached wireframe bind groups
            if gpu_data.wireframe_view_bind_group.is_none() {
                gpu_data.wireframe_view_bind_group = Some(
                    self.create_wireframe_view_bind_group(&gpu_data.wireframe_view_uniform_buffer),
                );
            }
            if gpu_data.wireframe_model_bind_group.is_none() {
                let edge_size = (num_edges * std::mem::size_of::<GpuEdge2D>()) as u64;
                gpu_data.wireframe_model_bind_group = Some(self.create_wireframe_model_bind_group(
                    &gpu_data.wireframe_model_uniform_buffer,
                    &gpu_data.wireframe_edge_buffer,
                    edge_size,
                ));
            }

            let wireframe_view_bind_group = gpu_data.wireframe_view_bind_group.as_ref().unwrap();
            let wireframe_model_bind_group = gpu_data.wireframe_model_bind_group.as_ref().unwrap();

            // Create wireframe render pass
            {
                let mut render_pass =
                    context
                        .encoder
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("planar_wireframe_render_pass"),
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

                render_pass.set_pipeline(&self.wireframe_pipeline);
                render_pass.set_bind_group(0, wireframe_view_bind_group, &[]);
                render_pass.set_bind_group(1, wireframe_model_bind_group, &[]);

                // Set instance vertex buffers (5 total: positions, colors, deformations, lines_colors, lines_widths)
                render_pass.set_vertex_buffer(0, inst_positions_buf.slice(..));
                render_pass.set_vertex_buffer(1, inst_colors_buf.slice(..));
                render_pass.set_vertex_buffer(2, inst_deformations_buf.slice(..));
                render_pass.set_vertex_buffer(3, inst_lines_colors_buf.slice(..));
                render_pass.set_vertex_buffer(4, inst_lines_widths_buf.slice(..));

                // Draw: 6 vertices per edge, num_instances instances
                let num_vertices = (num_edges * 6) as u32;
                render_pass.draw(0..num_vertices, 0..num_instances as u32);
            }
        }

        // Point rendering
        if data.points_size() > 0.0 {
            // Build vertex list from mesh if needed
            let coords_len = mesh.coords().read().unwrap().len();
            let coords_hash = coords_len as u64;

            if gpu_data.points_vertices.is_none()
                || gpu_data.points_vertices_mesh_hash != coords_hash
            {
                let coords_guard = mesh.coords().read().unwrap();

                if let Some(coords) = coords_guard.data() {
                    gpu_data.points_vertices = Some(coords.clone());
                    gpu_data.points_vertices_mesh_hash = coords_hash;
                    // Invalidate model bind group since vertices changed
                    gpu_data.points_model_bind_group = None;
                }
            }

            // Get vertices info and convert to GPU format
            let (num_verts, gpu_verts) = {
                let verts = match &gpu_data.points_vertices {
                    Some(v) => v,
                    None => return,
                };
                let num = verts.len();
                if num == 0 {
                    return;
                }
                let gpu_v: Vec<GpuVertex2D> = verts
                    .iter()
                    .map(|p| GpuVertex2D {
                        position: p.coords.into(),
                    })
                    .collect();
                (num, gpu_v)
            };

            // Ensure vertex buffer capacity
            gpu_data.ensure_vertex_buffer_capacity(num_verts);

            // Upload vertices to GPU
            ctxt.write_buffer(
                &gpu_data.points_vertex_buffer,
                0,
                bytemuck::cast_slice(&gpu_verts),
            );

            // Update points view uniforms (same format as wireframe)
            let points_view_uniforms = WireframeViewUniforms {
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
                &gpu_data.points_view_uniform_buffer,
                0,
                bytemuck::bytes_of(&points_view_uniforms),
            );

            // Update points model uniforms
            let formatted_transform = transform.to_homogeneous();
            let formatted_scale = Matrix2::from_diagonal(&Vector2::new(scale.x, scale.y));

            // Get default color from object or use white
            let default_color = data
                .points_color()
                .map(|c| [c.x, c.y, c.z, 1.0])
                .unwrap_or([1.0, 1.0, 1.0, 1.0]);

            let points_model_uniforms = PointsModelUniforms {
                model: Self::mat3_to_padded(&formatted_transform),
                scale: Self::mat2_to_padded(&formatted_scale),
                num_vertices: num_verts as u32,
                default_size: data.points_size(),
                use_perspective: if data.points_use_perspective() { 1 } else { 0 },
                _padding1: 0.0,
                default_color,
            };
            ctxt.write_buffer(
                &gpu_data.points_model_uniform_buffer,
                0,
                bytemuck::bytes_of(&points_model_uniforms),
            );

            // Get or create cached points bind groups
            if gpu_data.points_view_bind_group.is_none() {
                gpu_data.points_view_bind_group =
                    Some(self.create_points_view_bind_group(&gpu_data.points_view_uniform_buffer));
            }
            if gpu_data.points_model_bind_group.is_none() {
                let vertex_size = (num_verts * std::mem::size_of::<GpuVertex2D>()) as u64;
                gpu_data.points_model_bind_group = Some(self.create_points_model_bind_group(
                    &gpu_data.points_model_uniform_buffer,
                    &gpu_data.points_vertex_buffer,
                    vertex_size,
                ));
            }

            let points_view_bind_group = gpu_data.points_view_bind_group.as_ref().unwrap();
            let points_model_bind_group = gpu_data.points_model_bind_group.as_ref().unwrap();

            // Create points render pass
            {
                let mut render_pass =
                    context
                        .encoder
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("planar_points_render_pass"),
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

                render_pass.set_pipeline(&self.points_pipeline);
                render_pass.set_bind_group(0, points_view_bind_group, &[]);
                render_pass.set_bind_group(1, points_model_bind_group, &[]);

                // Set instance vertex buffers (5 total: positions, colors, deformations, points_colors, points_sizes)
                render_pass.set_vertex_buffer(0, inst_positions_buf.slice(..));
                render_pass.set_vertex_buffer(1, inst_colors_buf.slice(..));
                render_pass.set_vertex_buffer(2, inst_deformations_buf.slice(..));
                render_pass.set_vertex_buffer(3, inst_points_colors_buf.slice(..));
                render_pass.set_vertex_buffer(4, inst_points_sizes_buf.slice(..));

                // Draw: 6 vertices per point, num_instances instances
                let num_draw_vertices = (num_verts * 6) as u32;
                render_pass.draw(0..num_draw_vertices, 0..num_instances as u32);
            }
        }
    }
}
