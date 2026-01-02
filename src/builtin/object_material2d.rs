use crate::camera::Camera2d;
use crate::context::Context;
use crate::resource::vertex_index::VERTEX_INDEX_FORMAT;
use crate::resource::{
    DynamicUniformBuffer, GpuData, Material2d, GpuMesh2d, RenderContext2d, Texture,
};
use crate::scene::{InstancesBuffer2d, ObjectData2d};
use bytemuck::{Pod, Zeroable};
use glamx::{Mat2, Mat3, Pose2, Vec2};
use std::any::Any;
use std::cell::Cell;

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
    color: [f32; 4],
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

/// Per-object GPU data for ObjectMaterial2d.
pub struct ObjectMaterial2dGpuData {
    // Cached bind groups for surface rendering
    texture_bind_group: Option<wgpu::BindGroup>,
    cached_texture_ptr: usize,
    /// Offset into the dynamic object uniform buffer, set during prepare() phase.
    object_uniform_offset: Option<u32>,
    // Wireframe rendering data
    wireframe_view_uniform_buffer: wgpu::Buffer,
    wireframe_model_uniform_buffer: wgpu::Buffer,
    wireframe_edge_buffer: wgpu::Buffer,
    wireframe_edge_capacity: usize,
    wireframe_view_bind_group: Option<wgpu::BindGroup>,
    wireframe_model_bind_group: Option<wgpu::BindGroup>,
    /// Cached wireframe edges in local coordinates (built lazily from mesh).
    wireframe_edges: Option<Vec<(Vec2, Vec2)>>,
    /// Hash of mesh faces to detect when edges need rebuilding.
    wireframe_edges_mesh_hash: u64,
    /// Cached wireframe view uniforms.
    wireframe_view_uniforms: WireframeViewUniforms,
    /// Cached wireframe model uniforms.
    wireframe_model_uniforms: WireframeModelUniforms,
    /// Number of edges to render.
    wireframe_num_edges: usize,
    /// Whether wireframe uniforms are prepared.
    wireframe_prepared: bool,
    // Point rendering data
    points_view_uniform_buffer: wgpu::Buffer,
    points_model_uniform_buffer: wgpu::Buffer,
    points_vertex_buffer: wgpu::Buffer,
    points_vertex_capacity: usize,
    points_view_bind_group: Option<wgpu::BindGroup>,
    points_model_bind_group: Option<wgpu::BindGroup>,
    /// Cached vertices for point rendering (built lazily from mesh).
    points_vertices: Option<Vec<Vec2>>,
    /// Hash of mesh coords to detect when vertices need rebuilding.
    points_vertices_mesh_hash: u64,
    /// Cached points view uniforms.
    points_view_uniforms: WireframeViewUniforms,
    /// Cached points model uniforms.
    points_model_uniforms: PointsModelUniforms,
    /// Number of vertices to render as points.
    points_num_vertices: usize,
    /// Whether points uniforms are prepared.
    points_prepared: bool,
}

impl ObjectMaterial2dGpuData {
    pub fn new() -> Self {
        let ctxt = Context::get();

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
            texture_bind_group: None,
            cached_texture_ptr: 0,
            object_uniform_offset: None,
            wireframe_view_uniform_buffer,
            wireframe_model_uniform_buffer,
            wireframe_edge_buffer,
            wireframe_edge_capacity,
            wireframe_view_bind_group: None,
            wireframe_model_bind_group: None,
            wireframe_edges: None,
            wireframe_edges_mesh_hash: 0,
            wireframe_view_uniforms: WireframeViewUniforms {
                view: [[0.0; 4]; 3],
                proj: [[0.0; 4]; 3],
                viewport: [0.0; 4],
            },
            wireframe_model_uniforms: WireframeModelUniforms {
                model: [[0.0; 4]; 3],
                scale: [[0.0; 4]; 2],
                num_edges: 0,
                default_width: 0.0,
                use_perspective: 0,
                _padding1: 0.0,
                default_color: [0.0; 4],
            },
            wireframe_num_edges: 0,
            wireframe_prepared: false,
            points_view_uniform_buffer,
            points_model_uniform_buffer,
            points_vertex_buffer,
            points_vertex_capacity,
            points_view_bind_group: None,
            points_model_bind_group: None,
            points_vertices: None,
            points_vertices_mesh_hash: 0,
            points_view_uniforms: WireframeViewUniforms {
                view: [[0.0; 4]; 3],
                proj: [[0.0; 4]; 3],
                viewport: [0.0; 4],
            },
            points_model_uniforms: PointsModelUniforms {
                model: [[0.0; 4]; 3],
                scale: [[0.0; 4]; 2],
                num_vertices: 0,
                default_size: 0.0,
                use_perspective: 0,
                _padding1: 0.0,
                default_color: [0.0; 4],
            },
            points_num_vertices: 0,
            points_prepared: false,
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

impl Default for ObjectMaterial2dGpuData {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuData for ObjectMaterial2dGpuData {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// The default material used to draw 2D objects.
///
/// ## Performance Optimization
///
/// This material uses dynamic uniform buffers to batch uniform data writes:
/// - Frame uniforms (view, projection) are written once per frame
/// - Object uniforms are accumulated in a dynamic buffer and flushed once
/// - This significantly reduces the number of `write_buffer` calls per frame
pub struct ObjectMaterial2d {
    pipeline: wgpu::RenderPipeline,
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

    // === Dynamic uniform buffer system ===
    /// Shared frame uniform buffer (view, projection)
    frame_uniform_buffer: wgpu::Buffer,
    /// Shared bind group for frame uniforms
    frame_bind_group: wgpu::BindGroup,
    /// Dynamic buffer for object uniforms
    object_uniform_buffer: DynamicUniformBuffer<ObjectUniforms>,
    /// Bind group for object uniforms (recreated when buffer grows)
    object_bind_group: Option<wgpu::BindGroup>,
    /// Frame counter for detecting new frames
    frame_counter: Cell<u64>,
    /// Last frame we processed (to detect new frame)
    last_frame: Cell<u64>,
}

impl Default for ObjectMaterial2d {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectMaterial2d {
    /// Creates a new `ObjectMaterial2d`.
    pub fn new() -> ObjectMaterial2d {
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

        // Object bind group uses dynamic offset for batched uniforms
        let object_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("planar_material_object_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true, // Enable dynamic offsets!
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
        let shader = ctxt.create_shader_module(
            Some("planar_material_shader"),
            include_str!("object2d.wgsl"),
        );

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
            include_str!("wireframe_polyline2d.wgsl"),
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
            include_str!("wireframe_points2d.wgsl"),
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

        // === Create shared dynamic buffer resources ===

        // Frame uniform buffer (written once per frame)
        let frame_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planar_shared_frame_uniform_buffer"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create frame bind group
        let frame_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planar_shared_frame_bind_group"),
            layout: &frame_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_uniform_buffer.as_entire_binding(),
            }],
        });

        // Dynamic buffer for object uniforms
        let object_uniform_buffer =
            DynamicUniformBuffer::<ObjectUniforms>::new("planar_dynamic_object_uniform_buffer");

        // Create initial object bind group
        let object_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planar_dynamic_object_bind_group"),
            layout: &object_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: object_uniform_buffer.buffer(),
                    offset: 0,
                    size: std::num::NonZeroU64::new(object_uniform_buffer.aligned_size()),
                }),
            }],
        });

        ObjectMaterial2d {
            pipeline,
            object_bind_group_layout,
            texture_bind_group_layout,
            wireframe_pipeline,
            wireframe_view_bind_group_layout,
            wireframe_model_bind_group_layout,
            points_pipeline,
            points_view_bind_group_layout,
            points_model_bind_group_layout,
            frame_uniform_buffer,
            frame_bind_group,
            object_uniform_buffer,
            object_bind_group: Some(object_bind_group),
            frame_counter: Cell::new(0),
            last_frame: Cell::new(u64::MAX),
        }
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
    fn mat3_to_padded(m: &Mat3) -> [[f32; 4]; 3] {
        let cols = m.to_cols_array_2d();
        [
            [cols[0][0], cols[0][1], cols[0][2], 0.0],
            [cols[1][0], cols[1][1], cols[1][2], 0.0],
            [cols[2][0], cols[2][1], cols[2][2], 0.0],
        ]
    }

    /// Helper to convert mat2x2 to padded array for uniforms.
    fn mat2_to_padded(m: &Mat2) -> [[f32; 4]; 2] {
        let cols = m.to_cols_array_2d();
        [
            [cols[0][0], cols[0][1], 0.0, 0.0],
            [cols[1][0], cols[1][1], 0.0, 0.0],
        ]
    }
}

impl Material2d for ObjectMaterial2d {
    fn create_gpu_data(&self) -> Box<dyn GpuData> {
        Box::new(ObjectMaterial2dGpuData::new())
    }

    fn begin_frame(&mut self) {
        self.frame_counter
            .set(self.frame_counter.get().wrapping_add(1));
        self.object_uniform_buffer.clear();
    }

    fn flush(&mut self) {
        let ctxt = Context::get();

        // Flush returns true if buffer was reallocated
        let buffer_reallocated = self.object_uniform_buffer.flush();

        // Recreate bind group if buffer was reallocated
        if buffer_reallocated {
            self.object_bind_group = Some(ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("planar_dynamic_object_bind_group"),
                layout: &self.object_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: self.object_uniform_buffer.buffer(),
                        offset: 0,
                        size: std::num::NonZeroU64::new(self.object_uniform_buffer.aligned_size()),
                    }),
                }],
            }));
        }
    }

    fn prepare(
        &mut self,
        transform: Pose2,
        scale: Vec2,
        camera: &mut dyn Camera2d,
        data: &ObjectData2d,
        mesh: &mut GpuMesh2d,
        _instances: &mut InstancesBuffer2d,
        gpu_data: &mut dyn GpuData,
        context: &RenderContext2d,
    ) {
        let ctxt = Context::get();

        // Downcast gpu_data to our specific type
        let gpu_data = gpu_data
            .as_any_mut()
            .downcast_mut::<ObjectMaterial2dGpuData>()
            .expect("ObjectMaterial2d requires ObjectMaterial2dGpuData");

        // Check if this is a new frame (first object being prepared)
        let current_frame = self.frame_counter.get();
        let is_new_frame = current_frame != self.last_frame.get();

        if is_new_frame {
            self.last_frame.set(current_frame);

            // Get camera matrices
            let (view, proj) = camera.view_transform_pair();

            // Write frame uniforms once per frame
            let frame_uniforms = FrameUniforms {
                view: Self::mat3_to_padded(&view),
                proj: Self::mat3_to_padded(&proj),
            };

            ctxt.write_buffer(
                &self.frame_uniform_buffer,
                0,
                bytemuck::bytes_of(&frame_uniforms),
            );
        }

        // Surface rendering uniforms
        if data.surface_rendering_active() {
            // Compute object uniforms
            let formatted_transform = transform.to_mat3();
            let formatted_scale = Mat2::from_diagonal(scale);

            let color = data.color();
            let object_uniforms = ObjectUniforms {
                model: Self::mat3_to_padded(&formatted_transform),
                scale: Self::mat2_to_padded(&formatted_scale),
                color: [color.r, color.g, color.b, color.a],
            };

            // Push to dynamic buffer and store offset in gpu_data
            let object_offset = self.object_uniform_buffer.push(&object_uniforms);
            gpu_data.object_uniform_offset = Some(object_offset);

            // Cache texture bind group, invalidate if texture changed
            let texture_ptr = std::sync::Arc::as_ptr(data.texture()) as usize;
            if gpu_data.texture_bind_group.is_none() || gpu_data.cached_texture_ptr != texture_ptr {
                gpu_data.texture_bind_group = Some(self.create_texture_bind_group(data.texture()));
                gpu_data.cached_texture_ptr = texture_ptr;
            }
        }

        // Wireframe rendering uniforms
        gpu_data.wireframe_prepared = false;
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
                        let idx_a = face[0] as usize;
                        let idx_b = face[1] as usize;
                        let idx_c = face[2] as usize;
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
            if let Some(edges) = &gpu_data.wireframe_edges {
                let num_edges = edges.len();
                if num_edges > 0 {
                    let gpu_edges: Vec<GpuEdge2D> = edges
                        .iter()
                        .map(|(a, b)| GpuEdge2D {
                            point_a: (*a).into(),
                            point_b: (*b).into(),
                        })
                        .collect();

                    // Ensure edge buffer capacity
                    gpu_data.ensure_edge_buffer_capacity(num_edges);

                    // Upload edges to GPU
                    ctxt.write_buffer(
                        &gpu_data.wireframe_edge_buffer,
                        0,
                        bytemuck::cast_slice(&gpu_edges),
                    );

                    // Compute wireframe view uniforms
                    let (view, proj) = camera.view_transform_pair();
                    gpu_data.wireframe_view_uniforms = WireframeViewUniforms {
                        view: Self::mat3_to_padded(&view),
                        proj: Self::mat3_to_padded(&proj),
                        viewport: [
                            0.0,
                            0.0,
                            context.viewport_width as f32,
                            context.viewport_height as f32,
                        ],
                    };

                    // Compute wireframe model uniforms
                    let formatted_transform = transform.to_mat3();
                    let formatted_scale = Mat2::from_diagonal(scale);

                    let default_color = data
                        .lines_color()
                        .map(|c| [c.r, c.g, c.b, c.a])
                        .unwrap_or([1.0, 1.0, 1.0, 1.0]);

                    gpu_data.wireframe_model_uniforms = WireframeModelUniforms {
                        model: Self::mat3_to_padded(&formatted_transform),
                        scale: Self::mat2_to_padded(&formatted_scale),
                        num_edges: num_edges as u32,
                        default_width: data.lines_width(),
                        use_perspective: if data.lines_use_perspective() { 1 } else { 0 },
                        _padding1: 0.0,
                        default_color,
                    };

                    // Write uniforms to GPU
                    ctxt.write_buffer(
                        &gpu_data.wireframe_view_uniform_buffer,
                        0,
                        bytemuck::bytes_of(&gpu_data.wireframe_view_uniforms),
                    );
                    ctxt.write_buffer(
                        &gpu_data.wireframe_model_uniform_buffer,
                        0,
                        bytemuck::bytes_of(&gpu_data.wireframe_model_uniforms),
                    );

                    // Get or create cached wireframe bind groups
                    if gpu_data.wireframe_view_bind_group.is_none() {
                        gpu_data.wireframe_view_bind_group =
                            Some(self.create_wireframe_view_bind_group(
                                &gpu_data.wireframe_view_uniform_buffer,
                            ));
                    }
                    if gpu_data.wireframe_model_bind_group.is_none() {
                        let edge_size = (num_edges * std::mem::size_of::<GpuEdge2D>()) as u64;
                        gpu_data.wireframe_model_bind_group =
                            Some(self.create_wireframe_model_bind_group(
                                &gpu_data.wireframe_model_uniform_buffer,
                                &gpu_data.wireframe_edge_buffer,
                                edge_size,
                            ));
                    }

                    gpu_data.wireframe_num_edges = num_edges;
                    gpu_data.wireframe_prepared = true;
                }
            }
        }

        // Point rendering uniforms
        gpu_data.points_prepared = false;
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
            if let Some(verts) = &gpu_data.points_vertices {
                let num_verts = verts.len();
                if num_verts > 0 {
                    let gpu_verts: Vec<GpuVertex2D> = verts
                        .iter()
                        .map(|p| GpuVertex2D {
                            position: (*p).into(),
                        })
                        .collect();

                    // Ensure vertex buffer capacity
                    gpu_data.ensure_vertex_buffer_capacity(num_verts);

                    // Upload vertices to GPU
                    ctxt.write_buffer(
                        &gpu_data.points_vertex_buffer,
                        0,
                        bytemuck::cast_slice(&gpu_verts),
                    );

                    // Compute points view uniforms
                    let (view, proj) = camera.view_transform_pair();
                    gpu_data.points_view_uniforms = WireframeViewUniforms {
                        view: Self::mat3_to_padded(&view),
                        proj: Self::mat3_to_padded(&proj),
                        viewport: [
                            0.0,
                            0.0,
                            context.viewport_width as f32,
                            context.viewport_height as f32,
                        ],
                    };

                    // Compute points model uniforms
                    let formatted_transform = transform.to_mat3();
                    let formatted_scale = Mat2::from_diagonal(scale);

                    let default_color = data
                        .points_color()
                        .map(|c| [c.r, c.g, c.b, c.a])
                        .unwrap_or([1.0, 1.0, 1.0, 1.0]);

                    gpu_data.points_model_uniforms = PointsModelUniforms {
                        model: Self::mat3_to_padded(&formatted_transform),
                        scale: Self::mat2_to_padded(&formatted_scale),
                        num_vertices: num_verts as u32,
                        default_size: data.points_size(),
                        use_perspective: if data.points_use_perspective() { 1 } else { 0 },
                        _padding1: 0.0,
                        default_color,
                    };

                    // Write uniforms to GPU
                    ctxt.write_buffer(
                        &gpu_data.points_view_uniform_buffer,
                        0,
                        bytemuck::bytes_of(&gpu_data.points_view_uniforms),
                    );
                    ctxt.write_buffer(
                        &gpu_data.points_model_uniform_buffer,
                        0,
                        bytemuck::bytes_of(&gpu_data.points_model_uniforms),
                    );

                    // Get or create cached points bind groups
                    if gpu_data.points_view_bind_group.is_none() {
                        gpu_data.points_view_bind_group =
                            Some(self.create_points_view_bind_group(
                                &gpu_data.points_view_uniform_buffer,
                            ));
                    }
                    if gpu_data.points_model_bind_group.is_none() {
                        let vertex_size = (num_verts * std::mem::size_of::<GpuVertex2D>()) as u64;
                        gpu_data.points_model_bind_group =
                            Some(self.create_points_model_bind_group(
                                &gpu_data.points_model_uniform_buffer,
                                &gpu_data.points_vertex_buffer,
                                vertex_size,
                            ));
                    }

                    gpu_data.points_num_vertices = num_verts;
                    gpu_data.points_prepared = true;
                }
            }
        }
    }

    fn render(
        &mut self,
        _transform: Pose2,
        _scale: Vec2,
        _camera: &mut dyn Camera2d,
        data: &ObjectData2d,
        mesh: &mut GpuMesh2d,
        instances: &mut InstancesBuffer2d,
        gpu_data: &mut dyn GpuData,
        render_pass: &mut wgpu::RenderPass<'_>,
        _context: &RenderContext2d,
    ) {
        // Downcast gpu_data to our specific type
        let gpu_data = gpu_data
            .as_any_mut()
            .downcast_mut::<ObjectMaterial2dGpuData>()
            .expect("ObjectMaterial2d requires ObjectMaterial2dGpuData");

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

            // Get the pre-computed object uniform offset from prepare() phase
            let object_offset = gpu_data
                .object_uniform_offset
                .expect("prepare() must be called before render()");

            let texture_bind_group = match gpu_data.texture_bind_group.as_ref() {
                Some(bg) => bg,
                None => return,
            };
            let object_bind_group = self.object_bind_group.as_ref().unwrap();

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.frame_bind_group, &[]);
            // Use dynamic offset for object uniforms!
            render_pass.set_bind_group(1, object_bind_group, &[object_offset]);
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

        // Wireframe rendering
        if gpu_data.wireframe_prepared {
            let wireframe_view_bind_group = match gpu_data.wireframe_view_bind_group.as_ref() {
                Some(bg) => bg,
                None => return,
            };
            let wireframe_model_bind_group = match gpu_data.wireframe_model_bind_group.as_ref() {
                Some(bg) => bg,
                None => return,
            };

            render_pass.set_pipeline(&self.wireframe_pipeline);
            render_pass.set_bind_group(0, wireframe_view_bind_group, &[]);
            render_pass.set_bind_group(1, wireframe_model_bind_group, &[]);

            // Set instance vertex buffers
            render_pass.set_vertex_buffer(0, inst_positions_buf.slice(..));
            render_pass.set_vertex_buffer(1, inst_colors_buf.slice(..));
            render_pass.set_vertex_buffer(2, inst_deformations_buf.slice(..));
            render_pass.set_vertex_buffer(3, inst_lines_colors_buf.slice(..));
            render_pass.set_vertex_buffer(4, inst_lines_widths_buf.slice(..));

            // Draw: 6 vertices per edge, num_instances instances
            let num_vertices = (gpu_data.wireframe_num_edges * 6) as u32;
            render_pass.draw(0..num_vertices, 0..num_instances as u32);
        }

        // Point rendering
        if gpu_data.points_prepared {
            let points_view_bind_group = match gpu_data.points_view_bind_group.as_ref() {
                Some(bg) => bg,
                None => return,
            };
            let points_model_bind_group = match gpu_data.points_model_bind_group.as_ref() {
                Some(bg) => bg,
                None => return,
            };

            render_pass.set_pipeline(&self.points_pipeline);
            render_pass.set_bind_group(0, points_view_bind_group, &[]);
            render_pass.set_bind_group(1, points_model_bind_group, &[]);

            // Set instance vertex buffers
            render_pass.set_vertex_buffer(0, inst_positions_buf.slice(..));
            render_pass.set_vertex_buffer(1, inst_colors_buf.slice(..));
            render_pass.set_vertex_buffer(2, inst_deformations_buf.slice(..));
            render_pass.set_vertex_buffer(3, inst_points_colors_buf.slice(..));
            render_pass.set_vertex_buffer(4, inst_points_sizes_buf.slice(..));

            // Draw: 6 vertices per point, num_instances instances
            let num_draw_vertices = (gpu_data.points_num_vertices * 6) as u32;
            render_pass.draw(0..num_draw_vertices, 0..num_instances as u32);
        }
    }
}
