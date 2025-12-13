use crate::camera::Camera;
use crate::context::Context;
use crate::light::Light;
use crate::resource::vertex_index::VERTEX_INDEX_FORMAT;
use crate::resource::{GpuData, GpuMesh, Material, RenderContext, Texture};
use crate::scene::{InstancesBuffer, ObjectData};
use bytemuck::{Pod, Zeroable};
use na::{Isometry3, Matrix3, Point3, Vector3};
use std::any::Any;

/// Frame-level uniforms (view, projection, light).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FrameUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    light_position: [f32; 3],
    _padding: f32,
}

/// Object-level uniforms (transform, scale, color).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ObjectUniforms {
    transform: [[f32; 4]; 4],
    ntransform: [[f32; 4]; 3], // mat3x3 padded to mat3x4 for alignment
    scale: [[f32; 4]; 3],      // mat3x3 padded to mat3x4 for alignment
    color: [f32; 3],
    _padding: f32,
}

/// View uniforms for wireframe rendering (includes viewport).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WireframeViewUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    viewport: [f32; 4], // x, y, width, height
}

/// Model uniforms for wireframe rendering.
/// Layout must match WGSL struct:
/// - transform: mat4x4<f32> at offset 0 (64 bytes)
/// - scale: vec3<f32> at offset 64 (12 bytes, aligned to 16)
/// - num_edges: u32 at offset 76 (4 bytes)
/// - default_color: vec4<f32> at offset 80 (16 bytes, aligned to 16)
/// - default_width: f32 at offset 96 (4 bytes)
/// - use_perspective: u32 at offset 100 (4 bytes)
/// - _padding: vec2<f32> at offset 104 (8 bytes)
///
/// Total: 112 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WireframeModelUniforms {
    transform: [[f32; 4]; 4], // 64 bytes at offset 0
    scale: [f32; 3],          // 12 bytes at offset 64
    num_edges: u32,           // 4 bytes at offset 76
    default_color: [f32; 4],  // 16 bytes at offset 80
    default_width: f32,       // 4 bytes at offset 96
    use_perspective: u32,     // 4 bytes at offset 100
    _padding: [f32; 2],       // 8 bytes at offset 104 to align to 16-byte boundary
}

/// Edge data in GPU format (matches shader struct).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuEdge {
    point_a: [f32; 3],
    _pad_a: f32,
    point_b: [f32; 3],
    _pad_b: f32,
}

/// Model uniforms for point rendering.
/// Layout must match wireframe_points.wgsl ModelUniforms struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PointsModelUniforms {
    transform: [[f32; 4]; 4], // 64 bytes at offset 0
    scale: [f32; 3],          // 12 bytes at offset 64
    num_vertices: u32,        // 4 bytes at offset 76
    default_color: [f32; 4],  // 16 bytes at offset 80
    default_size: f32,        // 4 bytes at offset 96
    use_perspective: u32,     // 4 bytes at offset 100
    _padding: [f32; 2],       // 8 bytes at offset 104 to align to 16-byte boundary
}

/// Vertex data in GPU format for points (matches shader struct).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuVertex {
    position: [f32; 3],
    _pad: f32,
}

/// Per-object GPU data for ObjectMaterial.
///
/// Each object in the scene has its own instance of this struct,
/// containing uniform buffers specific to that object.
pub struct ObjectMaterialGpuData {
    frame_uniform_buffer: wgpu::Buffer,
    object_uniform_buffer: wgpu::Buffer,
    // Cached bind groups (created lazily)
    frame_bind_group: Option<wgpu::BindGroup>,
    object_bind_group: Option<wgpu::BindGroup>,
    // Cached texture bind group with pointer to detect texture changes
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
    wireframe_edges: Option<Vec<(Point3<f32>, Point3<f32>)>>,
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
    points_vertices: Option<Vec<Point3<f32>>>,
    /// Hash of mesh coords to detect when vertices need rebuilding.
    points_vertices_mesh_hash: u64,
}

impl ObjectMaterialGpuData {
    /// Creates new per-object GPU data.
    pub fn new() -> Self {
        let ctxt = Context::get();

        let frame_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object_material_frame_uniform_buffer"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let object_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object_material_object_uniform_buffer"),
            size: std::mem::size_of::<ObjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Wireframe uniform buffers
        let wireframe_view_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wireframe_view_uniform_buffer"),
            size: std::mem::size_of::<WireframeViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let wireframe_model_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wireframe_model_uniform_buffer"),
            size: std::mem::size_of::<WireframeModelUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initial edge storage buffer (will grow as needed)
        let wireframe_edge_capacity = 1024;
        let wireframe_edge_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wireframe_edge_buffer"),
            size: (std::mem::size_of::<GpuEdge>() * wireframe_edge_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Point rendering uniform buffers (reuse same view uniform layout as wireframe)
        let points_view_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("points_view_uniform_buffer"),
            size: std::mem::size_of::<WireframeViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let points_model_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("points_model_uniform_buffer"),
            size: std::mem::size_of::<PointsModelUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initial vertex storage buffer for points (will grow as needed)
        let points_vertex_capacity = 1024;
        let points_vertex_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("points_vertex_buffer"),
            size: (std::mem::size_of::<GpuVertex>() * points_vertex_capacity) as u64,
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

    /// Ensures the edge buffer has enough capacity, growing if needed.
    fn ensure_edge_buffer_capacity(&mut self, needed: usize) {
        if needed > self.wireframe_edge_capacity {
            let ctxt = Context::get();
            let new_capacity = needed.next_power_of_two();
            self.wireframe_edge_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wireframe_edge_buffer"),
                size: (std::mem::size_of::<GpuEdge>() * new_capacity) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.wireframe_edge_capacity = new_capacity;
            // Invalidate bind group since buffer changed
            self.wireframe_model_bind_group = None;
        }
    }

    /// Ensures the vertex buffer for points has enough capacity, growing if needed.
    fn ensure_vertex_buffer_capacity(&mut self, needed: usize) {
        if needed > self.points_vertex_capacity {
            let ctxt = Context::get();
            let new_capacity = needed.next_power_of_two();
            self.points_vertex_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
                label: Some("points_vertex_buffer"),
                size: (std::mem::size_of::<GpuVertex>() * new_capacity) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.points_vertex_capacity = new_capacity;
            // Invalidate bind group since buffer changed
            self.points_model_bind_group = None;
        }
    }
}

impl Default for ObjectMaterialGpuData {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuData for ObjectMaterialGpuData {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// The default material used to draw objects.
///
/// This struct holds shared resources (pipeline, bind group layouts) that
/// are used by all objects. Per-object resources are stored in
/// `ObjectMaterialGpuData` instances.
pub struct ObjectMaterial {
    /// Pipeline with backface culling enabled
    pipeline_cull: wgpu::RenderPipeline,
    /// Pipeline with backface culling disabled
    pipeline_no_cull: wgpu::RenderPipeline,
    frame_bind_group_layout: wgpu::BindGroupLayout,
    object_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    // Wireframe rendering resources
    wireframe_pipeline: wgpu::RenderPipeline,
    wireframe_view_bind_group_layout: wgpu::BindGroupLayout,
    wireframe_model_bind_group_layout: wgpu::BindGroupLayout,
    // Point rendering resources
    points_pipeline: wgpu::RenderPipeline,
    points_view_bind_group_layout: wgpu::BindGroupLayout,
    points_model_bind_group_layout: wgpu::BindGroupLayout,
}

impl Default for ObjectMaterial {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectMaterial {
    /// Creates a new `ObjectMaterial`.
    pub fn new() -> ObjectMaterial {
        let ctxt = Context::get();

        // Create bind group layouts
        let frame_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("object_material_frame_bind_group_layout"),
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

        let object_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("object_material_object_bind_group_layout"),
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
                label: Some("object_material_texture_bind_group_layout"),
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
            label: Some("object_material_pipeline_layout"),
            bind_group_layouts: &[
                &frame_bind_group_layout,
                &object_bind_group_layout,
                &texture_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Load shader
        let shader =
            ctxt.create_shader_module(Some("object_material_shader"), include_str!("default.wgsl"));

        // Vertex buffer layouts
        // Note: We use separate buffers for instance data (positions, colors, deformations)
        // instead of interleaving them, to avoid per-frame data conversion overhead.
        let vertex_buffer_layouts = [
            // Buffer 0: Vertex positions
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            // Buffer 1: Texture coordinates
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                }],
            },
            // Buffer 2: Normals
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            // Buffer 3: Instance positions (Point3<f32>)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3, // inst_tra
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            // Buffer 4: Instance colors ([f32; 4])
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 4, // inst_color
                    format: wgpu::VertexFormat::Float32x4,
                }],
            },
            // Buffer 5: Instance deformations (3x Vector3<f32> = 3 columns of 3x3 matrix)
            // Stored as 3 consecutive vec3s per instance
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 9]>() as wgpu::BufferAddress, // 3 vec3s
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    // inst_def_0 (column 0)
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 5,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    // inst_def_1 (column 1)
                    wgpu::VertexAttribute {
                        offset: 12, // 3 * sizeof(f32)
                        shader_location: 6,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    // inst_def_2 (column 2)
                    wgpu::VertexAttribute {
                        offset: 24, // 6 * sizeof(f32)
                        shader_location: 7,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            },
        ];

        // Helper to create a pipeline with specific cull mode
        let create_pipeline = |cull_mode: Option<wgpu::Face>, label: &str| {
            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
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
                    cull_mode,
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
            })
        };

        let pipeline_cull =
            create_pipeline(Some(wgpu::Face::Back), "object_material_pipeline_cull");
        let pipeline_no_cull = create_pipeline(None, "object_material_pipeline_no_cull");

        // Create wireframe shader and pipelines for lines/points
        // Note: _wireframe_shader, _wireframe_pipeline_layout, and _wireframe_vertex_buffer_layouts
        // were previously used for the old PointList pipeline but are now replaced by the new
        // wireframe_points.wgsl shader. Keeping them here in case they're needed for 1px fallback.
        let _wireframe_shader =
            ctxt.create_shader_module(Some("wireframe_shader"), include_str!("wireframe.wgsl"));

        // Pipeline layout for wireframe (only needs frame and object uniforms, no texture)
        let _wireframe_pipeline_layout =
            ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("wireframe_pipeline_layout"),
                bind_group_layouts: &[&frame_bind_group_layout, &object_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Vertex buffer layouts for wireframe (position only + instance data)
        let _wireframe_vertex_buffer_layouts = [
            // Buffer 0: Vertex positions
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            // Buffer 1: Instance positions (Point3<f32>)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3, // inst_tra
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            // Buffer 2: Instance colors ([f32; 4])
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 4, // inst_color
                    format: wgpu::VertexFormat::Float32x4,
                }],
            },
            // Buffer 3: Instance deformations (3x Vector3<f32> = 3 columns of 3x3 matrix)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 9]>() as wgpu::BufferAddress, // 3 vec3s
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    // inst_def_0 (column 0)
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 5,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    // inst_def_1 (column 1)
                    wgpu::VertexAttribute {
                        offset: 12, // 3 * sizeof(f32)
                        shader_location: 6,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    // inst_def_2 (column 2)
                    wgpu::VertexAttribute {
                        offset: 24, // 6 * sizeof(f32)
                        shader_location: 7,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            },
        ];

        // Create wireframe bind group layouts
        let wireframe_view_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("wireframe_view_bind_group_layout"),
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
                label: Some("wireframe_model_bind_group_layout"),
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

        let wireframe_polyline_pipeline_layout =
            ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("wireframe_polyline_pipeline_layout"),
                bind_group_layouts: &[
                    &wireframe_view_bind_group_layout,
                    &wireframe_model_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        // Load wireframe polyline shader
        let wireframe_polyline_shader = ctxt.create_shader_module(
            Some("wireframe_polyline_shader"),
            include_str!("wireframe_polyline.wgsl"),
        );

        // Instance vertex buffer layouts for wireframe (matching InstancesBuffer)
        let wireframe_instance_buffer_layouts = [
            // Buffer 0: positions (Point3<f32>)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            // Buffer 1: colors ([f32; 4]) - not used but needed for layout consistency
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                }],
            },
            // Buffer 2: deformations - all 3 columns from same buffer with stride = 3*vec3
            // Matrix3 is stored as 3 consecutive Vector3 columns (36 bytes total)
            wgpu::VertexBufferLayout {
                array_stride: (std::mem::size_of::<[f32; 3]>() * 3) as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    // Column 0 at offset 0
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    // Column 1 at offset 12
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<[f32; 3]>() as u64,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    // Column 2 at offset 24
                    wgpu::VertexAttribute {
                        offset: (std::mem::size_of::<[f32; 3]>() * 2) as u64,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            },
            // Buffer 3: lines_colors ([f32; 4])
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                }],
            },
            // Buffer 4: lines_widths (f32)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32,
                }],
            },
        ];

        let wireframe_pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("wireframe_polyline_pipeline"),
            layout: Some(&wireframe_polyline_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &wireframe_polyline_shader,
                entry_point: Some("vs_main"),
                buffers: &wireframe_instance_buffer_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &wireframe_polyline_shader,
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

        // Create points bind group layouts (same view layout as wireframe, different model layout)
        let points_view_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("points_view_bind_group_layout"),
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
                label: Some("points_model_bind_group_layout"),
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
            label: Some("points_pipeline_layout"),
            bind_group_layouts: &[
                &points_view_bind_group_layout,
                &points_model_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Load points shader
        let points_shader = ctxt.create_shader_module(
            Some("wireframe_points_shader"),
            include_str!("wireframe_points.wgsl"),
        );

        // Instance vertex buffer layouts for points (similar to wireframe but with points_colors/sizes)
        let points_instance_buffer_layouts = [
            // Buffer 0: positions (Point3<f32>)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            // Buffer 1: colors ([f32; 4]) - not used but needed for layout consistency
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                }],
            },
            // Buffer 2: deformations - all 3 columns from same buffer with stride = 3*vec3
            wgpu::VertexBufferLayout {
                array_stride: (std::mem::size_of::<[f32; 3]>() * 3) as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<[f32; 3]>() as u64,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: (std::mem::size_of::<[f32; 3]>() * 2) as u64,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            },
            // Buffer 3: points_colors ([f32; 4])
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                }],
            },
            // Buffer 4: points_sizes (f32)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32,
                }],
            },
        ];

        let points_pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("wireframe_points_pipeline"),
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

        ObjectMaterial {
            pipeline_cull,
            pipeline_no_cull,
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
            label: Some("object_material_frame_bind_group"),
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
            label: Some("object_material_object_bind_group"),
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
            label: Some("object_material_texture_bind_group"),
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
            label: Some("wireframe_view_bind_group"),
            layout: &self.wireframe_view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }

    fn create_wireframe_model_bind_group(
        &self,
        model_buffer: &wgpu::Buffer,
        edge_buffer: &wgpu::Buffer,
        edge_size: u64,
    ) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wireframe_model_bind_group"),
            layout: &self.wireframe_model_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: edge_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(edge_size),
                    }),
                },
            ],
        })
    }

    fn create_points_view_bind_group(&self, buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("points_view_bind_group"),
            layout: &self.points_view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }

    fn create_points_model_bind_group(
        &self,
        model_buffer: &wgpu::Buffer,
        vertex_buffer: &wgpu::Buffer,
        vertex_size: u64,
    ) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("points_model_bind_group"),
            layout: &self.points_model_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: vertex_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(vertex_size),
                    }),
                },
            ],
        })
    }
}

impl Material for ObjectMaterial {
    fn create_gpu_data(&self) -> Box<dyn GpuData> {
        Box::new(ObjectMaterialGpuData::new())
    }

    fn render(
        &mut self,
        pass: usize,
        transform: &Isometry3<f32>,
        scale: &Vector3<f32>,
        camera: &mut dyn Camera,
        light: &Light,
        data: &ObjectData,
        mesh: &mut GpuMesh,
        instances: &mut InstancesBuffer,
        gpu_data: &mut dyn GpuData,
        context: &mut RenderContext,
    ) {
        let ctxt = Context::get();

        let render_surface = data.surface_rendering_active();
        let render_wireframe = data.lines_width() > 0.0;
        let render_points = data.points_size() > 0.0;

        // Nothing to render
        if !render_surface && !render_wireframe && !render_points {
            return;
        }

        // Downcast gpu_data to our specific type
        let gpu_data = gpu_data
            .as_any_mut()
            .downcast_mut::<ObjectMaterialGpuData>()
            .expect("ObjectMaterial requires ObjectMaterialGpuData");

        // Create frame uniforms and write to per-object buffer
        let (view, proj) = camera.view_transform_pair(pass);
        let light_pos = match light {
            Light::Absolute(p) => *p,
            Light::StickToCamera => camera.eye(),
        };

        let frame_uniforms = FrameUniforms {
            view: view.to_homogeneous().into(),
            proj: proj.into(),
            light_position: light_pos.coords.into(),
            _padding: 0.0,
        };

        ctxt.write_buffer(
            &gpu_data.frame_uniform_buffer,
            0,
            bytemuck::bytes_of(&frame_uniforms),
        );

        // Create object uniforms and write to per-object buffer
        let formatted_transform = transform.to_homogeneous();
        let ntransform = transform.rotation.to_rotation_matrix().into_inner();
        let formatted_scale = Matrix3::from_diagonal(&Vector3::new(scale.x, scale.y, scale.z));

        // Pad mat3x3 to mat3x4 for proper alignment
        let ntransform_padded: [[f32; 4]; 3] = [
            [
                ntransform[(0, 0)],
                ntransform[(1, 0)],
                ntransform[(2, 0)],
                0.0,
            ],
            [
                ntransform[(0, 1)],
                ntransform[(1, 1)],
                ntransform[(2, 1)],
                0.0,
            ],
            [
                ntransform[(0, 2)],
                ntransform[(1, 2)],
                ntransform[(2, 2)],
                0.0,
            ],
        ];
        let scale_padded: [[f32; 4]; 3] = [
            [
                formatted_scale[(0, 0)],
                formatted_scale[(1, 0)],
                formatted_scale[(2, 0)],
                0.0,
            ],
            [
                formatted_scale[(0, 1)],
                formatted_scale[(1, 1)],
                formatted_scale[(2, 1)],
                0.0,
            ],
            [
                formatted_scale[(0, 2)],
                formatted_scale[(1, 2)],
                formatted_scale[(2, 2)],
                0.0,
            ],
        ];

        let object_uniforms = ObjectUniforms {
            transform: formatted_transform.into(),
            ntransform: ntransform_padded,
            scale: scale_padded,
            color: (*data.color()).into(),
            _padding: 0.0,
        };

        ctxt.write_buffer(
            &gpu_data.object_uniform_buffer,
            0,
            bytemuck::bytes_of(&object_uniforms),
        );

        // Load instance data directly to GPU without conversion
        let num_instances = instances.len();
        instances.positions.load_to_gpu();
        instances.colors.load_to_gpu();
        instances.deformations.load_to_gpu();

        // Ensure mesh buffers are on GPU
        mesh.coords().write().unwrap().load_to_gpu();
        mesh.uvs().write().unwrap().load_to_gpu();
        mesh.normals().write().unwrap().load_to_gpu();
        mesh.faces().write().unwrap().load_to_gpu();

        let coords_buffer = mesh.coords().read().unwrap();
        let uvs_buffer = mesh.uvs().read().unwrap();
        let normals_buffer = mesh.normals().read().unwrap();
        let faces_buffer = mesh.faces().read().unwrap();

        let coords_buf = match coords_buffer.buffer() {
            Some(b) => b,
            None => return,
        };
        let uvs_buf = match uvs_buffer.buffer() {
            Some(b) => b,
            None => return,
        };
        let normals_buf = match normals_buffer.buffer() {
            Some(b) => b,
            None => return,
        };
        let faces_buf = match faces_buffer.buffer() {
            Some(b) => b,
            None => return,
        };

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

        // Render surface (filled triangles)
        if render_surface {
            let frame_bind_group = gpu_data.frame_bind_group.as_ref().unwrap();
            let object_bind_group = gpu_data.object_bind_group.as_ref().unwrap();
            let texture_bind_group = gpu_data.texture_bind_group.as_ref().unwrap();
            let mut render_pass = context
                .encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("object_material_render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: context.color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: context.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

            // Select pipeline based on backface culling setting
            let pipeline = if data.backface_culling_enabled() {
                &self.pipeline_cull
            } else {
                &self.pipeline_no_cull
            };
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, frame_bind_group, &[]);
            render_pass.set_bind_group(1, object_bind_group, &[]);
            render_pass.set_bind_group(2, texture_bind_group, &[]);

            // Set vertex buffers for mesh data
            render_pass.set_vertex_buffer(0, coords_buf.slice(..));
            render_pass.set_vertex_buffer(1, uvs_buf.slice(..));
            render_pass.set_vertex_buffer(2, normals_buf.slice(..));

            // Set instance buffers directly (no per-frame conversion needed)
            render_pass.set_vertex_buffer(3, inst_positions_buf.slice(..));
            render_pass.set_vertex_buffer(4, inst_colors_buf.slice(..));
            render_pass.set_vertex_buffer(5, inst_deformations_buf.slice(..));

            render_pass.set_index_buffer(faces_buf.slice(..), VERTEX_INDEX_FORMAT);

            render_pass.draw_indexed(0..mesh.num_indices(), 0, 0..num_instances as u32);
        }

        // Render wireframe (thick lines using polyline technique)
        if render_wireframe {
            // Build wireframe edges from mesh if needed
            // Use a simple hash of the faces buffer length as a cache key
            let faces_len = mesh.faces().read().unwrap().len();
            let faces_hash = faces_len as u64;

            if gpu_data.wireframe_edges.is_none()
                || gpu_data.wireframe_edges_mesh_hash != faces_hash
            {
                let coords_guard = mesh.coords().read().unwrap();
                let faces_guard = mesh.faces().read().unwrap();

                if let (Some(coords), Some(faces)) = (coords_guard.data(), faces_guard.data()) {
                    let mut edges = Vec::with_capacity(faces.len() * 3);
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

            // Get edges info and convert to GPU format first
            let (num_edges, gpu_edges) = {
                let edges = match &gpu_data.wireframe_edges {
                    Some(e) => e,
                    None => return,
                };
                let num = edges.len();
                if num == 0 {
                    return;
                }
                let gpu_e: Vec<GpuEdge> = edges
                    .iter()
                    .map(|(a, b)| GpuEdge {
                        point_a: a.coords.into(),
                        _pad_a: 0.0,
                        point_b: b.coords.into(),
                        _pad_b: 0.0,
                    })
                    .collect();
                (num, gpu_e)
            };

            // Now we can safely mutate gpu_data since edges borrow is done
            {
                // Load wireframe instance buffers to GPU
                instances.lines_colors.load_to_gpu();
                instances.lines_widths.load_to_gpu();

                let inst_lines_colors_buf = match instances.lines_colors.buffer() {
                    Some(b) => b,
                    None => return,
                };
                let inst_lines_widths_buf = match instances.lines_widths.buffer() {
                    Some(b) => b,
                    None => return,
                };

                // Ensure edge buffer capacity and upload edges
                gpu_data.ensure_edge_buffer_capacity(num_edges);

                ctxt.write_buffer(
                    &gpu_data.wireframe_edge_buffer,
                    0,
                    bytemuck::cast_slice(&gpu_edges),
                );

                // Update wireframe view uniforms
                let wireframe_view_uniforms = WireframeViewUniforms {
                    view: view.to_homogeneous().into(),
                    proj: proj.into(),
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
                let wireframe_color = data.lines_color().unwrap_or(data.color());
                let wireframe_model_uniforms = WireframeModelUniforms {
                    transform: formatted_transform.into(),
                    scale: (*scale).into(),
                    num_edges: num_edges as u32,
                    default_color: [wireframe_color.x, wireframe_color.y, wireframe_color.z, 1.0],
                    default_width: data.lines_width(),
                    use_perspective: if data.lines_use_perspective() { 1 } else { 0 },
                    _padding: [0.0; 2],
                };
                ctxt.write_buffer(
                    &gpu_data.wireframe_model_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&wireframe_model_uniforms),
                );

                // Get or create wireframe bind groups
                if gpu_data.wireframe_view_bind_group.is_none() {
                    gpu_data.wireframe_view_bind_group =
                        Some(self.create_wireframe_view_bind_group(
                            &gpu_data.wireframe_view_uniform_buffer,
                        ));
                }
                if gpu_data.wireframe_model_bind_group.is_none() {
                    let edge_size = (num_edges * std::mem::size_of::<GpuEdge>()) as u64;
                    gpu_data.wireframe_model_bind_group =
                        Some(self.create_wireframe_model_bind_group(
                            &gpu_data.wireframe_model_uniform_buffer,
                            &gpu_data.wireframe_edge_buffer,
                            edge_size,
                        ));
                }

                let wireframe_view_bind_group =
                    gpu_data.wireframe_view_bind_group.as_ref().unwrap();
                let wireframe_model_bind_group =
                    gpu_data.wireframe_model_bind_group.as_ref().unwrap();

                // Begin wireframe render pass
                let mut render_pass =
                    context
                        .encoder
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("wireframe_render_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: context.color_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: context.depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });

                render_pass.set_pipeline(&self.wireframe_pipeline);
                render_pass.set_bind_group(0, wireframe_view_bind_group, &[]);
                render_pass.set_bind_group(1, wireframe_model_bind_group, &[]);

                // Set instance vertex buffers (5 total: positions, colors, deformations, lines_colors, lines_widths)
                render_pass.set_vertex_buffer(0, inst_positions_buf.slice(..));
                render_pass.set_vertex_buffer(1, inst_colors_buf.slice(..));
                render_pass.set_vertex_buffer(2, inst_deformations_buf.slice(..)); // Contains all 3 columns
                render_pass.set_vertex_buffer(3, inst_lines_colors_buf.slice(..));
                render_pass.set_vertex_buffer(4, inst_lines_widths_buf.slice(..));

                // Draw: 6 vertices per edge (computed from vertex_index), num_instances instances
                let num_vertices = (num_edges * 6) as u32;
                render_pass.draw(0..num_vertices, 0..num_instances as u32);
            }
        }

        // Render points
        if render_points {
            // Build vertex cache if needed (using mesh coords hash)
            let coords_hash = {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                let coords = mesh.coords().read().unwrap();
                coords.len().hash(&mut hasher);
                // Simple hash based on length - vertices rarely change
                hasher.finish()
            };

            if gpu_data.points_vertices.is_none()
                || gpu_data.points_vertices_mesh_hash != coords_hash
            {
                // Rebuild vertex cache from mesh coords
                let coords_guard = mesh.coords().read().unwrap();
                if let Some(coords) = coords_guard.data() {
                    gpu_data.points_vertices = Some(coords.to_vec());
                    gpu_data.points_vertices_mesh_hash = coords_hash;
                    // Invalidate model bind group since vertices changed
                    gpu_data.points_model_bind_group = None;
                }
            }

            // Get vertices info and convert to GPU format
            let (num_vertices, gpu_vertices) = {
                let vertices = match &gpu_data.points_vertices {
                    Some(v) => v,
                    None => return,
                };
                let num = vertices.len();
                if num == 0 {
                    return;
                }
                let gpu_v: Vec<GpuVertex> = vertices
                    .iter()
                    .map(|p| GpuVertex {
                        position: p.coords.into(),
                        _pad: 0.0,
                    })
                    .collect();
                (num, gpu_v)
            };

            // Now we can safely mutate gpu_data since vertices borrow is done
            {
                // Load point instance buffers to GPU
                instances.points_colors.load_to_gpu();
                instances.points_sizes.load_to_gpu();

                let inst_points_colors_buf = match instances.points_colors.buffer() {
                    Some(b) => b,
                    None => return,
                };
                let inst_points_sizes_buf = match instances.points_sizes.buffer() {
                    Some(b) => b,
                    None => return,
                };

                // Ensure vertex buffer capacity and upload vertices
                gpu_data.ensure_vertex_buffer_capacity(num_vertices);

                ctxt.write_buffer(
                    &gpu_data.points_vertex_buffer,
                    0,
                    bytemuck::cast_slice(&gpu_vertices),
                );

                // Update points view uniforms (same format as wireframe)
                let points_view_uniforms = WireframeViewUniforms {
                    view: view.to_homogeneous().into(),
                    proj: proj.into(),
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
                let points_color = data.points_color().unwrap_or(data.color());
                let points_model_uniforms = PointsModelUniforms {
                    transform: formatted_transform.into(),
                    scale: (*scale).into(),
                    num_vertices: num_vertices as u32,
                    default_color: [points_color.x, points_color.y, points_color.z, 1.0],
                    default_size: data.points_size(),
                    use_perspective: if data.points_use_perspective() { 1 } else { 0 },
                    _padding: [0.0; 2],
                };
                ctxt.write_buffer(
                    &gpu_data.points_model_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&points_model_uniforms),
                );

                // Get or create points bind groups
                if gpu_data.points_view_bind_group.is_none() {
                    gpu_data.points_view_bind_group = Some(
                        self.create_points_view_bind_group(&gpu_data.points_view_uniform_buffer),
                    );
                }
                if gpu_data.points_model_bind_group.is_none() {
                    let vertex_size = (num_vertices * std::mem::size_of::<GpuVertex>()) as u64;
                    gpu_data.points_model_bind_group = Some(self.create_points_model_bind_group(
                        &gpu_data.points_model_uniform_buffer,
                        &gpu_data.points_vertex_buffer,
                        vertex_size,
                    ));
                }

                let points_view_bind_group = gpu_data.points_view_bind_group.as_ref().unwrap();
                let points_model_bind_group = gpu_data.points_model_bind_group.as_ref().unwrap();

                // Begin points render pass
                let mut render_pass =
                    context
                        .encoder
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("points_render_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: context.color_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: context.depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });

                render_pass.set_pipeline(&self.points_pipeline);
                render_pass.set_bind_group(0, points_view_bind_group, &[]);
                render_pass.set_bind_group(1, points_model_bind_group, &[]);

                // Set instance vertex buffers (5 total: positions, colors, deformations, points_colors, points_sizes)
                render_pass.set_vertex_buffer(0, inst_positions_buf.slice(..));
                render_pass.set_vertex_buffer(1, inst_colors_buf.slice(..));
                render_pass.set_vertex_buffer(2, inst_deformations_buf.slice(..)); // Contains all 3 columns
                render_pass.set_vertex_buffer(3, inst_points_colors_buf.slice(..));
                render_pass.set_vertex_buffer(4, inst_points_sizes_buf.slice(..));

                // Draw: 6 vertices per point (computed from vertex_index), num_instances instances
                let num_draw_vertices = (num_vertices * 6) as u32;
                render_pass.draw(0..num_draw_vertices, 0..num_instances as u32);
            }
        }
    }
}

/// Vertex shader of the default object material.
pub static OBJECT_VERTEX_SRC: &str = include_str!("default.wgsl");
/// Fragment shader of the default object material.
pub static OBJECT_FRAGMENT_SRC: &str = include_str!("default.wgsl");
