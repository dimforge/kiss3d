use crate::camera::Camera3d;
use crate::context::Context;
use crate::light::{LightCollection, LightType, MAX_LIGHTS};
use crate::resource::vertex_index::VERTEX_INDEX_FORMAT;
use crate::resource::{DynamicUniformBuffer, GpuData, GpuMesh3d, Material3d, RenderContext, Texture};
use crate::scene::{InstancesBuffer3d, ObjectData3d};
use bytemuck::{Pod, Zeroable};
use glamx::{Mat3, Pose3, Vec3};
use std::any::Any;
use std::cell::Cell;

/// GPU representation of a single light.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuLight {
    position: [f32; 3],
    light_type: u32, // 0=point, 1=directional, 2=spot
    direction: [f32; 3],
    intensity: f32,
    color: [f32; 3],
    inner_cone_cos: f32,
    outer_cone_cos: f32,
    attenuation_radius: f32,
    _padding: [f32; 2],
}

impl Default for GpuLight {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            light_type: 0,
            direction: [0.0, 0.0, -1.0],
            intensity: 0.0,
            color: [1.0, 1.0, 1.0],
            inner_cone_cos: 1.0,
            outer_cone_cos: 0.0,
            attenuation_radius: 100.0,
            _padding: [0.0; 2],
        }
    }
}

/// Frame-level uniforms (view, projection, lights).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FrameUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    lights: [GpuLight; MAX_LIGHTS],
    num_lights: u32,
    ambient_intensity: f32,
    _padding: [f32; 2],
}

/// Object-level uniforms (transform, scale, color, PBR properties).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ObjectUniforms {
    transform: [[f32; 4]; 4],
    ntransform: [[f32; 4]; 3], // mat3x3 padded to mat3x4 for alignment
    scale: [[f32; 4]; 3],      // mat3x3 padded to mat3x4 for alignment
    color: [f32; 4],
    metallic: f32,
    roughness: f32,
    _pad0: [f32; 2],
    emissive: [f32; 4],
    // Texture presence flags (0.0 or 1.0 - WGSL doesn't support bool in uniforms)
    has_normal_map: f32,
    has_metallic_roughness_map: f32,
    has_ao_map: f32,
    has_emissive_map: f32,
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
/// This struct now only contains wireframe and points rendering data.
/// The main uniform buffers and shared view uniforms are managed by ObjectMaterial.
pub struct ObjectMaterialGpuData {
    // Cached texture bind group with pointer to detect texture changes
    texture_bind_group: Option<wgpu::BindGroup>,
    cached_texture_ptr: usize,
    /// Offset into the dynamic object uniform buffer, set during prepare() phase.
    object_uniform_offset: Option<u32>,
    // PBR texture bind group (normal, metallic-roughness, ao, emissive maps)
    pbr_texture_bind_group: Option<wgpu::BindGroup>,
    cached_normal_map_ptr: usize,
    cached_metallic_roughness_map_ptr: usize,
    cached_ao_map_ptr: usize,
    cached_emissive_map_ptr: usize,
    // Wireframe rendering data (model uniforms are per-object)
    wireframe_model_uniform_buffer: wgpu::Buffer,
    wireframe_edge_buffer: wgpu::Buffer,
    wireframe_edge_capacity: usize,
    wireframe_model_bind_group: Option<wgpu::BindGroup>,
    /// Cached wireframe edges in local coordinates (built lazily from mesh).
    wireframe_edges: Option<Vec<(Vec3, Vec3)>>,
    /// Hash of mesh faces to detect when edges need rebuilding.
    wireframe_edges_mesh_hash: u64,
    /// Cached wireframe model uniforms (written during prepare).
    wireframe_model_uniforms: WireframeModelUniforms,
    // Point rendering data (model uniforms are per-object)
    points_model_uniform_buffer: wgpu::Buffer,
    points_vertex_buffer: wgpu::Buffer,
    points_vertex_capacity: usize,
    points_model_bind_group: Option<wgpu::BindGroup>,
    /// Cached vertices for point rendering (built lazily from mesh).
    points_vertices: Option<Vec<Vec3>>,
    /// Hash of mesh coords to detect when vertices need rebuilding.
    points_vertices_mesh_hash: u64,
    /// Cached points model uniforms (written during prepare).
    points_model_uniforms: PointsModelUniforms,
}

impl ObjectMaterialGpuData {
    /// Creates new per-object GPU data.
    pub fn new() -> Self {
        let ctxt = Context::get();

        // Wireframe model uniform buffer (per-object)
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

        // Point model uniform buffer (per-object)
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
            texture_bind_group: None,
            cached_texture_ptr: 0,
            object_uniform_offset: None,
            // PBR texture caching
            pbr_texture_bind_group: None,
            cached_normal_map_ptr: 0,
            cached_metallic_roughness_map_ptr: 0,
            cached_ao_map_ptr: 0,
            cached_emissive_map_ptr: 0,
            // Wireframe rendering
            wireframe_model_uniform_buffer,
            wireframe_edge_buffer,
            wireframe_edge_capacity,
            wireframe_model_bind_group: None,
            wireframe_edges: None,
            wireframe_edges_mesh_hash: 0,
            wireframe_model_uniforms: WireframeModelUniforms {
                transform: [[0.0; 4]; 4],
                scale: [0.0; 3],
                num_edges: 0,
                default_color: [0.0; 4],
                default_width: 0.0,
                use_perspective: 0,
                _padding: [0.0; 2],
            },
            points_model_uniform_buffer,
            points_vertex_buffer,
            points_vertex_capacity,
            points_model_bind_group: None,
            points_vertices: None,
            points_vertices_mesh_hash: 0,
            points_model_uniforms: PointsModelUniforms {
                transform: [[0.0; 4]; 4],
                scale: [0.0; 3],
                num_vertices: 0,
                default_color: [0.0; 4],
                default_size: 0.0,
                use_perspective: 0,
                _padding: [0.0; 2],
            },
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
/// This struct holds shared resources (pipeline, bind group layouts, dynamic buffers)
/// that are used by all objects. Per-object resources for wireframe/points are stored
/// in `ObjectMaterialGpuData` instances.
///
/// ## Performance Optimization
///
/// This material uses dynamic uniform buffers to batch uniform data writes:
/// - Frame uniforms (view, projection, light) are written once per frame
/// - Object uniforms are accumulated in a dynamic buffer and flushed once
/// - Wireframe/points view uniforms (view, proj, viewport) are shared and written once per frame
/// - This significantly reduces the number of `write_buffer` calls per frame
pub struct ObjectMaterial {
    /// Pipeline with backface culling enabled
    pipeline_cull: wgpu::RenderPipeline,
    /// Pipeline with backface culling disabled
    pipeline_no_cull: wgpu::RenderPipeline,
    object_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    /// PBR texture bind group layout (normal, metallic-roughness, ao, emissive maps)
    pbr_texture_bind_group_layout: wgpu::BindGroupLayout,
    /// Default PBR textures for when user hasn't set any
    default_normal_map: std::sync::Arc<crate::resource::Texture>,
    default_metallic_roughness_map: std::sync::Arc<crate::resource::Texture>,
    default_ao_map: std::sync::Arc<crate::resource::Texture>,
    default_emissive_map: std::sync::Arc<crate::resource::Texture>,
    // Wireframe rendering resources
    wireframe_pipeline: wgpu::RenderPipeline,
    wireframe_model_bind_group_layout: wgpu::BindGroupLayout,
    // Point rendering resources
    points_pipeline: wgpu::RenderPipeline,
    points_model_bind_group_layout: wgpu::BindGroupLayout,

    // === Dynamic uniform buffer system ===
    /// Shared frame uniform buffer (view, projection, light)
    frame_uniform_buffer: wgpu::Buffer,
    /// Shared bind group for frame uniforms
    frame_bind_group: wgpu::BindGroup,
    /// Dynamic buffer for object uniforms
    object_uniform_buffer: DynamicUniformBuffer<ObjectUniforms>,
    /// Bind group for object uniforms (recreated when buffer grows)
    object_bind_group: Option<wgpu::BindGroup>,
    /// Capacity when bind group was last created (to detect regrowth)
    object_bind_group_capacity: u64,
    /// Frame counter for detecting new frames
    frame_counter: Cell<u64>,
    /// Last frame we processed (to detect new frame)
    last_frame: Cell<u64>,

    // === Shared wireframe/points view uniforms ===
    /// Shared wireframe view uniform buffer (view, proj, viewport - same for all objects)
    wireframe_view_uniform_buffer: wgpu::Buffer,
    /// Shared wireframe view bind group
    wireframe_view_bind_group: wgpu::BindGroup,
    /// Shared points view uniform buffer (view, proj, viewport - same for all objects)
    points_view_uniform_buffer: wgpu::Buffer,
    /// Shared points view bind group
    points_view_bind_group: wgpu::BindGroup,
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

        // Object bind group uses dynamic offset for batched uniforms
        let object_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("object_material_object_bind_group_layout"),
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

        // PBR texture bind group layout (group 3): normal, metallic-roughness, ao, emissive maps
        let pbr_texture_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("object_material_pbr_texture_bind_group_layout"),
                entries: &[
                    // Normal map
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
                    // Metallic-roughness map
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Ambient occlusion map
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Emissive map
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create default PBR textures
        let default_normal_map = crate::resource::Texture::new_default_normal_map();
        let default_metallic_roughness_map = crate::resource::Texture::new_default_metallic_roughness_map();
        let default_ao_map = crate::resource::Texture::new_default_ao_map();
        let default_emissive_map = crate::resource::Texture::new_default_emissive_map();

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("object_material_pipeline_layout"),
            bind_group_layouts: &[
                &frame_bind_group_layout,
                &object_bind_group_layout,
                &texture_bind_group_layout,
                &pbr_texture_bind_group_layout,
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
            include_str!("wireframe_polyline3d.wgsl"),
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
            include_str!("wireframe_points3d.wgsl"),
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

        // === Create shared dynamic buffer resources ===

        // Frame uniform buffer (written once per frame)
        let frame_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shared_frame_uniform_buffer"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create frame bind group
        let frame_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared_frame_bind_group"),
            layout: &frame_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_uniform_buffer.as_entire_binding(),
            }],
        });

        // Dynamic buffer for object uniforms
        let object_uniform_buffer =
            DynamicUniformBuffer::<ObjectUniforms>::new("dynamic_object_uniform_buffer");

        // Create initial object bind group
        let object_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dynamic_object_bind_group"),
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

        // Get capacity before moving into struct
        let object_bind_group_capacity = object_uniform_buffer.capacity();

        // === Shared wireframe/points view uniform buffers ===
        // These contain view, proj, and viewport which are the same for all objects in a frame

        let wireframe_view_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shared_wireframe_view_uniform_buffer"),
            size: std::mem::size_of::<WireframeViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let wireframe_view_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared_wireframe_view_bind_group"),
            layout: &wireframe_view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wireframe_view_uniform_buffer.as_entire_binding(),
            }],
        });

        let points_view_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shared_points_view_uniform_buffer"),
            size: std::mem::size_of::<WireframeViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let points_view_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared_points_view_bind_group"),
            layout: &points_view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: points_view_uniform_buffer.as_entire_binding(),
            }],
        });

        ObjectMaterial {
            pipeline_cull,
            pipeline_no_cull,
            object_bind_group_layout,
            texture_bind_group_layout,
            pbr_texture_bind_group_layout,
            default_normal_map,
            default_metallic_roughness_map,
            default_ao_map,
            default_emissive_map,
            wireframe_pipeline,
            wireframe_model_bind_group_layout,
            points_pipeline,
            points_model_bind_group_layout,
            frame_uniform_buffer,
            frame_bind_group,
            object_uniform_buffer,
            object_bind_group: Some(object_bind_group),
            object_bind_group_capacity,
            frame_counter: Cell::new(0),
            last_frame: Cell::new(u64::MAX),
            wireframe_view_uniform_buffer,
            wireframe_view_bind_group,
            points_view_uniform_buffer,
            points_view_bind_group,
        }
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

    fn create_pbr_texture_bind_group(
        &self,
        normal_map: &Texture,
        metallic_roughness_map: &Texture,
        ao_map: &Texture,
        emissive_map: &Texture,
    ) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("object_material_pbr_texture_bind_group"),
            layout: &self.pbr_texture_bind_group_layout,
            entries: &[
                // Normal map
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&normal_map.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&normal_map.sampler),
                },
                // Metallic-roughness map
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&metallic_roughness_map.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&metallic_roughness_map.sampler),
                },
                // Ambient occlusion map
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&ao_map.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&ao_map.sampler),
                },
                // Emissive map
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&emissive_map.view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&emissive_map.sampler),
                },
            ],
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

    /// Signals the start of a new frame.
    ///
    /// This clears the dynamic object uniform buffer and resets the frame counter.
    /// Should be called before rendering any objects for a new frame.
    pub fn begin_frame(&mut self) {
        self.frame_counter
            .set(self.frame_counter.get().wrapping_add(1));
        self.object_uniform_buffer.clear();
    }

    /// Flushes the accumulated object uniforms to the GPU.
    ///
    /// This performs a single `write_buffer` call with all accumulated object data.
    /// Should be called after all objects have been processed for the frame.
    pub fn flush(&mut self) {
        let ctxt = Context::get();

        self.object_uniform_buffer.flush();

        // Recreate bind group if buffer grew
        if self.object_uniform_buffer.capacity() != self.object_bind_group_capacity {
            self.object_bind_group = Some(ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dynamic_object_bind_group"),
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
            self.object_bind_group_capacity = self.object_uniform_buffer.capacity();
        }
    }
}

impl Material3d for ObjectMaterial {
    fn create_gpu_data(&self) -> Box<dyn GpuData> {
        Box::new(ObjectMaterialGpuData::new())
    }

    fn begin_frame(&mut self) {
        self.frame_counter
            .set(self.frame_counter.get().wrapping_add(1));
        self.object_uniform_buffer.clear();
    }

    fn prepare(
        &mut self,
        pass: usize,
        transform: Pose3,
        scale: Vec3,
        camera: &mut dyn Camera3d,
        lights: &LightCollection,
        data: &ObjectData3d,
        gpu_data: &mut dyn GpuData,
        viewport_width: u32,
        viewport_height: u32,
    ) {
        let ctxt = Context::get();

        // Downcast gpu_data to our specific type
        let gpu_data = gpu_data
            .as_any_mut()
            .downcast_mut::<ObjectMaterialGpuData>()
            .expect("ObjectMaterial requires ObjectMaterialGpuData");

        // Check if this is a new frame (first object being prepared)
        let current_frame = self.frame_counter.get();
        let is_new_frame = current_frame != self.last_frame.get();

        if is_new_frame {
            self.last_frame.set(current_frame);

            // Write frame uniforms once per frame
            let (view, proj) = camera.view_transform_pair(pass);

            // Convert collected lights to GPU format
            let mut gpu_lights: [GpuLight; MAX_LIGHTS] = [GpuLight::default(); MAX_LIGHTS];
            for (i, collected_light) in lights.lights.iter().take(MAX_LIGHTS).enumerate() {
                let (light_type, inner_cone_cos, outer_cone_cos, attenuation_radius) =
                    match &collected_light.light_type {
                        LightType::Point { attenuation_radius } => (0u32, 1.0, 0.0, *attenuation_radius),
                        LightType::Directional(_) => (1u32, 1.0, 0.0, 0.0),
                        LightType::Spot {
                            inner_cone_angle,
                            outer_cone_angle,
                            attenuation_radius,
                        } => (
                            2u32,
                            inner_cone_angle.cos(),
                            outer_cone_angle.cos(),
                            *attenuation_radius,
                        ),
                    };

                gpu_lights[i] = GpuLight {
                    position: collected_light.world_position.into(),
                    light_type,
                    direction: collected_light.world_direction.into(),
                    intensity: collected_light.intensity,
                    color: collected_light.color.into(),
                    inner_cone_cos,
                    outer_cone_cos,
                    attenuation_radius,
                    _padding: [0.0; 2],
                };
            }

            let frame_uniforms = FrameUniforms {
                view: view.to_mat4().to_cols_array_2d(),
                proj: proj.to_cols_array_2d(),
                lights: gpu_lights,
                num_lights: lights.lights.len().min(MAX_LIGHTS) as u32,
                ambient_intensity: lights.ambient,
                _padding: [0.0; 2],
            };

            ctxt.write_buffer(
                &self.frame_uniform_buffer,
                0,
                bytemuck::bytes_of(&frame_uniforms),
            );

            // Write shared wireframe/points view uniforms once per frame
            // These contain view, proj, viewport which are same for all objects
            let wireframe_view_uniforms = WireframeViewUniforms {
                view: view.to_mat4().to_cols_array_2d(),
                proj: proj.to_cols_array_2d(),
                viewport: [0.0, 0.0, viewport_width as f32, viewport_height as f32],
            };

            ctxt.write_buffer(
                &self.wireframe_view_uniform_buffer,
                0,
                bytemuck::bytes_of(&wireframe_view_uniforms),
            );

            // Points use the same view uniform layout
            ctxt.write_buffer(
                &self.points_view_uniform_buffer,
                0,
                bytemuck::bytes_of(&wireframe_view_uniforms),
            );
        }

        // Create object uniforms
        let formatted_transform = transform.to_mat4();
        let ntransform = Mat3::from_quat(transform.rotation);
        let formatted_scale = Mat3::from_diagonal(scale);

        // Pad mat3x3 to mat3x4 for proper alignment
        let ntransform_cols = ntransform.to_cols_array_2d();
        let ntransform_padded: [[f32; 4]; 3] = [
            [
                ntransform_cols[0][0],
                ntransform_cols[0][1],
                ntransform_cols[0][2],
                0.0,
            ],
            [
                ntransform_cols[1][0],
                ntransform_cols[1][1],
                ntransform_cols[1][2],
                0.0,
            ],
            [
                ntransform_cols[2][0],
                ntransform_cols[2][1],
                ntransform_cols[2][2],
                0.0,
            ],
        ];
        let scale_cols = formatted_scale.to_cols_array_2d();
        let scale_padded: [[f32; 4]; 3] = [
            [scale_cols[0][0], scale_cols[0][1], scale_cols[0][2], 0.0],
            [scale_cols[1][0], scale_cols[1][1], scale_cols[1][2], 0.0],
            [scale_cols[2][0], scale_cols[2][1], scale_cols[2][2], 0.0],
        ];

        let color = data.color();
        let emissive = data.emissive();
        let object_uniforms = ObjectUniforms {
            transform: formatted_transform.to_cols_array_2d(),
            ntransform: ntransform_padded,
            scale: scale_padded,
            color: [color.r, color.g, color.b, color.a],
            metallic: data.metallic(),
            roughness: data.roughness(),
            _pad0: [0.0; 2],
            emissive: [emissive.r, emissive.g, emissive.b, emissive.a],
            has_normal_map: if data.normal_map().is_some() { 1.0 } else { 0.0 },
            has_metallic_roughness_map: if data.metallic_roughness_map().is_some() { 1.0 } else { 0.0 },
            has_ao_map: if data.ao_map().is_some() { 1.0 } else { 0.0 },
            has_emissive_map: if data.emissive_map().is_some() { 1.0 } else { 0.0 },
        };

        // Push to dynamic buffer and store offset in gpu_data
        let object_offset = self.object_uniform_buffer.push(&object_uniforms);
        gpu_data.object_uniform_offset = Some(object_offset);

        // Prepare wireframe model uniforms if needed (view uniforms are shared)
        let render_wireframe = data.lines_width() > 0.0;
        if render_wireframe {
            // Compute model uniforms (num_edges will be set in render when mesh is available)
            let wireframe_color = data.lines_color().unwrap_or(data.color());
            let cached_num_edges = gpu_data
                .wireframe_edges
                .as_ref()
                .map(|e| e.len())
                .unwrap_or(0) as u32;
            gpu_data.wireframe_model_uniforms = WireframeModelUniforms {
                transform: formatted_transform.to_cols_array_2d(),
                scale: scale.into(),
                num_edges: cached_num_edges,
                default_color: [
                    wireframe_color.r,
                    wireframe_color.g,
                    wireframe_color.b,
                    wireframe_color.a,
                ],
                default_width: data.lines_width(),
                use_perspective: if data.lines_use_perspective() { 1 } else { 0 },
                _padding: [0.0; 2],
            };

            // Write model uniforms to GPU (view uniforms are shared and written once per frame)
            ctxt.write_buffer(
                &gpu_data.wireframe_model_uniform_buffer,
                0,
                bytemuck::bytes_of(&gpu_data.wireframe_model_uniforms),
            );
        }

        // Prepare points model uniforms if needed (view uniforms are shared)
        let render_points = data.points_size() > 0.0;
        if render_points {
            // Compute model uniforms (num_vertices will be set in render when mesh is available)
            let points_color = data.points_color().unwrap_or(data.color());
            let cached_num_vertices = gpu_data
                .points_vertices
                .as_ref()
                .map(|v| v.len())
                .unwrap_or(0) as u32;
            gpu_data.points_model_uniforms = PointsModelUniforms {
                transform: formatted_transform.to_cols_array_2d(),
                scale: scale.into(),
                num_vertices: cached_num_vertices,
                default_color: [points_color.r, points_color.g, points_color.b, points_color.a],
                default_size: data.points_size(),
                use_perspective: if data.points_use_perspective() { 1 } else { 0 },
                _padding: [0.0; 2],
            };

            // Write model uniforms to GPU (view uniforms are shared and written once per frame)
            ctxt.write_buffer(
                &gpu_data.points_model_uniform_buffer,
                0,
                bytemuck::bytes_of(&gpu_data.points_model_uniforms),
            );
        }
    }

    fn flush(&mut self) {
        let ctxt = Context::get();

        self.object_uniform_buffer.flush();

        // Recreate bind group if buffer grew
        if self.object_uniform_buffer.capacity() != self.object_bind_group_capacity {
            self.object_bind_group = Some(ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dynamic_object_bind_group"),
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
            self.object_bind_group_capacity = self.object_uniform_buffer.capacity();
        }
    }

    fn render(
        &mut self,
        _pass: usize,
        _transform: Pose3,
        _scale: Vec3,
        _camera: &mut dyn Camera3d,
        _lights: &LightCollection,
        data: &ObjectData3d,
        mesh: &mut GpuMesh3d,
        instances: &mut InstancesBuffer3d,
        gpu_data: &mut dyn GpuData,
        render_pass: &mut wgpu::RenderPass<'_>,
        _context: &RenderContext,
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

        // Get the pre-computed object uniform offset from prepare() phase
        let object_offset = gpu_data
            .object_uniform_offset
            .expect("prepare() must be called before render()");

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

        // Cache texture bind group, invalidate if texture changed
        let texture_ptr = std::sync::Arc::as_ptr(data.texture()) as usize;
        if gpu_data.texture_bind_group.is_none() || gpu_data.cached_texture_ptr != texture_ptr {
            gpu_data.texture_bind_group = Some(self.create_texture_bind_group(data.texture()));
            gpu_data.cached_texture_ptr = texture_ptr;
        }

        // Cache PBR texture bind group, invalidate if any PBR texture changed
        let normal_map = data.normal_map().unwrap_or(&self.default_normal_map);
        let metallic_roughness_map = data.metallic_roughness_map().unwrap_or(&self.default_metallic_roughness_map);
        let ao_map = data.ao_map().unwrap_or(&self.default_ao_map);
        let emissive_map = data.emissive_map().unwrap_or(&self.default_emissive_map);

        let normal_ptr = std::sync::Arc::as_ptr(normal_map) as usize;
        let mr_ptr = std::sync::Arc::as_ptr(metallic_roughness_map) as usize;
        let ao_ptr = std::sync::Arc::as_ptr(ao_map) as usize;
        let emissive_ptr = std::sync::Arc::as_ptr(emissive_map) as usize;

        let pbr_textures_changed = gpu_data.pbr_texture_bind_group.is_none()
            || gpu_data.cached_normal_map_ptr != normal_ptr
            || gpu_data.cached_metallic_roughness_map_ptr != mr_ptr
            || gpu_data.cached_ao_map_ptr != ao_ptr
            || gpu_data.cached_emissive_map_ptr != emissive_ptr;

        if pbr_textures_changed {
            gpu_data.pbr_texture_bind_group = Some(self.create_pbr_texture_bind_group(
                normal_map,
                metallic_roughness_map,
                ao_map,
                emissive_map,
            ));
            gpu_data.cached_normal_map_ptr = normal_ptr;
            gpu_data.cached_metallic_roughness_map_ptr = mr_ptr;
            gpu_data.cached_ao_map_ptr = ao_ptr;
            gpu_data.cached_emissive_map_ptr = emissive_ptr;
        }

        // Render surface (filled triangles)
        if render_surface {
            let texture_bind_group = gpu_data.texture_bind_group.as_ref().unwrap();
            let pbr_texture_bind_group = gpu_data.pbr_texture_bind_group.as_ref().unwrap();
            let object_bind_group = self.object_bind_group.as_ref().unwrap();

            // Select pipeline based on backface culling setting
            let pipeline = if data.backface_culling_enabled() {
                &self.pipeline_cull
            } else {
                &self.pipeline_no_cull
            };
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &self.frame_bind_group, &[]);
            // Use dynamic offset for object uniforms!
            render_pass.set_bind_group(1, object_bind_group, &[object_offset]);
            render_pass.set_bind_group(2, texture_bind_group, &[]);
            render_pass.set_bind_group(3, pbr_texture_bind_group, &[]);

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
                        point_a: (*a).into(),
                        _pad_a: 0.0,
                        point_b: (*b).into(),
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

                // Ensure edge buffer capacity and upload edges (geometry data)
                gpu_data.ensure_edge_buffer_capacity(num_edges);

                ctxt.write_buffer(
                    &gpu_data.wireframe_edge_buffer,
                    0,
                    bytemuck::cast_slice(&gpu_edges),
                );

                // Update num_edges in model uniforms if it changed from prepare()
                if gpu_data.wireframe_model_uniforms.num_edges != num_edges as u32 {
                    gpu_data.wireframe_model_uniforms.num_edges = num_edges as u32;
                    ctxt.write_buffer(
                        &gpu_data.wireframe_model_uniform_buffer,
                        0,
                        bytemuck::bytes_of(&gpu_data.wireframe_model_uniforms),
                    );
                }

                // Get or create wireframe model bind group (view bind group is shared)
                if gpu_data.wireframe_model_bind_group.is_none() {
                    let edge_size = (num_edges * std::mem::size_of::<GpuEdge>()) as u64;
                    gpu_data.wireframe_model_bind_group =
                        Some(self.create_wireframe_model_bind_group(
                            &gpu_data.wireframe_model_uniform_buffer,
                            &gpu_data.wireframe_edge_buffer,
                            edge_size,
                        ));
                }

                let wireframe_model_bind_group =
                    gpu_data.wireframe_model_bind_group.as_ref().unwrap();

                render_pass.set_pipeline(&self.wireframe_pipeline);
                // Use shared view bind group (written once per frame)
                render_pass.set_bind_group(0, &self.wireframe_view_bind_group, &[]);
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
                        position: (*p).into(),
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

                // Ensure vertex buffer capacity and upload vertices (geometry data)
                gpu_data.ensure_vertex_buffer_capacity(num_vertices);

                ctxt.write_buffer(
                    &gpu_data.points_vertex_buffer,
                    0,
                    bytemuck::cast_slice(&gpu_vertices),
                );

                // Update num_vertices in model uniforms if it changed from prepare()
                if gpu_data.points_model_uniforms.num_vertices != num_vertices as u32 {
                    gpu_data.points_model_uniforms.num_vertices = num_vertices as u32;
                    ctxt.write_buffer(
                        &gpu_data.points_model_uniform_buffer,
                        0,
                        bytemuck::bytes_of(&gpu_data.points_model_uniforms),
                    );
                }

                // Get or create points model bind group (view bind group is shared)
                if gpu_data.points_model_bind_group.is_none() {
                    let vertex_size = (num_vertices * std::mem::size_of::<GpuVertex>()) as u64;
                    gpu_data.points_model_bind_group = Some(self.create_points_model_bind_group(
                        &gpu_data.points_model_uniform_buffer,
                        &gpu_data.points_vertex_buffer,
                        vertex_size,
                    ));
                }

                let points_model_bind_group = gpu_data.points_model_bind_group.as_ref().unwrap();

                render_pass.set_pipeline(&self.points_pipeline);
                // Use shared view bind group (written once per frame)
                render_pass.set_bind_group(0, &self.points_view_bind_group, &[]);
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
