use crate::camera::Camera3d;
use crate::context::Context;
use crate::light::LightCollection;
use crate::resource::vertex_index::VERTEX_INDEX_FORMAT;
use crate::resource::{DynamicUniformBuffer, GpuData, GpuMesh3d, Material3d, RenderContext};
use crate::scene::{InstancesBuffer3d, ObjectData3d};
use bytemuck::{Pod, Zeroable};
use glamx::{Mat3, Pose3, Vec3};
use std::any::Any;
use std::cell::Cell;

/// Frame-level uniforms (view, projection).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FrameUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
}

/// Object-level uniforms (transform, scale).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ObjectUniforms {
    transform: [[f32; 4]; 4],
    scale: [[f32; 4]; 3], // mat3x3 padded to mat3x4 for alignment
    _padding: [f32; 4],
}

/// Per-object GPU data for NormalsMaterial.
pub struct NormalsMaterialGpuData {
    /// Offset into the shared dynamic object uniform buffer.
    object_uniform_offset: Option<u32>,
}

impl NormalsMaterialGpuData {
    pub fn new() -> Self {
        Self {
            object_uniform_offset: None,
        }
    }
}

impl Default for NormalsMaterialGpuData {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuData for NormalsMaterialGpuData {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A material that draws normals of an object.
///
/// ## Performance Optimization
///
/// This material uses dynamic uniform buffers to batch uniform data writes:
/// - Frame uniforms (view, projection) are written once per frame
/// - Object uniforms are accumulated in a dynamic buffer and flushed once
pub struct NormalsMaterial {
    /// Pipeline with backface culling enabled
    pipeline_cull: wgpu::RenderPipeline,
    /// Pipeline with backface culling disabled
    pipeline_no_cull: wgpu::RenderPipeline,
    object_bind_group_layout: wgpu::BindGroupLayout,

    // === Dynamic uniform buffer system ===
    /// Shared frame uniform buffer
    frame_uniform_buffer: wgpu::Buffer,
    /// Shared frame bind group
    frame_bind_group: wgpu::BindGroup,
    /// Dynamic buffer for object uniforms
    object_uniform_buffer: DynamicUniformBuffer<ObjectUniforms>,
    /// Bind group for object uniforms (recreated when buffer grows)
    object_bind_group: Option<wgpu::BindGroup>,
    /// Frame counter for detecting new frames
    frame_counter: Cell<u64>,
    /// Last frame we processed
    last_frame: Cell<u64>,
}

impl Default for NormalsMaterial {
    fn default() -> Self {
        Self::new()
    }
}

impl NormalsMaterial {
    /// Creates a new NormalsMaterial.
    pub fn new() -> NormalsMaterial {
        let ctxt = Context::get();

        // Create bind group layouts
        let frame_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("normals_material_frame_bind_group_layout"),
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
                label: Some("normals_material_object_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true, // Enable dynamic offsets!
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("normals_material_pipeline_layout"),
            bind_group_layouts: &[&frame_bind_group_layout, &object_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Load shader
        let shader = ctxt.create_shader_module(
            Some("normals_material_shader"),
            include_str!("normals.wgsl"),
        );

        // Vertex buffer layouts
        let vertex_buffer_layouts = [
            // Vertex positions
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
            // Normals
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }],
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
                        blend: Some(wgpu::BlendState::REPLACE),
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
            create_pipeline(Some(wgpu::Face::Back), "normals_material_pipeline_cull");
        let pipeline_no_cull = create_pipeline(None, "normals_material_pipeline_no_cull");

        // === Create shared dynamic buffer resources ===

        // Frame uniform buffer (written once per frame)
        let frame_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("normals_shared_frame_uniform_buffer"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create frame bind group
        let frame_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normals_shared_frame_bind_group"),
            layout: &frame_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_uniform_buffer.as_entire_binding(),
            }],
        });

        // Dynamic buffer for object uniforms
        let object_uniform_buffer =
            DynamicUniformBuffer::<ObjectUniforms>::new("normals_dynamic_object_uniform_buffer");

        // Create initial object bind group
        let object_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normals_dynamic_object_bind_group"),
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

        NormalsMaterial {
            pipeline_cull,
            pipeline_no_cull,
            object_bind_group_layout,
            frame_uniform_buffer,
            frame_bind_group,
            object_uniform_buffer,
            object_bind_group: Some(object_bind_group),
            frame_counter: Cell::new(0),
            last_frame: Cell::new(u64::MAX),
        }
    }
}

impl Material3d for NormalsMaterial {
    fn create_gpu_data(&self) -> Box<dyn GpuData> {
        Box::new(NormalsMaterialGpuData::new())
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
                label: Some("normals_dynamic_object_bind_group"),
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
        pass: usize,
        transform: Pose3,
        scale: Vec3,
        camera: &mut dyn Camera3d,
        _lights: &LightCollection,
        _data: &ObjectData3d,
        gpu_data: &mut dyn GpuData,
        _viewport_width: u32,
        _viewport_height: u32,
    ) {
        let ctxt = Context::get();

        // Downcast gpu_data to our specific type
        let gpu_data = gpu_data
            .as_any_mut()
            .downcast_mut::<NormalsMaterialGpuData>()
            .expect("NormalsMaterial requires NormalsMaterialGpuData");

        // Check if this is a new frame (first object being prepared)
        let current_frame = self.frame_counter.get();
        let is_new_frame = current_frame != self.last_frame.get();

        if is_new_frame {
            self.last_frame.set(current_frame);

            // Compute frame uniforms and write once per frame
            let (view, proj) = camera.view_transform_pair(pass);
            let frame_uniforms = FrameUniforms {
                view: view.to_mat4().to_cols_array_2d(),
                proj: proj.to_cols_array_2d(),
            };

            ctxt.write_buffer(
                &self.frame_uniform_buffer,
                0,
                bytemuck::bytes_of(&frame_uniforms),
            );
        }

        // Compute object uniforms
        let formatted_transform = transform.to_mat4();
        let formatted_scale = Mat3::from_diagonal(scale);

        // Pad mat3x3 to mat3x4 for proper alignment
        let scale_cols = formatted_scale.to_cols_array_2d();
        let scale_padded: [[f32; 4]; 3] = [
            [scale_cols[0][0], scale_cols[0][1], scale_cols[0][2], 0.0],
            [scale_cols[1][0], scale_cols[1][1], scale_cols[1][2], 0.0],
            [scale_cols[2][0], scale_cols[2][1], scale_cols[2][2], 0.0],
        ];

        let object_uniforms = ObjectUniforms {
            transform: formatted_transform.to_cols_array_2d(),
            scale: scale_padded,
            _padding: [0.0; 4],
        };

        // Push to dynamic buffer and store offset in gpu_data
        let object_offset = self.object_uniform_buffer.push(&object_uniforms);
        gpu_data.object_uniform_offset = Some(object_offset);
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
        _instances: &mut InstancesBuffer3d,
        gpu_data: &mut dyn GpuData,
        render_pass: &mut wgpu::RenderPass<'_>,
        _context: &RenderContext,
    ) {
        if !data.surface_rendering_active() {
            return;
        }

        // Downcast gpu_data to our specific type
        let gpu_data = gpu_data
            .as_any_mut()
            .downcast_mut::<NormalsMaterialGpuData>()
            .expect("NormalsMaterial requires NormalsMaterialGpuData");

        // Get the pre-computed object uniform offset from prepare() phase
        let object_offset = gpu_data
            .object_uniform_offset
            .expect("prepare() must be called before render()");

        // Ensure mesh buffers are on GPU
        mesh.coords().write().unwrap().load_to_gpu();
        mesh.normals().write().unwrap().load_to_gpu();
        mesh.faces().write().unwrap().load_to_gpu();

        let coords_buffer = mesh.coords().read().unwrap();
        let normals_buffer = mesh.normals().read().unwrap();
        let faces_buffer = mesh.faces().read().unwrap();

        let coords_buf = match coords_buffer.buffer() {
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

        render_pass.set_vertex_buffer(0, coords_buf.slice(..));
        render_pass.set_vertex_buffer(1, normals_buf.slice(..));
        render_pass.set_index_buffer(faces_buf.slice(..), VERTEX_INDEX_FORMAT);

        render_pass.draw_indexed(0..mesh.num_indices(), 0, 0..1);
    }
}

/// A vertex shader for coloring each point of an object depending on its normal.
pub static NORMAL_VERTEX_SRC: &str = include_str!("normals.wgsl");

/// A fragment shader for coloring each point of an object depending on its normal.
pub static NORMAL_FRAGMENT_SRC: &str = include_str!("normals.wgsl");
