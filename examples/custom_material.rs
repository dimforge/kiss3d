extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::camera::Camera;
use kiss3d::context::Context;
use kiss3d::light::Light;
use kiss3d::resource::vertex_index::VERTEX_INDEX_FORMAT;
use kiss3d::resource::{GpuData, GpuMesh, Material, RenderContext};
use kiss3d::scene::{InstancesBuffer, ObjectData};
use kiss3d::window::Window;
use na::{Isometry3, Matrix3, Translation3, UnitQuaternion, Vector3};
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: custom_material").await;
    let mut c = window.add_sphere(1.0);
    let material = Rc::new(RefCell::new(
        Box::new(NormalMaterial::new()) as Box<dyn Material + 'static>,
    ));

    c.set_material(material);
    c.append_translation(&Translation3::new(0.0, 0.0, 2.0));

    let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);

    while window.render().await {
        c.prepend_to_local_rotation(&rot);
    }
}

/// Frame-level uniforms (view, projection).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
}

/// Object-level uniforms (transform, scale).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectUniforms {
    transform: [[f32; 4]; 4],
    scale: [[f32; 4]; 3], // mat3x3 padded to mat3x4 for alignment
    _padding: [f32; 4],
}

/// Per-object GPU data for NormalMaterial.
pub struct NormalMaterialGpuData {
    frame_uniform_buffer: wgpu::Buffer,
    object_uniform_buffer: wgpu::Buffer,
}

impl NormalMaterialGpuData {
    pub fn new() -> Self {
        let ctxt = Context::get();

        let frame_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("custom_material_frame_uniform_buffer"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let object_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("custom_material_object_uniform_buffer"),
            size: std::mem::size_of::<ObjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            frame_uniform_buffer,
            object_uniform_buffer,
        }
    }
}

impl GpuData for NormalMaterialGpuData {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A material that draws normals of an object.
pub struct NormalMaterial {
    pipeline: wgpu::RenderPipeline,
    frame_bind_group_layout: wgpu::BindGroupLayout,
    object_bind_group_layout: wgpu::BindGroupLayout,
}

impl NormalMaterial {
    pub fn new() -> NormalMaterial {
        let ctxt = Context::get();

        // Create bind group layouts
        let frame_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("custom_material_frame_bind_group_layout"),
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
                label: Some("custom_material_object_bind_group_layout"),
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
            label: Some("custom_material_pipeline_layout"),
            bind_group_layouts: &[&frame_bind_group_layout, &object_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create shader module from WGSL source
        let shader = ctxt.create_shader_module(Some("custom_material_shader"), NORMAL_SHADER_SRC);

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

        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("custom_material_pipeline"),
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
                cull_mode: Some(wgpu::Face::Back),
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

        NormalMaterial {
            pipeline,
            frame_bind_group_layout,
            object_bind_group_layout,
        }
    }

    fn create_frame_bind_group(&self, buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("custom_material_frame_bind_group"),
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
            label: Some("custom_material_object_bind_group"),
            layout: &self.object_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }
}

impl Material for NormalMaterial {
    fn create_gpu_data(&self) -> Box<dyn GpuData> {
        Box::new(NormalMaterialGpuData::new())
    }

    fn render(
        &mut self,
        pass: usize,
        transform: &Isometry3<f32>,
        scale: &Vector3<f32>,
        camera: &mut dyn Camera,
        _: &Light,
        data: &ObjectData,
        mesh: &mut GpuMesh,
        _instances: &mut InstancesBuffer,
        gpu_data: &mut dyn GpuData,
        context: &mut RenderContext,
    ) {
        let ctxt = Context::get();

        if !data.surface_rendering_active() {
            return;
        }

        // Downcast gpu_data to our specific type
        let gpu_data = gpu_data
            .as_any_mut()
            .downcast_mut::<NormalMaterialGpuData>()
            .expect("NormalMaterial requires NormalMaterialGpuData");

        // Update frame uniforms
        let (view, proj) = camera.view_transform_pair(pass);

        let frame_uniforms = FrameUniforms {
            view: view.to_homogeneous().into(),
            proj: proj.into(),
        };
        ctxt.write_buffer(
            &gpu_data.frame_uniform_buffer,
            0,
            bytemuck::bytes_of(&frame_uniforms),
        );

        // Update object uniforms
        let formatted_transform = transform.to_homogeneous();
        let formatted_scale = Matrix3::from_diagonal(&Vector3::new(scale.x, scale.y, scale.z));

        // Pad mat3x3 to mat3x4 for proper alignment
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
            scale: scale_padded,
            _padding: [0.0; 4],
        };
        ctxt.write_buffer(
            &gpu_data.object_uniform_buffer,
            0,
            bytemuck::bytes_of(&object_uniforms),
        );

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

        // Create bind groups with per-object buffers
        let frame_bind_group = self.create_frame_bind_group(&gpu_data.frame_uniform_buffer);
        let object_bind_group = self.create_object_bind_group(&gpu_data.object_uniform_buffer);

        // Create render pass
        {
            let mut render_pass = context
                .encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("custom_material_render_pass"),
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

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &frame_bind_group, &[]);
            render_pass.set_bind_group(1, &object_bind_group, &[]);

            render_pass.set_vertex_buffer(0, coords_buf.slice(..));
            render_pass.set_vertex_buffer(1, normals_buf.slice(..));
            render_pass.set_index_buffer(faces_buf.slice(..), VERTEX_INDEX_FORMAT);

            render_pass.draw_indexed(0..mesh.num_indices(), 0, 0..1);
        }
    }
}

// WGSL shader that colors each point based on its normal
static NORMAL_SHADER_SRC: &str = "
// Bind group 0: Frame uniforms
struct FrameUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

// Bind group 1: Object uniforms
struct ObjectUniforms {
    transform: mat4x4<f32>,
    scale: mat3x3<f32>,
}

@group(1) @binding(0)
var<uniform> object: ObjectUniforms;

// Vertex input
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

// Vertex output / Fragment input
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ls_normal: vec3<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let scaled_pos = object.scale * vertex.position;
    out.clip_position = frame.proj * frame.view * object.transform * vec4<f32>(scaled_pos, 1.0);
    out.ls_normal = vertex.normal;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Map normal from [-1, 1] to [0, 1] for visualization
    let color = (in.ls_normal + 1.0) / 2.0;
    return vec4<f32>(color, 1.0);
}
";
