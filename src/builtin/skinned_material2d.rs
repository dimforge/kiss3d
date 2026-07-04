//! 2D skeletal mesh deformation with GPU skinning ([`SkinnedMesh2d`]).
//!
//! A skinned mesh binds each vertex to up to four bones of a [`Bone2d`] skeleton with
//! blend weights. Posing a bone re-deforms the mesh on the GPU (Spine / DragonBones
//! style): [`update`](SkinnedMesh2d::update) walks the bone hierarchy, multiplies each
//! bone's world transform by its inverse bind pose, and uploads the resulting joint
//! matrices, which the vertex shader blends per vertex.
//!
//! The mesh exposes a [`SceneNode2d`] to add to the scene; set its texture with the
//! node's `set_texture_*` methods.

use crate::camera::Camera2d;
use crate::context::Context;
use crate::resource::vertex_index::VERTEX_INDEX_FORMAT;
use crate::resource::{
    multisample_state, GpuData, GpuMesh2d, Material2d, PipelineCache, RenderContext2d,
    TextureManager,
};
use crate::scene::{InstancesBuffer2d, Object2d, ObjectData2d, SceneNode2d};
use bytemuck::{Pod, Zeroable};
use glamx::{Mat3, Pose2, Vec2};
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

/// Maximum number of bones in a [`SkinnedMesh2d`] skeleton (matches `MAX_JOINTS` in
/// `skinned2d.wgsl`).
pub const MAX_JOINTS_2D: usize = 32;

/// One vertex of a skinned 2D mesh: position, UV, and up to four bone influences.
#[derive(Copy, Clone, Debug)]
pub struct SkinVertex2d {
    /// Rest-pose position.
    pub position: Vec2,
    /// Texture coordinate.
    pub uv: Vec2,
    /// Indices of the (up to four) influencing bones.
    pub joints: [u32; 4],
    /// Blend weights for `joints` (should sum to 1).
    pub weights: [f32; 4],
}

/// A bone in a [`SkinnedMesh2d`] skeleton.
#[derive(Copy, Clone, Debug)]
pub struct Bone2d {
    /// Parent bone index, or `None` for a root. A bone's parent must come before it.
    pub parent: Option<usize>,
    /// Local transform relative to the parent.
    pub local: Pose2,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FrameUniforms {
    view: [[f32; 4]; 3],
    proj: [[f32; 4]; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ObjectUniforms {
    model: [[f32; 4]; 3],
    color: [f32; 4],
    joints: [[f32; 4]; MAX_JOINTS_2D * 3],
}

fn mat3_to_padded(m: &Mat3) -> [[f32; 4]; 3] {
    let c = m.to_cols_array_2d();
    [
        [c[0][0], c[0][1], c[0][2], 0.0],
        [c[1][0], c[1][1], c[1][2], 0.0],
        [c[2][0], c[2][1], c[2][2], 0.0],
    ]
}

/// A 2D mesh deformed by a bone skeleton via GPU skinning (see the [module docs](crate::builtin)).
pub struct SkinnedMesh2d {
    bones: Vec<Bone2d>,
    inverse_bind: Vec<Mat3>,
    transform: Pose2,
    color: [f32; 4],
    object_uniform_buffer: wgpu::Buffer,
    node: SceneNode2d,
}

impl SkinnedMesh2d {
    /// Builds a skinned mesh from `vertices`, triangle `faces`, and a `bones`
    /// skeleton. The bones' initial `local` transforms define the bind pose (used to
    /// compute the inverse-bind matrices), so pose the skeleton into its rest shape
    /// before calling this. Bones must be ordered parents-before-children.
    pub fn new(
        vertices: Vec<SkinVertex2d>,
        faces: Vec<[u32; 3]>,
        bones: Vec<Bone2d>,
    ) -> SkinnedMesh2d {
        assert!(
            bones.len() <= MAX_JOINTS_2D,
            "SkinnedMesh2d supports at most {} bones",
            MAX_JOINTS_2D
        );
        let ctxt = Context::get();

        // Inverse bind matrices from the initial (bind) pose.
        let bind_world = Self::world_matrices(&bones);
        let inverse_bind = bind_world.iter().map(|m| m.inverse()).collect();

        // Split vertices into the per-attribute GPU buffers.
        let positions: Vec<[f32; 2]> = vertices.iter().map(|v| v.position.into()).collect();
        let uvs: Vec<[f32; 2]> = vertices.iter().map(|v| v.uv.into()).collect();
        let joints: Vec<[u32; 4]> = vertices.iter().map(|v| v.joints).collect();
        let weights: Vec<[f32; 4]> = vertices.iter().map(|v| v.weights).collect();
        let indices: Vec<u32> = faces.iter().flat_map(|f| f.iter().copied()).collect();
        let num_indices = indices.len() as u32;

        let pos_buf = ctxt.create_buffer_init(
            Some("skinned2d_pos"),
            bytemuck::cast_slice(&positions),
            wgpu::BufferUsages::VERTEX,
        );
        let uv_buf = ctxt.create_buffer_init(
            Some("skinned2d_uv"),
            bytemuck::cast_slice(&uvs),
            wgpu::BufferUsages::VERTEX,
        );
        let joint_buf = ctxt.create_buffer_init(
            Some("skinned2d_joints"),
            bytemuck::cast_slice(&joints),
            wgpu::BufferUsages::VERTEX,
        );
        let weight_buf = ctxt.create_buffer_init(
            Some("skinned2d_weights"),
            bytemuck::cast_slice(&weights),
            wgpu::BufferUsages::VERTEX,
        );
        let index_buf = ctxt.create_buffer_init(
            Some("skinned2d_indices"),
            bytemuck::cast_slice(&indices),
            wgpu::BufferUsages::INDEX,
        );

        let object_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned2d_object_uniform"),
            size: std::mem::size_of::<ObjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let material = SkinnedMaterial2d::new(SkinnedBuffers {
            pos: pos_buf,
            uv: uv_buf,
            joints: joint_buf,
            weights: weight_buf,
            index: index_buf,
            num_indices,
            object_uniform: object_uniform_buffer.clone(),
        });
        let material: Rc<RefCell<Box<dyn Material2d + 'static>>> =
            Rc::new(RefCell::new(Box::new(material)));

        let dummy_mesh = Rc::new(RefCell::new(GpuMesh2d::new(
            vec![Vec2::ZERO, Vec2::ZERO, Vec2::ZERO],
            vec![[0, 0, 0]],
            None,
            false,
        )));
        let tex = TextureManager::get_global_manager(|tm| tm.get_default());
        let object = Object2d::new(dummy_mesh, 1.0, 1.0, 1.0, tex, material);
        let node = SceneNode2d::new(Vec2::ONE, Pose2::IDENTITY, Some(object));

        let mut mesh = SkinnedMesh2d {
            bones,
            inverse_bind,
            transform: Pose2::IDENTITY,
            color: [1.0; 4],
            object_uniform_buffer,
            node,
        };
        mesh.update();
        mesh
    }

    /// World-space matrix of each bone (walking parents; assumes parents precede children).
    fn world_matrices(bones: &[Bone2d]) -> Vec<Mat3> {
        let mut world = Vec::with_capacity(bones.len());
        for bone in bones {
            let local = bone.local.to_mat3();
            let m = match bone.parent {
                Some(p) => world[p] * local,
                None => local,
            };
            world.push(m);
        }
        world
    }

    /// The scene node rendering this mesh. Add it to the scene and set its texture.
    pub fn node(&self) -> SceneNode2d {
        self.node.clone()
    }

    /// Sets a bone's local transform (relative to its parent), e.g. to animate it.
    /// Call [`update`](Self::update) afterwards to apply.
    pub fn set_bone_local(&mut self, bone: usize, local: Pose2) {
        if bone < self.bones.len() {
            self.bones[bone].local = local;
        }
    }

    /// Returns a bone's current local transform.
    pub fn bone_local(&self, bone: usize) -> Pose2 {
        self.bones[bone].local
    }

    /// Number of bones in the skeleton.
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Sets the whole-mesh model transform (position/rotation/scale of the skeleton).
    pub fn set_transform(&mut self, transform: Pose2) {
        self.transform = transform;
    }

    /// Sets a uniform color/tint multiplied into the mesh's texture.
    pub fn set_color(&mut self, color: [f32; 4]) {
        self.color = color;
    }

    /// Recomputes joint matrices from the current bone poses and uploads them.
    /// Call once per frame after posing bones, before rendering.
    pub fn update(&mut self) {
        let ctxt = Context::get();
        let world = Self::world_matrices(&self.bones);

        let mut joints = [[0.0f32; 4]; MAX_JOINTS_2D * 3];
        for (i, (w, inv)) in world.iter().zip(self.inverse_bind.iter()).enumerate() {
            let joint = *w * *inv;
            let padded = mat3_to_padded(&joint);
            joints[i * 3] = padded[0];
            joints[i * 3 + 1] = padded[1];
            joints[i * 3 + 2] = padded[2];
        }
        // Bones beyond the skeleton stay identity (so zero-weight slots are harmless).
        for i in world.len()..MAX_JOINTS_2D {
            joints[i * 3] = [1.0, 0.0, 0.0, 0.0];
            joints[i * 3 + 1] = [0.0, 1.0, 0.0, 0.0];
            joints[i * 3 + 2] = [0.0, 0.0, 1.0, 0.0];
        }

        let uniforms = ObjectUniforms {
            model: mat3_to_padded(&self.transform.to_mat3()),
            color: self.color,
            joints,
        };
        ctxt.write_buffer(
            &self.object_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }
}

/// Buffers handed to the skinned material (all cheap-to-clone wgpu handles).
struct SkinnedBuffers {
    pos: wgpu::Buffer,
    uv: wgpu::Buffer,
    joints: wgpu::Buffer,
    weights: wgpu::Buffer,
    index: wgpu::Buffer,
    num_indices: u32,
    object_uniform: wgpu::Buffer,
}

struct SkinnedMaterial2dGpuData {
    object_bind_group: Option<wgpu::BindGroup>,
    texture_bind_group: Option<wgpu::BindGroup>,
    cached_texture_ptr: usize,
}

impl GpuData for SkinnedMaterial2dGpuData {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

struct SkinnedMaterial2d {
    pipeline: PipelineCache,
    frame_bind_group: wgpu::BindGroup,
    frame_uniform_buffer: wgpu::Buffer,
    object_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    buffers: SkinnedBuffers,
}

impl SkinnedMaterial2d {
    fn new(buffers: SkinnedBuffers) -> SkinnedMaterial2d {
        let ctxt = Context::get();

        let frame_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("skinned2d_frame_layout"),
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
                label: Some("skinned2d_object_layout"),
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
                label: Some("skinned2d_texture_layout"),
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
            label: Some("skinned2d_pipeline_layout"),
            bind_group_layouts: &[
                Some(&frame_bind_group_layout),
                Some(&object_bind_group_layout),
                Some(&texture_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let shader = ctxt.create_shader_module(
            Some("skinned2d_shader"),
            &crate::builtin::compile_shader_with_common(
                "package::skinned2d",
                include_str!("skinned2d.wgsl"),
            ),
        );

        let pipeline = PipelineCache::new(move |sample_count| {
            let ctxt = Context::get();
            let vertex_layouts = [
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    }],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x2,
                    }],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[u32; 4]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Uint32x4,
                    }],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32x4,
                    }],
                },
            ];

            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("skinned2d_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_layouts,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: Context::render_format(),
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
                multisample: multisample_state(sample_count),
                multiview_mask: None,
                cache: None,
            })
        });

        let frame_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned2d_frame_uniform"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinned2d_frame_bind_group"),
            layout: &frame_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_uniform_buffer.as_entire_binding(),
            }],
        });

        SkinnedMaterial2d {
            pipeline,
            frame_bind_group,
            frame_uniform_buffer,
            object_bind_group_layout,
            texture_bind_group_layout,
            buffers,
        }
    }
}

impl Material2d for SkinnedMaterial2d {
    fn create_gpu_data(&self) -> Box<dyn GpuData> {
        Box::new(SkinnedMaterial2dGpuData {
            object_bind_group: None,
            texture_bind_group: None,
            cached_texture_ptr: 0,
        })
    }

    fn prepare(
        &mut self,
        _transform: Pose2,
        _scale: Vec2,
        camera: &mut dyn Camera2d,
        data: &ObjectData2d,
        _mesh: &mut GpuMesh2d,
        _instances: &mut InstancesBuffer2d,
        gpu_data: &mut dyn GpuData,
        _context: &RenderContext2d,
    ) {
        let ctxt = Context::get();
        let (view, proj) = camera.view_transform_pair();
        let frame = FrameUniforms {
            view: mat3_to_padded(&view),
            proj: mat3_to_padded(&proj),
        };
        ctxt.write_buffer(&self.frame_uniform_buffer, 0, bytemuck::bytes_of(&frame));

        let gpu_data = gpu_data
            .as_any_mut()
            .downcast_mut::<SkinnedMaterial2dGpuData>()
            .expect("SkinnedMaterial2d requires SkinnedMaterial2dGpuData");

        if gpu_data.object_bind_group.is_none() {
            gpu_data.object_bind_group = Some(ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("skinned2d_object_bind_group"),
                layout: &self.object_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.object_uniform.as_entire_binding(),
                }],
            }));
        }

        let texture = data.texture();
        let texture_ptr = std::sync::Arc::as_ptr(texture) as usize;
        if gpu_data.texture_bind_group.is_none() || gpu_data.cached_texture_ptr != texture_ptr {
            gpu_data.texture_bind_group =
                Some(ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("skinned2d_texture_bind_group"),
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
                }));
            gpu_data.cached_texture_ptr = texture_ptr;
        }
    }

    fn render(
        &mut self,
        _transform: Pose2,
        _scale: Vec2,
        _camera: &mut dyn Camera2d,
        _data: &ObjectData2d,
        _mesh: &mut GpuMesh2d,
        _instances: &mut InstancesBuffer2d,
        gpu_data: &mut dyn GpuData,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext2d,
    ) {
        let gpu_data = gpu_data
            .as_any_mut()
            .downcast_mut::<SkinnedMaterial2dGpuData>()
            .expect("SkinnedMaterial2d requires SkinnedMaterial2dGpuData");
        let object_bind_group = match gpu_data.object_bind_group.as_ref() {
            Some(bg) => bg,
            None => return,
        };
        let texture_bind_group = match gpu_data.texture_bind_group.as_ref() {
            Some(bg) => bg,
            None => return,
        };

        let pipeline = self.pipeline.get(context.sample_count);
        render_pass.set_pipeline(&pipeline);
        render_pass.set_bind_group(0, &self.frame_bind_group, &[]);
        render_pass.set_bind_group(1, object_bind_group, &[]);
        render_pass.set_bind_group(2, texture_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.buffers.pos.slice(..));
        render_pass.set_vertex_buffer(1, self.buffers.uv.slice(..));
        render_pass.set_vertex_buffer(2, self.buffers.joints.slice(..));
        render_pass.set_vertex_buffer(3, self.buffers.weights.slice(..));
        render_pass.set_index_buffer(self.buffers.index.slice(..), VERTEX_INDEX_FORMAT);
        render_pass.draw_indexed(0..self.buffers.num_indices, 0, 0..1);
    }
}
