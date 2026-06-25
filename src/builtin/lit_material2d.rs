//! Dynamically-lit 2D material with optional normal mapping.
//!
//! [`LitMaterial2d`] shades a sprite with the 2D lights from the global
//! [`Light2dManager`](crate::light2d::Light2dManager): an ambient term plus per-light
//! diffuse and specular. With a normal map ([`ObjectData2d::set_normal_map`]) the
//! shading is per-pixel; without one the sprite is flat and still picks up each
//! light's radial falloff. Per-object shading knobs travel in [`LitParams`].

use crate::camera::Camera2d;
use crate::context::Context;
use crate::light2d::{Light2dKind, Light2dManager, MAX_LIGHTS_2D};
use crate::resource::vertex_index::VERTEX_INDEX_FORMAT;
use crate::resource::{
    multisample_state, GpuData, GpuMesh2d, Material2d, MaterialManager2d, PipelineCache,
    RenderContext2d, Texture,
};
use crate::scene::{InstancesBuffer2d, ObjectData2d};
use bytemuck::{Pod, Zeroable};
use glamx::{Mat2, Mat3, Pose2, Vec2};
use std::any::Any;
use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

/// Per-object shading parameters for [`LitMaterial2d`], stored in
/// [`ObjectData2d::lit_params`].
#[derive(Copy, Clone, Debug)]
pub struct LitParams {
    /// Specular highlight strength (0 disables specular).
    pub specular_strength: f32,
    /// Specular exponent (higher = tighter highlight).
    pub shininess: f32,
    /// Scales the normal map's tangent-space XY, controlling bump strength.
    pub normal_strength: f32,
}

impl Default for LitParams {
    fn default() -> Self {
        LitParams {
            specular_strength: 0.0,
            shininess: 16.0,
            normal_strength: 1.0,
        }
    }
}

impl LitParams {
    /// New parameters with the given specular strength and exponent.
    pub fn with_specular(mut self, strength: f32, shininess: f32) -> Self {
        self.specular_strength = strength;
        self.shininess = shininess;
        self
    }

    /// Sets the normal-map bump strength.
    pub fn with_normal_strength(mut self, strength: f32) -> Self {
        self.normal_strength = strength;
        self
    }
}

/// One GPU light, matching `Light` in `lit2d.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuLight {
    pos_height: [f32; 4],
    color_intensity: [f32; 4],
    dir_cone: [f32; 4],
    radius: [f32; 4],
}

/// Frame uniforms matching `FrameUniforms` in `lit2d.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FrameUniforms {
    view: [[f32; 4]; 3],
    proj: [[f32; 4]; 3],
    ambient_count: [f32; 4],
    lights: [GpuLight; MAX_LIGHTS_2D],
}

/// Per-object uniforms matching `ObjectUniforms` in `lit2d.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ObjectUniforms {
    model: [[f32; 4]; 3],
    scale: [[f32; 4]; 2],
    color: [f32; 4],
    params: [f32; 4],
}

/// Per-object GPU data for [`LitMaterial2d`].
pub struct LitMaterial2dGpuData {
    object_uniform_buffer: wgpu::Buffer,
    object_bind_group: Option<wgpu::BindGroup>,
    texture_bind_group: Option<wgpu::BindGroup>,
    cached_albedo_ptr: usize,
    cached_normal_ptr: usize,
}

impl LitMaterial2dGpuData {
    fn new() -> Self {
        let ctxt = Context::get();
        let object_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lit2d_object_uniform_buffer"),
            size: std::mem::size_of::<ObjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        LitMaterial2dGpuData {
            object_uniform_buffer,
            object_bind_group: None,
            texture_bind_group: None,
            cached_albedo_ptr: 0,
            cached_normal_ptr: 0,
        }
    }
}

impl GpuData for LitMaterial2dGpuData {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A material that shades 2D objects with the global dynamic 2D lights (see the
/// [module docs](self)).
pub struct LitMaterial2d {
    pipeline: PipelineCache,
    object_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    frame_uniform_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    default_normal_map: Arc<Texture>,
    frame_counter: Cell<u64>,
    last_frame: Cell<u64>,
}

impl Default for LitMaterial2d {
    fn default() -> Self {
        Self::new()
    }
}

impl LitMaterial2d {
    /// The name the lit material is registered under in the global 2D material manager.
    pub const NAME: &'static str = "lit2d";

    /// Creates a new lit material.
    pub fn new() -> LitMaterial2d {
        let ctxt = Context::get();

        let frame_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lit2d_frame_bind_group_layout"),
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
                label: Some("lit2d_object_bind_group_layout"),
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

        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let samp_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let texture_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lit2d_texture_bind_group_layout"),
                entries: &[tex_entry(0), samp_entry(1), tex_entry(2), samp_entry(3)],
            });

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lit2d_pipeline_layout"),
            bind_group_layouts: &[
                Some(&frame_bind_group_layout),
                Some(&object_bind_group_layout),
                Some(&texture_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let shader = ctxt.create_shader_module(
            Some("lit2d_shader"),
            &crate::builtin::compile_shader_with_common("package::lit2d", include_str!("lit2d.wgsl")),
        );

        let pipeline = PipelineCache::new(move |sample_count| {
            let ctxt = Context::get();
            let vertex_buffer_layouts = [
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
                    array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x2,
                    }],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32x4,
                    }],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 5,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                },
            ];

            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("lit2d_pipeline"),
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
            label: Some("lit2d_frame_uniform_buffer"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lit2d_frame_bind_group"),
            layout: &frame_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_uniform_buffer.as_entire_binding(),
            }],
        });

        LitMaterial2d {
            pipeline,
            object_bind_group_layout,
            texture_bind_group_layout,
            frame_uniform_buffer,
            frame_bind_group,
            default_normal_map: Texture::new_default_normal_map(),
            frame_counter: Cell::new(0),
            last_frame: Cell::new(u64::MAX),
        }
    }

    /// Returns the shared lit material from the global manager, registering it on
    /// first use. Used by [`SceneNode2d::lit_sprite`](crate::scene::SceneNode2d::lit_sprite).
    pub fn shared() -> Rc<std::cell::RefCell<Box<dyn Material2d + 'static>>> {
        MaterialManager2d::get_global_manager(|mm| {
            if let Some(mat) = mm.get(Self::NAME) {
                mat
            } else {
                let mat: Rc<std::cell::RefCell<Box<dyn Material2d + 'static>>> = Rc::new(
                    std::cell::RefCell::new(Box::new(LitMaterial2d::new()) as Box<dyn Material2d>),
                );
                mm.add(mat.clone(), Self::NAME);
                mat
            }
        })
    }

    fn mat3_to_padded(m: &Mat3) -> [[f32; 4]; 3] {
        let c = m.to_cols_array_2d();
        [
            [c[0][0], c[0][1], c[0][2], 0.0],
            [c[1][0], c[1][1], c[1][2], 0.0],
            [c[2][0], c[2][1], c[2][2], 0.0],
        ]
    }

    fn mat2_to_padded(m: &Mat2) -> [[f32; 4]; 2] {
        let c = m.to_cols_array_2d();
        [[c[0][0], c[0][1], 0.0, 0.0], [c[1][0], c[1][1], 0.0, 0.0]]
    }

    fn create_texture_bind_group(
        &self,
        albedo: &Texture,
        normal: &Texture,
    ) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lit2d_texture_bind_group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&albedo.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&albedo.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&normal.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&normal.sampler),
                },
            ],
        })
    }
}

impl Material2d for LitMaterial2d {
    fn create_gpu_data(&self) -> Box<dyn GpuData> {
        Box::new(LitMaterial2dGpuData::new())
    }

    fn begin_frame(&mut self) {
        self.frame_counter
            .set(self.frame_counter.get().wrapping_add(1));
    }

    fn prepare(
        &mut self,
        transform: Pose2,
        scale: Vec2,
        camera: &mut dyn Camera2d,
        data: &ObjectData2d,
        _mesh: &mut GpuMesh2d,
        _instances: &mut InstancesBuffer2d,
        gpu_data: &mut dyn GpuData,
        _context: &RenderContext2d,
    ) {
        let ctxt = Context::get();
        let gpu_data = gpu_data
            .as_any_mut()
            .downcast_mut::<LitMaterial2dGpuData>()
            .expect("LitMaterial2d requires LitMaterial2dGpuData");

        // Frame uniforms (view/proj + lights) once per frame, pulled from the global
        // 2D light manager.
        let current_frame = self.frame_counter.get();
        if current_frame != self.last_frame.get() {
            self.last_frame.set(current_frame);
            let (view, proj) = camera.view_transform_pair();
            let mut frame = FrameUniforms {
                view: Self::mat3_to_padded(&view),
                proj: Self::mat3_to_padded(&proj),
                ambient_count: [0.0; 4],
                lights: [GpuLight::zeroed(); MAX_LIGHTS_2D],
            };
            Light2dManager::get_global_manager(|mm| {
                let amb = mm.ambient();
                let lights = mm.lights();
                let n = lights.len().min(MAX_LIGHTS_2D);
                frame.ambient_count = [amb.r, amb.g, amb.b, n as f32];
                for (slot, light) in frame.lights.iter_mut().zip(lights.iter()) {
                    let kind = match light.kind {
                        Light2dKind::Point => 0.0,
                        Light2dKind::Spot => 1.0,
                    };
                    let dir = light.direction.normalize_or_zero();
                    *slot = GpuLight {
                        pos_height: [light.position.x, light.position.y, light.height, kind],
                        color_intensity: [
                            light.color.r,
                            light.color.g,
                            light.color.b,
                            light.intensity,
                        ],
                        dir_cone: [
                            dir.x,
                            dir.y,
                            light.inner_angle.cos(),
                            light.outer_angle.cos(),
                        ],
                        radius: [light.radius, 0.0, 0.0, 0.0],
                    };
                }
            });
            ctxt.write_buffer(&self.frame_uniform_buffer, 0, bytemuck::bytes_of(&frame));
        }

        // Per-object uniforms.
        let params = data.lit_params().unwrap_or_default();
        let has_normal = data.normal_map().is_some();
        let color = data.color();
        let uniforms = ObjectUniforms {
            model: Self::mat3_to_padded(&transform.to_mat3()),
            scale: Self::mat2_to_padded(&Mat2::from_diagonal(scale)),
            color: [color.r, color.g, color.b, color.a],
            params: [
                params.specular_strength,
                params.shininess,
                params.normal_strength,
                if has_normal { 1.0 } else { 0.0 },
            ],
        };
        ctxt.write_buffer(
            &gpu_data.object_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        if gpu_data.object_bind_group.is_none() {
            gpu_data.object_bind_group = Some(ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lit2d_object_bind_group"),
                layout: &self.object_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_data.object_uniform_buffer.as_entire_binding(),
                }],
            }));
        }

        // (Re)build the texture bind group when the albedo or normal map changes.
        let albedo = data.texture();
        let normal = data.normal_map().unwrap_or(&self.default_normal_map);
        let albedo_ptr = Arc::as_ptr(albedo) as usize;
        let normal_ptr = Arc::as_ptr(normal) as usize;
        if gpu_data.texture_bind_group.is_none()
            || gpu_data.cached_albedo_ptr != albedo_ptr
            || gpu_data.cached_normal_ptr != normal_ptr
        {
            gpu_data.texture_bind_group = Some(self.create_texture_bind_group(albedo, normal));
            gpu_data.cached_albedo_ptr = albedo_ptr;
            gpu_data.cached_normal_ptr = normal_ptr;
        }
    }

    fn render(
        &mut self,
        _transform: Pose2,
        _scale: Vec2,
        _camera: &mut dyn Camera2d,
        _data: &ObjectData2d,
        mesh: &mut GpuMesh2d,
        instances: &mut InstancesBuffer2d,
        gpu_data: &mut dyn GpuData,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext2d,
    ) {
        let gpu_data = gpu_data
            .as_any_mut()
            .downcast_mut::<LitMaterial2dGpuData>()
            .expect("LitMaterial2d requires LitMaterial2dGpuData");

        let num_instances = instances.len();
        instances.positions.load_to_gpu();
        instances.colors.load_to_gpu();
        instances.deformations.load_to_gpu();
        mesh.load_to_gpu();

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
        render_pass.set_vertex_buffer(0, coords_buf.slice(..));
        render_pass.set_vertex_buffer(1, uvs_buf.slice(..));
        render_pass.set_vertex_buffer(2, inst_positions_buf.slice(..));
        render_pass.set_vertex_buffer(3, inst_colors_buf.slice(..));
        render_pass.set_vertex_buffer(4, inst_deformations_buf.slice(..));
        render_pass.set_index_buffer(faces_buf.slice(..), VERTEX_INDEX_FORMAT);
        render_pass.draw_indexed(0..mesh.num_indices(), 0, 0..num_instances as u32);
    }
}
