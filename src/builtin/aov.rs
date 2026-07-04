//! Auxiliary render outputs (AOVs) for the rasterizer.
//!
//! In addition to the regular ("beauty") RGB image, robotics and embodied-AI
//! pipelines often need per-pixel *auxiliary* buffers describing the geometry
//! seen by the camera. This module renders three such outputs by re-using the
//! existing scene graph and camera, swapping only the material/target:
//!
//! - [`AovKind::Depth`] — linear, eye-space (metric) depth, into `R32Float`.
//! - [`AovKind::Normals`] — world- or camera-space surface normals, into
//!   `Rgba32Float` (encoded from `[-1, 1]` to `[0, 1]`).
//! - [`AovKind::Segmentation`] — the per-object integer id, into `R32Uint`.
//!
//! All targets are single-sampled (`sample_count = 1`) so the GPU→CPU read-back
//! is exact, with no MSAA resolve in the way.

use crate::camera::Camera3d;
use crate::context::Context;
use crate::resource::vertex_index::VERTEX_INDEX_FORMAT;
use crate::resource::DynamicUniformBuffer;
use crate::scene::SceneNode3d;
use bytemuck::{Pod, Zeroable};
use glamx::Mat3;

/// The texture format of the linear-depth auxiliary output.
pub const DEPTH_AOV_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// The texture format of the surface-normals auxiliary output.
pub const NORMALS_AOV_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
/// The texture format of the segmentation (object-id) auxiliary output.
pub const SEGMENTATION_AOV_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

/// Which auxiliary output a render pass produces.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AovKind {
    /// Linear eye-space depth into [`DEPTH_AOV_FORMAT`].
    Depth,
    /// World-space surface normals into [`NORMALS_AOV_FORMAT`].
    Normals,
    /// Camera-space surface normals into [`NORMALS_AOV_FORMAT`].
    CameraNormals,
    /// Per-object integer id into [`SEGMENTATION_AOV_FORMAT`].
    Segmentation,
}

impl AovKind {
    /// The texture format the auxiliary output is rendered into.
    pub fn format(self) -> wgpu::TextureFormat {
        match self {
            AovKind::Depth => DEPTH_AOV_FORMAT,
            AovKind::Normals | AovKind::CameraNormals => NORMALS_AOV_FORMAT,
            AovKind::Segmentation => SEGMENTATION_AOV_FORMAT,
        }
    }
}

/// Frame-level uniforms shared by all AOV passes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FrameUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    /// `flags.x = 1.0` selects camera-space normals; otherwise world-space.
    flags: [f32; 4],
}

/// Object-level uniforms shared by all AOV passes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ObjectUniforms {
    transform: [[f32; 4]; 4],
    scale: [[f32; 4]; 3], // mat3x3 padded to mat3x4 for alignment
    /// `extra[0]` holds the segmentation id; the rest is padding.
    extra: [u32; 4],
}

/// Renders the scene graph into auxiliary outputs (depth, normals, segmentation).
///
/// One `AovRenderer` owns the three pipelines and the shared uniform buffers;
/// it is created on first use and re-used across frames. It mirrors the
/// dynamic-uniform batching used by the regular materials: per-object uniforms
/// are accumulated into a single dynamic buffer addressed with dynamic offsets.
pub struct AovRenderer {
    pipeline_depth: wgpu::RenderPipeline,
    pipeline_normals: wgpu::RenderPipeline,
    pipeline_segmentation: wgpu::RenderPipeline,

    frame_uniform_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,

    object_bind_group_layout: wgpu::BindGroupLayout,
    object_uniform_buffer: DynamicUniformBuffer<ObjectUniforms>,
    object_bind_group: wgpu::BindGroup,

    /// GPU-only AOV visualization (raw values → display colors); created on
    /// first use of [`AovRenderer::visualize`].
    visualize: Option<AovVisualize>,
}

impl AovRenderer {
    /// Creates the AOV renderer and its three pipelines.
    pub fn new() -> AovRenderer {
        let ctxt = Context::get();

        let frame_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aov_frame_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
                label: Some("aov_object_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aov_pipeline_layout"),
            bind_group_layouts: &[
                Some(&frame_bind_group_layout),
                Some(&object_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let shader = ctxt.create_shader_module(Some("aov_shader"), include_str!("aov.wgsl"));

        let vertex_buffer_layouts = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }],
            },
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

        // All AOV passes share the same vertex stage and depth state; only the
        // fragment entry point and color-target format differ.
        let make_pipeline = |fs_entry: &str, format: wgpu::TextureFormat, label: &str| {
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
                    entry_point: Some(fs_entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    // Cull back faces: AOVs describe the closest visible surface.
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Context::depth_format(),
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };

        let pipeline_depth = make_pipeline("fs_depth", DEPTH_AOV_FORMAT, "aov_depth_pipeline");
        let pipeline_normals =
            make_pipeline("fs_normals", NORMALS_AOV_FORMAT, "aov_normals_pipeline");
        let pipeline_segmentation = make_pipeline(
            "fs_segmentation",
            SEGMENTATION_AOV_FORMAT,
            "aov_segmentation_pipeline",
        );

        let frame_uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aov_frame_uniform_buffer"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let frame_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aov_frame_bind_group"),
            layout: &frame_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_uniform_buffer.as_entire_binding(),
            }],
        });

        let object_uniform_buffer =
            DynamicUniformBuffer::<ObjectUniforms>::new("aov_object_uniform_buffer");
        let object_bind_group =
            Self::make_object_bind_group(&object_bind_group_layout, &object_uniform_buffer);

        AovRenderer {
            pipeline_depth,
            pipeline_normals,
            pipeline_segmentation,
            frame_uniform_buffer,
            frame_bind_group,
            object_bind_group_layout,
            object_uniform_buffer,
            object_bind_group,
            visualize: None,
        }
    }

    /// Renders the raw AOV in `raw_view` (of format [`AovKind::format`]) as a
    /// display-ready image into `target_view` (of format `target_format`):
    /// depth as fixed-range grayscale over `[0, depth_range]` world units
    /// (near = bright, background = black), normals as RGB, segmentation ids
    /// as distinct golden-ratio colors. Entirely on the GPU — no read-back.
    pub fn visualize_into(
        &mut self,
        kind: AovKind,
        encoder: &mut wgpu::CommandEncoder,
        raw_view: &wgpu::TextureView,
        target_view: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
        depth_range: f32,
    ) {
        if self
            .visualize
            .as_ref()
            .is_some_and(|v| v.target_format != target_format)
        {
            self.visualize = None;
        }
        self.visualize
            .get_or_insert_with(|| AovVisualize::new(target_format))
            .render(kind, encoder, raw_view, target_view, depth_range);
    }

    fn make_object_bind_group(
        layout: &wgpu::BindGroupLayout,
        buffer: &DynamicUniformBuffer<ObjectUniforms>,
    ) -> wgpu::BindGroup {
        let ctxt = Context::get();
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aov_object_bind_group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buffer.buffer(),
                    offset: 0,
                    size: std::num::NonZeroU64::new(buffer.aligned_size()),
                }),
            }],
        })
    }

    /// Renders the scene graph into the given color/depth views for one AOV.
    ///
    /// `color_view` must use the format reported by [`AovKind::format`] and a
    /// sample count of 1; `depth_view` must use [`Context::depth_format`]. Both
    /// are cleared at the start of the pass.
    pub fn render(
        &mut self,
        kind: AovKind,
        scene: &mut SceneNode3d,
        camera: &mut dyn Camera3d,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        let (view, proj) = camera.view_transform_pair(0);
        let flags = if kind == AovKind::CameraNormals {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [0.0, 0.0, 0.0, 0.0]
        };
        let frame_uniforms = FrameUniforms {
            view: view.to_mat4().to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            flags,
        };
        let ctxt = Context::get();
        ctxt.write_buffer(
            &self.frame_uniform_buffer,
            0,
            bytemuck::bytes_of(&frame_uniforms),
        );

        // Collect per-object uniforms and the matching draw list.
        self.object_uniform_buffer.clear();
        let mut draws: Vec<DrawItem> = Vec::new();
        Self::gather(scene, &mut self.object_uniform_buffer, &mut draws);

        if self.object_uniform_buffer.flush() {
            self.object_bind_group = Self::make_object_bind_group(
                &self.object_bind_group_layout,
                &self.object_uniform_buffer,
            );
        }

        // Depth and segmentation are integer/scalar; clear depth color to 0
        // (background), and the normals color likewise. The clear color is
        // ignored for the integer target but must be syntactically present.
        let clear = match kind {
            AovKind::Segmentation => wgpu::Color::TRANSPARENT,
            _ => wgpu::Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            },
        };

        let pipeline = match kind {
            AovKind::Depth => &self.pipeline_depth,
            AovKind::Normals | AovKind::CameraNormals => &self.pipeline_normals,
            AovKind::Segmentation => &self.pipeline_segmentation,
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("aov_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.frame_bind_group, &[]);

        for item in &draws {
            pass.set_bind_group(1, &self.object_bind_group, &[item.object_offset]);
            pass.set_vertex_buffer(0, item.coords.slice(..));
            pass.set_vertex_buffer(1, item.normals.slice(..));
            pass.set_index_buffer(item.faces.slice(..), VERTEX_INDEX_FORMAT);
            pass.draw_indexed(0..item.num_indices, 0, 0..1);
        }
    }

    /// Walks the scene graph, pushing per-object uniforms and collecting the
    /// GPU buffers needed to draw each visible, surface-rendered object.
    fn gather(
        scene: &mut SceneNode3d,
        objects: &mut DynamicUniformBuffer<ObjectUniforms>,
        draws: &mut Vec<DrawItem>,
    ) {
        scene.apply_to_objects_with_world_mut_recursive(&mut |transform, scale, obj| {
            if !obj.data().surface_rendering_active() {
                return;
            }

            let scale_mat = Mat3::from_diagonal(scale);
            let scale_cols = scale_mat.to_cols_array_2d();
            let scale_padded = [
                [scale_cols[0][0], scale_cols[0][1], scale_cols[0][2], 0.0],
                [scale_cols[1][0], scale_cols[1][1], scale_cols[1][2], 0.0],
                [scale_cols[2][0], scale_cols[2][1], scale_cols[2][2], 0.0],
            ];

            let uniforms = ObjectUniforms {
                transform: transform.to_mat4().to_cols_array_2d(),
                scale: scale_padded,
                extra: [obj.segmentation_id(), 0, 0, 0],
            };
            let object_offset = objects.push(&uniforms);

            // Ensure mesh buffers are resident, then snapshot the buffers.
            let mesh = obj.mesh();
            let mesh = mesh.borrow();
            mesh.coords().write().unwrap().load_to_gpu();
            mesh.normals().write().unwrap().load_to_gpu();
            mesh.faces().write().unwrap().load_to_gpu();

            let num_indices = mesh.num_indices();
            let coords = match mesh.coords().read().unwrap().buffer() {
                Some(b) => b.clone(),
                None => return,
            };
            let normals = match mesh.normals().read().unwrap().buffer() {
                Some(b) => b.clone(),
                None => return,
            };
            let faces = match mesh.faces().read().unwrap().buffer() {
                Some(b) => b.clone(),
                None => return,
            };

            draws.push(DrawItem {
                object_offset,
                coords,
                normals,
                faces,
                num_indices,
            });
        });
    }
}

impl Default for AovRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// A single queued draw for an AOV pass.
struct DrawItem {
    object_offset: u32,
    coords: wgpu::Buffer,
    normals: wgpu::Buffer,
    faces: wgpu::Buffer,
    num_indices: u32,
}

/// Uniforms of the AOV visualization pass (see `aov_visualize.wgsl`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct VisUniforms {
    /// `x`: float mode (0 = depth, 1 = normals); `y`: depth range;
    /// `z`: 1.0 when the target format is sRGB.
    params: [f32; 4],
}

/// Fullscreen pass turning a raw AOV texture into a display-ready image.
///
/// Two pipelines: one sampling the float AOV texture (depth/normals) and one
/// sampling the integer segmentation texture. Owned by [`AovRenderer`].
struct AovVisualize {
    target_format: wgpu::TextureFormat,
    layout_float: wgpu::BindGroupLayout,
    layout_seg: wgpu::BindGroupLayout,
    pipeline_float: wgpu::RenderPipeline,
    pipeline_seg: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
}

impl AovVisualize {
    fn new(target_format: wgpu::TextureFormat) -> AovVisualize {
        let ctxt = Context::get();

        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture_entry =
            |binding: u32, sample_type: wgpu::TextureSampleType| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            };

        let layout_float = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aov_visualize_float_layout"),
            entries: &[
                uniform_entry,
                texture_entry(1, wgpu::TextureSampleType::Float { filterable: false }),
            ],
        });
        let layout_seg = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aov_visualize_seg_layout"),
            entries: &[
                uniform_entry,
                texture_entry(2, wgpu::TextureSampleType::Uint),
            ],
        });

        let shader = ctxt.create_shader_module(
            Some("aov_visualize_shader"),
            include_str!("aov_visualize.wgsl"),
        );

        let make_pipeline = |layout: &wgpu::BindGroupLayout, fs_entry: &str, label: &str| {
            let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs_entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: None,
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
                multiview_mask: None,
                cache: None,
            })
        };

        let pipeline_float =
            make_pipeline(&layout_float, "fs_float", "aov_visualize_float_pipeline");
        let pipeline_seg = make_pipeline(&layout_seg, "fs_seg", "aov_visualize_seg_pipeline");

        let uniform_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aov_visualize_uniform_buffer"),
            size: std::mem::size_of::<VisUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        AovVisualize {
            target_format,
            layout_float,
            layout_seg,
            pipeline_float,
            pipeline_seg,
            uniform_buffer,
        }
    }

    fn render(
        &self,
        kind: AovKind,
        encoder: &mut wgpu::CommandEncoder,
        raw_view: &wgpu::TextureView,
        target_view: &wgpu::TextureView,
        depth_range: f32,
    ) {
        let ctxt = Context::get();
        let float_mode = match kind {
            AovKind::Depth => 0.0,
            _ => 1.0,
        };
        let is_srgb = if self.target_format.is_srgb() {
            1.0
        } else {
            0.0
        };
        ctxt.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&VisUniforms {
                params: [float_mode, depth_range, is_srgb, 0.0],
            }),
        );

        let seg = kind == AovKind::Segmentation;
        let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aov_visualize_bind_group"),
            layout: if seg {
                &self.layout_seg
            } else {
                &self.layout_float
            },
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: if seg { 2 } else { 1 },
                    resource: wgpu::BindingResource::TextureView(raw_view),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("aov_visualize_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(if seg {
            &self.pipeline_seg
        } else {
            &self.pipeline_float
        });
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
