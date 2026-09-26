//! A 2D material that samples the frame so far.
//!
//! A material whose `reads_screen` is true splits the 2D pass: the frame drawn
//! up to that object is copied into [`RenderContext2d::screen`], and the object
//! draws with that copy bound. This one refracts what is behind it, the way a
//! rippled pane of glass would.
//!
//! The copy is remade whenever the film resizes, and `screen_generation` says
//! when: a bind group built over the old texture has to be rebuilt.

use kiss3d::prelude::vertex_index::VERTEX_INDEX_FORMAT;
use kiss3d::prelude::*;
use kiss3d::resource::RenderContext2d;
use std::any::Any;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: screen_read2d").await;
    window.set_background_color(Color::new(0.02, 0.03, 0.08, 1.0));
    let mut camera = FixedView2d::default();
    let mut scene = SceneNode2d::empty();

    // The scene behind the glass: a row of rectangles the ripple has something
    // to bend.
    let colors = [RED, ORANGE, YELLOW, GREEN, CYAN, BLUE, MAGENTA];
    for (i, color) in colors.iter().enumerate() {
        let x = (i as f32 - 3.0) * 90.0;
        scene
            .add_rectangle(70.0, 320.0)
            .set_color(*color)
            .set_position(Vec2::new(x, 0.0));
    }

    // The pane itself. Its material reads the screen, so the pass is split
    // before it and everything above was already drawn into the copy.
    let glass = Rc::new(RefCell::new(
        Box::new(RippleMaterial2d::new()) as Box<dyn Material2d + 'static>
    ));
    let mut pane = scene.add_rectangle(420.0, 220.0).set_material(glass);
    pane.set_position(Vec2::new(0.0, 0.0));

    while window.render_2d(&mut scene, &mut camera).await {}
}

/// Everything the shader needs, in one uniform block. `mat3x3` is stored as
/// three `vec4` columns, the alignment WGSL asks for.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RippleUniforms {
    view: [[f32; 4]; 3],
    proj: [[f32; 4]; 3],
    model: [[f32; 4]; 3],
    scale: [[f32; 4]; 2],
    /// `(viewport width, viewport height, time, strength)`.
    params: [f32; 4],
}

/// This material keeps no per-object state, but the trait still hands each
/// object a box of its own.
struct NoGpuData;

impl GpuData for NoGpuData {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Refracts the frame behind it through a moving ripple.
struct RippleMaterial2d {
    pipeline: PipelineCache,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// Built over `RenderContext2d::screen`, and rebuilt when the generation
    /// beside it says that texture is a new one.
    bind_group: Option<wgpu::BindGroup>,
    bound_generation: u64,
    start: web_time::Instant,
}

impl RippleMaterial2d {
    fn new() -> Self {
        let ctxt = Context::get();

        let bind_group_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ripple2d_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ripple2d_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let shader = ctxt.create_shader_module(Some("ripple2d_shader"), SHADER);

        let vertex_buffer_layouts = [
            // Mesh positions.
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                }],
            }),
        ];

        let pipeline = PipelineCache::new(move |sample_count| {
            let ctxt = Context::get();
            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ripple2d_pipeline"),
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
                        // The 2D scene draws into the linear HDR film, like the
                        // 3D one; the tonemap resolves it to the surface.
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

        let uniform_buffer = ctxt.create_buffer_simple(
            Some("ripple2d_uniform_buffer"),
            std::mem::size_of::<RippleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ripple2d_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        RippleMaterial2d {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            sampler,
            bind_group: None,
            bound_generation: 0,
            start: web_time::Instant::now(),
        }
    }

    /// The bind group over this frame's screen copy, rebuilt when the copy is a
    /// new texture (a resize) and reused otherwise.
    fn screen_bind_group(&mut self, context: &RenderContext2d) -> Option<&wgpu::BindGroup> {
        let screen = context.screen.as_ref()?;
        if self.bind_group.is_none() || self.bound_generation != context.screen_generation {
            let ctxt = Context::get();
            self.bind_group = Some(ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ripple2d_bind_group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(screen),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            }));
            self.bound_generation = context.screen_generation;
        }
        self.bind_group.as_ref()
    }
}

/// `Mat3` as three `vec4` columns.
fn padded_mat3(m: &Mat3) -> [[f32; 4]; 3] {
    let c = m.to_cols_array_2d();
    [
        [c[0][0], c[0][1], c[0][2], 0.0],
        [c[1][0], c[1][1], c[1][2], 0.0],
        [c[2][0], c[2][1], c[2][2], 0.0],
    ]
}

impl Material2d for RippleMaterial2d {
    fn create_gpu_data(&self) -> Box<dyn GpuData> {
        Box::new(NoGpuData)
    }

    /// What splits the 2D pass around this object and fills
    /// `RenderContext2d::screen` before it draws.
    fn reads_screen(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn render(
        &mut self,
        transform: Pose2,
        scale: Vec2,
        camera: &mut dyn Camera2d,
        _data: &ObjectData2d,
        mesh: &mut GpuMesh2d,
        _instances: &mut InstancesBuffer2d,
        _gpu_data: &mut dyn GpuData,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext2d,
    ) {
        let ctxt = Context::get();
        mesh.load_to_gpu();

        let coords = mesh.coords().read().unwrap();
        let faces = mesh.faces().read().unwrap();
        let (Some(coords_buf), Some(faces_buf)) = (coords.buffer(), faces.buffer()) else {
            return;
        };

        let (view, proj) = camera.view_transform_pair();
        let model = transform.to_mat3();
        let scale = Mat2::from_diagonal(scale);
        let uniforms = RippleUniforms {
            view: padded_mat3(&view),
            proj: padded_mat3(&proj),
            model: padded_mat3(&model),
            scale: [
                [scale.x_axis.x, scale.x_axis.y, 0.0, 0.0],
                [scale.y_axis.x, scale.y_axis.y, 0.0, 0.0],
            ],
            params: [
                context.viewport_width as f32,
                context.viewport_height as f32,
                self.start.elapsed().as_secs_f32(),
                0.012,
            ],
        };
        ctxt.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let pipeline = self.pipeline.get(context.sample_count);
        let Some(bind_group) = self.screen_bind_group(context) else {
            return;
        };

        render_pass.set_pipeline(&pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_vertex_buffer(0, coords_buf.slice(..));
        render_pass.set_index_buffer(faces_buf.slice(..), VERTEX_INDEX_FORMAT);
        render_pass.draw_indexed(0..mesh.num_indices(), 0, 0..1);
    }
}

const SHADER: &str = r#"
struct Uniforms {
    view_0: vec4<f32>,
    view_1: vec4<f32>,
    view_2: vec4<f32>,
    proj_0: vec4<f32>,
    proj_1: vec4<f32>,
    proj_2: vec4<f32>,
    model_0: vec4<f32>,
    model_1: vec4<f32>,
    model_2: vec4<f32>,
    scale_0: vec4<f32>,
    scale_1: vec4<f32>,
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var screen_tex: texture_2d<f32>;
@group(0) @binding(2) var screen_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local: vec2<f32>,
}

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VertexOutput {
    let view = mat3x3<f32>(u.view_0.xyz, u.view_1.xyz, u.view_2.xyz);
    let proj = mat3x3<f32>(u.proj_0.xyz, u.proj_1.xyz, u.proj_2.xyz);
    let model = mat3x3<f32>(u.model_0.xyz, u.model_1.xyz, u.model_2.xyz);
    let scale = mat2x2<f32>(u.scale_0.xy, u.scale_1.xy);

    let model_pos = model * vec3<f32>(scale * position, 1.0);
    var projected = proj * view * model_pos;
    projected.z = 0.0;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(projected, 1.0);
    out.local = position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let viewport = u.params.xy;
    let time = u.params.z;
    let strength = u.params.w;

    // Where this fragment sits in the copy of the frame so far.
    let uv = in.clip_position.xy / viewport;

    // A pair of travelling waves, one per axis, bends the lookup.
    let offset = vec2<f32>(
        sin(uv.y * 38.0 + time * 2.3),
        cos(uv.x * 26.0 + time * 1.7),
    ) * strength;
    let refracted = textureSample(screen_tex, screen_sampler, clamp(uv + offset, vec2(0.0), vec2(1.0)));

    // A cool tint and a bright edge, so the pane reads as glass rather than as
    // a shimmering hole.
    let edge = smoothstep(0.46, 0.5, max(abs(in.local.x), abs(in.local.y)));
    let tinted = refracted.rgb * vec3<f32>(0.82, 0.9, 1.05) + vec3<f32>(0.02);
    return vec4<f32>(mix(tinted, vec3<f32>(0.7, 0.85, 1.0), edge * 0.5), 1.0);
}
"#;
