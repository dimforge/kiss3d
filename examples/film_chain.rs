//! The film-stage post-processing chain: effects that run on the HDR film,
//! before bloom and the tonemap, instead of on the finished LDR image.
//!
//! Both chains are handed to [`Window::render_chains`]. The same effect is
//! moved between them with the space bar, and the difference is what linear
//! light buys: a haze added to the film is real light, so it blows the
//! highlights out and blooms. The same haze added after the tonemap can only
//! wash the picture towards white.

use kiss3d::post_processing::{FormatPipelines, PostProcessingContext, PostProcessingEffect};
use kiss3d::prelude::*;
use kiss3d::resource::RenderTarget;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: film_chain").await;
    window.set_background_color(Color::new(0.01, 0.012, 0.02, 1.0));
    window.set_bloom_enabled(true);
    window.set_bloom(1.0, 0.6);

    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 1.5, 6.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(60.0))
        .set_position(Vec3::new(3.0, 5.0, 4.0));

    let mut sphere = scene.add_sphere(1.0);
    sphere.set_color(Color::new(0.15, 0.16, 0.2, 1.0));
    let mut lamp = scene.add_sphere(0.35);
    lamp.set_color(WHITE);
    lamp.set_emissive(Color::new(6.0, 4.4, 2.2, 1.0));
    lamp.set_position(Vec3::new(-2.0, 1.2, 1.0));

    let mut haze = Haze::new();
    let mut on_film = true;
    println!("Space: move the haze between the film chain and the post chain.");

    loop {
        for event in window.events().iter() {
            if let WindowEvent::Key(Key::Space, Action::Press, _) = event.value {
                on_film = !on_film;
                println!(
                    "haze on the {} chain",
                    if on_film { "film" } else { "post" }
                );
            }
        }

        let running = if on_film {
            window
                .render_chains(
                    Some(&mut scene),
                    None,
                    Some(&mut camera),
                    None,
                    None,
                    &mut [&mut haze],
                    &mut [],
                )
                .await
        } else {
            window
                .render_chain(
                    Some(&mut scene),
                    None,
                    Some(&mut camera),
                    None,
                    None,
                    &mut [&mut haze],
                )
                .await
        };
        if !running {
            break;
        }
    }
}

/// Adds a soft radial glow to whatever it is given.
///
/// On the film chain the glow is added in linear light, so it lifts values past
/// 1.0 and both blooms and tonemaps. On the post chain it is added to an image
/// that is already tonemapped, where the same numbers can only approach white.
struct Haze {
    pipeline: FormatPipelines,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    uniforms: HazeUniforms,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct HazeUniforms {
    /// `(viewport width, viewport height, intensity, unused)`.
    params: [f32; 4],
}

impl Haze {
    fn new() -> Self {
        let ctxt = Context::get();

        let bind_group_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("haze_bind_group_layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("haze_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let shader = ctxt.create_shader_module(Some("haze_shader"), SHADER);

        // One pipeline per format it is asked to write: the film chain's HDR
        // targets, the post chain's surface-format ones.
        let pipeline = FormatPipelines::new(move |format| {
            let ctxt = Context::get();
            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("haze_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    })],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
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
        });

        let vertices: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
        let vertex_buffer = ctxt.create_buffer_init(
            Some("haze_vertex_buffer"),
            bytemuck::cast_slice(&vertices),
            wgpu::BufferUsages::VERTEX,
        );
        let uniform_buffer = ctxt.create_buffer_simple(
            Some("haze_uniform_buffer"),
            std::mem::size_of::<HazeUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        Haze {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            vertex_buffer,
            uniforms: HazeUniforms {
                params: [1.0, 1.0, 1.6, 0.0],
            },
        }
    }
}

impl PostProcessingEffect for Haze {
    fn update(&mut self, _dt: f32, w: f32, h: f32, _znear: f32, _zfar: f32) {
        self.uniforms.params[0] = w;
        self.uniforms.params[1] = h;
    }

    fn draw(&mut self, target: &RenderTarget, context: &mut PostProcessingContext) {
        let ctxt = Context::get();
        let RenderTarget::Offscreen(source) = target else {
            return;
        };

        ctxt.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
        let pipeline = self.pipeline.get(context.output_format);
        let bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("haze_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source.color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&source.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("haze_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: context.output_view,
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
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..4, 0..1);
    }
}

const SHADER: &str = r#"
struct Uniforms {
    params: vec4<f32>,
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    out.uv = vec2<f32>(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let src = textureSample(src_tex, src_sampler, in.uv);

    // A warm blob left of centre, falling off smoothly.
    let aspect = u.params.x / max(u.params.y, 1.0);
    let d = distance(vec2<f32>(in.uv.x * aspect, in.uv.y), vec2<f32>(0.34 * aspect, 0.46));
    let glow = pow(clamp(1.0 - d * 2.6, 0.0, 1.0), 3.0) * u.params.z;

    return vec4<f32>(src.rgb + glow * vec3<f32>(1.0, 0.72, 0.38), src.a);
}
"#;
