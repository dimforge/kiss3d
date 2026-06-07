//! Screen-space reflections (SSR) for the rasterizer.
//!
//! Runs after the opaque pass + MSAA resolve, consuming the geometry G-buffer
//! produced by the shared prepass (view position, world normal + roughness, F0 +
//! metallic) and the resolved HDR scene color. A roughness-blur mip chain of the
//! scene is built, then a single fragment pass marches each glossy pixel's
//! reflection ray in screen space (DDA with binary-search refinement) and blends
//! the result additively into the scene as a *delta* over the environment specular
//! the forward pass already wrote (so reflections are not double-counted, and
//! screen misses fall back to the environment/probes). Single-frame; native/WebGPU
//! only.

use crate::context::Context;
use crate::resource::EnvLight;
use bytemuck::{Pod, Zeroable};
use glamx::Mat4;

/// Tunable SSR parameters.
#[derive(Copy, Clone, Debug)]
pub struct SsrSettings {
    /// Maximum ray-march steps per pixel.
    pub max_steps: u32,
    /// View-space tolerance for accepting a depth intersection.
    pub thickness: f32,
    /// Maximum view-space ray length before giving up.
    pub max_distance: f32,
    /// Surfaces rougher than this skip SSR (and fade out approaching it).
    pub roughness_cutoff: f32,
    /// Screen-edge fade width (in UV) to hide reflections running off-screen.
    pub edge_fade: f32,
    /// Global reflection intensity multiplier (on top of each object's per-object
    /// [`SsrMaterial::intensity`]).
    pub intensity: f32,
}

impl Default for SsrSettings {
    fn default() -> Self {
        SsrSettings {
            max_steps: 48,
            thickness: 0.5,
            max_distance: 60.0,
            roughness_cutoff: 0.6,
            edge_fade: 0.12,
            intensity: 1.0,
        }
    }
}

/// Per-object screen-space-reflection properties, set with
/// [`Object3d::set_ssr`](crate::scene::Object3d::set_ssr). `None` there means the
/// object receives no SSR (gated off); `Some(SsrMaterial { .. })` makes it receive
/// SSR with these properties. Combined with the window-global [`SsrSettings`]
/// (which holds the march-quality knobs shared by all objects).
#[derive(Copy, Clone, Debug)]
pub struct SsrMaterial {
    /// Per-object reflection strength (multiplies the global intensity). `0`
    /// disables SSR on this object, same as `set_ssr(None)`.
    pub intensity: f32,
    /// Treat hit surfaces as infinitely thick: accept any depth crossing as a hit
    /// (skips the thickness lower-bound test). Removes light-leak behind thin
    /// geometry and recovers hits the thickness test would otherwise reject.
    pub infinite_thick: bool,
    /// Fade this object's reflections quadratically with the hit distance.
    pub distance_attenuation: bool,
    /// Boost this object's reflections at grazing angles (Fresnel,
    /// on top of the physically-based BRDF weight).
    pub fresnel: bool,
}

impl Default for SsrMaterial {
    fn default() -> Self {
        SsrMaterial {
            intensity: 1.0,
            infinite_thick: false,
            distance_attenuation: true,
            fresnel: false,
        }
    }
}

impl SsrMaterial {
    /// Packs the per-object SSR properties into the vec4 the prepass writes into the
    /// G-buffer: `(intensity, infinite_thick, distance_attenuation, fresnel)`. An
    /// object with no SSR (`None`) packs to all-zero (intensity 0 = disabled).
    pub fn pack(this: Option<SsrMaterial>) -> [f32; 4] {
        match this {
            Some(m) => [
                m.intensity.max(0.0),
                if m.infinite_thick { 1.0 } else { 0.0 },
                if m.distance_attenuation { 1.0 } else { 0.0 },
                if m.fresnel { 1.0 } else { 0.0 },
            ],
            None => [0.0, 0.0, 0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SsrUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    params0: [f32; 4],
    params1: [f32; 4],
    ibl: [f32; 4],
    misc: [f32; 4],
}

/// Owns the SSR reflection mip chain, pipelines and uniform for one window.
pub struct Ssr {
    settings: SsrSettings,
    width: u32,
    height: u32,

    // Roughness-blur mip chain, built from the resolved scene each frame.
    _refl_texture: wgpu::Texture,
    refl_view: wgpu::TextureView,
    refl_mips: u32,

    sampler: wgpu::Sampler,
    // 1x1 black env fallback bound when no skybox/IBL is set.
    _env_fallback_texture: wgpu::Texture,
    env_fallback_view: wgpu::TextureView,

    // Reflection mip-chain build (reuses env_downsample.wgsl).
    downsample_layout: wgpu::BindGroupLayout,
    downsample_pipeline: wgpu::RenderPipeline,

    // SSR additive pass.
    ssr_layout: wgpu::BindGroupLayout,
    ssr_pipeline: wgpu::RenderPipeline,
    ssr_uniform: wgpu::Buffer,
}

impl Ssr {
    /// Creates the SSR resources for the given size.
    pub fn new(width: u32, height: u32) -> Ssr {
        let ctxt = Context::get();
        let w = width.max(1);
        let h = height.max(1);

        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssr_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let (refl_texture, refl_view, refl_mips) = Self::make_refl(w, h);

        // 1x1 black env fallback.
        let env_fallback_texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssr_env_fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctxt.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &env_fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0u8; 8],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let env_fallback_view =
            env_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // The downsample shader uses only a texture + sampler (no uniform).
        let downsample_shader = ctxt.create_shader_module(
            Some("ssr_downsample"),
            include_str!("../builtin/env_downsample.wgsl"),
        );
        let downsample_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr_downsample_layout"),
            entries: &[
                tex_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let downsample_pipeline = make_fullscreen_pipeline(
            "ssr_downsample",
            &downsample_shader,
            &downsample_layout,
            None,
            wgpu::ColorWrites::ALL,
        );

        // SSR additive pass: viewpos, normal, material, refl-chain, env, per-object
        // SSR params + sampler + uniform.
        let ssr_shader =
            ctxt.create_shader_module(Some("ssr"), include_str!("../builtin/ssr.wgsl"));
        let ssr_layout = make_layout("ssr_layout", 6);
        // Additive blend so the SSR delta adds onto the resolved scene; the COLOR
        // write mask leaves alpha untouched.
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let ssr_pipeline = make_fullscreen_pipeline(
            "ssr",
            &ssr_shader,
            &ssr_layout,
            Some(additive),
            wgpu::ColorWrites::COLOR,
        );

        let ssr_uniform = ctxt.create_buffer_simple(
            Some("ssr_uniform"),
            std::mem::size_of::<SsrUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        Ssr {
            settings: SsrSettings::default(),
            width: w,
            height: h,
            _refl_texture: refl_texture,
            refl_view,
            refl_mips,
            sampler,
            _env_fallback_texture: env_fallback_texture,
            env_fallback_view,
            downsample_layout,
            downsample_pipeline,
            ssr_layout,
            ssr_pipeline,
            ssr_uniform,
        }
    }

    fn make_refl(w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView, u32) {
        let ctxt = Context::get();
        // Cap the chain so the coarsest mip stays a few texels wide.
        let mips = (32 - w.max(h).leading_zeros()).clamp(1, 7);
        let texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssr_reflection_chain"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::post_processing::HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view, mips)
    }

    /// Resizes the SSR targets if needed.
    pub fn resize(&mut self, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.width == w && self.height == h {
            return;
        }
        let (tex, view, mips) = Self::make_refl(w, h);
        self._refl_texture = tex;
        self.refl_view = view;
        self.refl_mips = mips;
        self.width = w;
        self.height = h;
    }

    /// Mutable access to the SSR settings.
    pub fn settings_mut(&mut self) -> &mut SsrSettings {
        &mut self.settings
    }

    /// Runs SSR: builds the reflection mip chain from `scene_view`, then marches
    /// and blends the reflection delta additively into `scene_view`. `viewpos`,
    /// `normal` and `material` are the prepass G-buffer attachments; `view`/`proj`
    /// are the (pass-0) camera matrices; `env` is the global IBL fallback (if any).
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_view: &wgpu::TextureView,
        viewpos: &wgpu::TextureView,
        normal: &wgpu::TextureView,
        material: &wgpu::TextureView,
        ssr_params: &wgpu::TextureView,
        view: Mat4,
        proj: Mat4,
        env: Option<EnvLight<'_>>,
    ) {
        let ctxt = Context::get();

        // 1. Build the reflection mip chain (mip 0 = resolved scene).
        self.build_refl_chain(encoder, scene_view);

        // 2. Upload uniforms.
        let (ibl_has, ibl_max_lod, ibl_intensity, ibl_rotation, env_view) = match &env {
            Some(e) => (
                1.0,
                (e.mip_count.max(1) - 1) as f32,
                e.intensity,
                e.rotation,
                e.view,
            ),
            None => (0.0, 0.0, 0.0, 0.0, &self.env_fallback_view),
        };
        let s = &self.settings;
        ctxt.write_buffer(
            &self.ssr_uniform,
            0,
            bytemuck::bytes_of(&SsrUniforms {
                view: view.to_cols_array_2d(),
                proj: proj.to_cols_array_2d(),
                params0: [
                    1.0 / self.width as f32,
                    1.0 / self.height as f32,
                    s.max_steps as f32,
                    s.thickness,
                ],
                params1: [0.0, s.max_distance, s.roughness_cutoff, s.edge_fade],
                ibl: [ibl_has, ibl_max_lod, ibl_intensity, ibl_rotation],
                misc: [(self.refl_mips.max(1) - 1) as f32, s.intensity, 0.0, 0.0],
            }),
        );

        // 3. SSR additive pass into the resolved scene.
        let bg = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssr_bg"),
            layout: &self.ssr_layout,
            entries: &[
                tex_bind(0, viewpos),
                tex_bind(1, normal),
                tex_bind(2, material),
                tex_bind(3, &self.refl_view),
                tex_bind(4, env_view),
                tex_bind(5, ssr_params),
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.ssr_uniform.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssr_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: scene_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.ssr_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Box-downsamples `scene_view` into the reflection mip chain (mip 0 = a copy
    /// of the resolved scene, coarser mips = blurrier reflections).
    fn build_refl_chain(&self, encoder: &mut wgpu::CommandEncoder, scene_view: &wgpu::TextureView) {
        let ctxt = Context::get();
        for mip in 0..self.refl_mips {
            let prev_view = if mip == 0 {
                None
            } else {
                Some(self._refl_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("ssr_refl_src"),
                    base_mip_level: mip - 1,
                    mip_level_count: Some(1),
                    ..Default::default()
                }))
            };
            let src_ref: &wgpu::TextureView = prev_view.as_ref().unwrap_or(scene_view);
            let dst = self._refl_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("ssr_refl_dst"),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let bg = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssr_downsample_bg"),
                layout: &self.downsample_layout,
                entries: &[
                    tex_bind(0, src_ref),
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssr_downsample_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst,
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
            pass.set_pipeline(&self.downsample_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// Builds a bind-group layout: `n_tex` filtered textures, then a sampler, then a
/// uniform buffer (all fragment-stage).
fn make_layout(label: &str, n_tex: u32) -> wgpu::BindGroupLayout {
    let ctxt = Context::get();
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..n_tex).map(tex_entry).collect();
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: n_tex,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    });
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: n_tex + 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn tex_bind<'a>(binding: u32, view: &'a wgpu::TextureView) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn make_fullscreen_pipeline(
    label: &str,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    blend: Option<wgpu::BlendState>,
    write_mask: wgpu::ColorWrites,
) -> wgpu::RenderPipeline {
    let ctxt = Context::get();
    let pl = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: crate::post_processing::HDR_FORMAT,
                blend,
                write_mask,
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
}
