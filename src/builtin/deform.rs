//! Shared GPU resources for vertex deformation (skeletal skinning + morph targets).
//!
//! Skinning and morph targets are unified into a single **deform** path: a mesh that
//! carries a joint palette *or* morph targets is drawn with the deformed pipeline
//! variant, whose vertex shader applies, in order:
//!
//! 1. **Morph**: `pos += Σ weightᵢ · Δposᵢ` (and the same for normals when present),
//! 2. **Skin**: the joint-palette blend, or — when the mesh isn't skinned — the
//!    ordinary object-transform path.
//!
//! A single [`DeformControl`] uniform (`has_skin`, `num_targets`, weights, …) gates
//! both, so there is just one extra shader variant instead of a 2×2 matrix. The
//! per-vertex skin joints/weights and the morph deltas are all read from **storage
//! buffers** indexed by `@builtin(vertex_index)`, so the deformed vertex layout is
//! identical to the plain one (no extra vertex buffers) and a morph-only mesh never
//! needs fabricated dummy joint/weight buffers.
//!
//! The deform data is bound as a single extra bind group (group 3 in the color/prepass
//! pipelines, group 2 in the shadow pipelines), so the total stays within the
//! WebGPU/WebGL2 cap of four groups and the deform path runs on **every target,
//! including web** — both the color pass and the shadow pass deform skinned/morphed
//! meshes, so an animated caster's shadow tracks its current pose everywhere.

use crate::context::Context;
use bytemuck::{Pod, Zeroable};
use std::cell::RefCell;

/// Maximum number of morph targets blended per mesh. Targets beyond
/// this are dropped at load time (with a warning).
pub const MAX_MORPH_TARGETS: usize = 64;

/// Per-object control uniform for the deform vertex shader. Bound as the last entry
/// of the deform bind group (group 4 in the color/prepass pipelines, group 2 in the
/// shadow pipelines).
///
/// Layout must match the WGSL `DeformControl` struct: four `u32`s (16 bytes) then a
/// `array<vec4<f32>, MAX_MORPH_TARGETS/4>` of weights (4 weights packed per `vec4`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DeformControl {
    /// Number of active morph targets (`0` disables morphing).
    pub num_targets: u32,
    /// Number of vertices each morph target spans (the storage-buffer row stride).
    pub num_vertices: u32,
    /// `1` when the mesh is skinned (apply the joint palette), else `0`.
    pub has_skin: u32,
    /// `1` when per-target morph normal deltas are present, else `0`.
    pub has_morph_normals: u32,
    /// Morph weights, packed four per `vec4` (so `[MAX_MORPH_TARGETS / 4]` of them).
    pub weights: [[f32; 4]; MAX_MORPH_TARGETS / 4],
}

impl Default for DeformControl {
    fn default() -> Self {
        Zeroable::zeroed()
    }
}

impl DeformControl {
    /// Packs a flat weight slice into the `vec4`-padded `weights` field, clamping the
    /// target count to [`MAX_MORPH_TARGETS`].
    pub fn set_weights(&mut self, weights: &[f32]) {
        let n = weights.len().min(MAX_MORPH_TARGETS);
        self.num_targets = n as u32;
        self.weights = [[0.0; 4]; MAX_MORPH_TARGETS / 4];
        for (i, &w) in weights[..n].iter().enumerate() {
            self.weights[i >> 2][i & 3] = w;
        }
    }
}

/// Lazily-built process-wide deform resources: the shared bind-group layout (so the
/// per-object bind group is compatible with both the color and shadow pipelines) and
/// the 1-element fallback buffers bound for the streams a given mesh lacks.
struct DeformGlobals {
    layout: wgpu::BindGroupLayout,
    /// 1-element identity joint palette, bound when a (morph-only) mesh isn't skinned.
    identity_palette: wgpu::Buffer,
    /// 1-element zero joints/weights, bound when a (morph-only) mesh has no skin.
    dummy_joints: wgpu::Buffer,
    dummy_weights: wgpu::Buffer,
    /// 1-element zero morph delta, bound when a (skin-only) mesh has no morph
    /// positions/normals. Shared by both delta bindings.
    dummy_morph: wgpu::Buffer,
}

thread_local! {
    static GLOBALS: RefCell<Option<DeformGlobals>> = const { RefCell::new(None) };
}

impl DeformGlobals {
    fn new() -> Self {
        let ctxt = Context::get();

        // 0..4 read-only storage (palette, skin joints, skin weights, morph
        // positions, morph normals); 5 the control uniform. All vertex-stage.
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deform_bind_group_layout"),
            entries: &[
                storage(0),
                storage(1),
                storage(2),
                storage(3),
                storage(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let identity = glamx::Mat4::IDENTITY.to_cols_array();
        let identity_palette = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deform_identity_palette"),
            size: std::mem::size_of::<[f32; 16]>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctxt.write_buffer(&identity_palette, 0, bytemuck::cast_slice(&identity));

        let dummy = |label, bytes: &[u8]| {
            let buf = ctxt.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            ctxt.write_buffer(&buf, 0, bytes);
            buf
        };

        DeformGlobals {
            layout,
            identity_palette,
            dummy_joints: dummy("deform_dummy_joints", bytemuck::cast_slice(&[0u32; 4])),
            dummy_weights: dummy("deform_dummy_weights", bytemuck::cast_slice(&[0.0f32; 4])),
            dummy_morph: dummy("deform_dummy_morph", bytemuck::cast_slice(&[0.0f32; 4])),
        }
    }
}

fn with_globals<R>(f: impl FnOnce(&DeformGlobals) -> R) -> R {
    GLOBALS.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(DeformGlobals::new());
        }
        f(cell.borrow().as_ref().unwrap())
    })
}

/// The shared deform bind-group layout, used as group 4 of the color/prepass
/// pipelines and group 2 of the shadow pipelines. All these pipelines reference the
/// *same* layout object, so one per-object bind group works in every pass.
pub fn deform_bind_group_layout() -> wgpu::BindGroupLayout {
    with_globals(|g| g.layout.clone())
}

/// Per-object GPU deform state: the [`DeformControl`] uniform buffer and a cached
/// bind group over it plus the object's palette/skin/morph storage buffers.
///
/// Rebuilt only when a source buffer changes (e.g. the joint palette is reallocated
/// as it grows); the control uniform is rewritten every frame from the current morph
/// weights. Lives on the object and is refreshed once per frame, so both the color
/// and shadow passes read the same bind group.
pub struct DeformGpu {
    control: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    /// Identities of the bound buffers (`0` = fallback), to skip needless rebuilds.
    key: Option<[usize; 5]>,
}

impl DeformGpu {
    /// Allocates the per-object control uniform buffer.
    pub fn new() -> Self {
        let ctxt = Context::get();
        let control = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deform_control_buffer"),
            size: std::mem::size_of::<DeformControl>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            control,
            bind_group: None,
            key: None,
        }
    }

    /// Writes `ctrl` into the control uniform and (re)builds the bind group when any
    /// source buffer changed.
    pub fn update(
        &mut self,
        ctrl: &DeformControl,
        palette: Option<&wgpu::Buffer>,
        joints: Option<&wgpu::Buffer>,
        weights: Option<&wgpu::Buffer>,
        morph_pos: Option<&wgpu::Buffer>,
        morph_nrm: Option<&wgpu::Buffer>,
    ) {
        let ctxt = Context::get();
        ctxt.write_buffer(&self.control, 0, bytemuck::bytes_of(ctrl));

        let ptr = |b: Option<&wgpu::Buffer>| b.map_or(0, |x| x as *const wgpu::Buffer as usize);
        let key = [
            ptr(palette),
            ptr(joints),
            ptr(weights),
            ptr(morph_pos),
            ptr(morph_nrm),
        ];
        if self.bind_group.is_none() || self.key != Some(key) {
            self.bind_group = Some(build_deform_bind_group(
                "deform_bind_group",
                palette,
                joints,
                weights,
                morph_pos,
                morph_nrm,
                &self.control,
            ));
            self.key = Some(key);
        }
    }

    /// The cached deform bind group, if built.
    pub fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bind_group.as_ref()
    }
}

impl Default for DeformGpu {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds the per-object deform bind group, substituting the shared fallback buffers
/// for any stream the mesh lacks (`None`). `control` is the object's own
/// [`DeformControl`] uniform buffer.
pub fn build_deform_bind_group(
    label: &str,
    palette: Option<&wgpu::Buffer>,
    joints: Option<&wgpu::Buffer>,
    weights: Option<&wgpu::Buffer>,
    morph_pos: Option<&wgpu::Buffer>,
    morph_nrm: Option<&wgpu::Buffer>,
    control: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let ctxt = Context::get();
    with_globals(|g| {
        let palette = palette.unwrap_or(&g.identity_palette);
        let joints = joints.unwrap_or(&g.dummy_joints);
        let weights = weights.unwrap_or(&g.dummy_weights);
        let morph_pos = morph_pos.unwrap_or(&g.dummy_morph);
        let morph_nrm = morph_nrm.unwrap_or(&g.dummy_morph);
        ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &g.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: palette.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: joints.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: morph_pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: morph_nrm.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: control.as_entire_binding(),
                },
            ],
        })
    })
}
