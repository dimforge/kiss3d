//! Data structure of a scene node.

use crate::camera::Camera3d;
use crate::color::Color;
use crate::context::Context;
use crate::light::LightCollection;
use crate::resource::vertex_index::{VertexIndex, VERTEX_INDEX_FORMAT};
use crate::resource::{
    AllocationType, BufferType, GPUVec, GpuData, GpuMesh3d, Material3d, RenderContext, RenderPhase,
    Texture, TextureManager,
};
use crate::scene::SceneNodeData3d;
use glamx::{Mat3, Mat4, Pose3, Vec2, Vec3};
use std::any::Any;
use std::cell::RefCell;
use std::path::Path;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Skeletal skinning binding for a skinned mesh.
///
/// Holds **weak** references to the skeleton's joint nodes (so the skin never
/// keeps the scene graph alive or forms a reference cycle), the per-joint inverse
/// bind matrices from glTF, and a per-frame scratch `palette` of joint matrices.
///
/// The palette is recomputed each frame from the (already-propagated) joint world
/// transforms as `palette[j] = joint_world[j] * inverse_bind[j]` and uploaded to a
/// GPU storage buffer that the skinned vertex shader reads. Per the glTF spec the
/// skinned mesh node's own transform is ignored, so the palette is expressed
/// directly in world space.
///
/// GPU skinning requires a 5th bind group, which exceeds the WebGPU/WebGL2 cap of
/// four, so the rasterizer's skinning is **native-only**; on the web a skinned mesh
/// falls back to its bind pose. On native the skinned deformation is applied in
/// every pass that draws the mesh: the color pass, the SSAO depth prepass, and both
/// the opaque and translucent (transmittance) **shadow** passes, so animated
/// characters cast correctly-posed (and correctly-tinted) shadows. The **path
/// tracer** also renders the animated pose — it has no GPU skinning, so the geometry
/// is CPU-skinned into world space when the scene is gathered (see
/// `raytracer::scene_data`), which works on every platform.
pub struct Skin3d {
    /// Weak handles to the joint nodes, in glTF skin-joint order.
    pub(crate) joints: Vec<Weak<RefCell<SceneNodeData3d>>>,
    /// Inverse bind matrix per joint (same order as `joints`).
    pub(crate) inverse_bind: Vec<Mat4>,
    /// Joint matrix palette, recomputed every frame (same length as `joints`).
    pub(crate) palette: Vec<Mat4>,
    /// GPU storage buffer holding `palette`, uploaded once per frame by
    /// [`upload`](Self::upload). Shared by the color, prepass, and shadow passes.
    palette_buffer: Option<wgpu::Buffer>,
    /// Capacity of `palette_buffer`, in `mat4x4`s.
    palette_capacity: usize,
}

impl Skin3d {
    /// Creates a skin from its joint node handles and inverse bind matrices
    /// (which must have equal length). The palette starts at identity.
    pub(crate) fn new(
        joints: Vec<Weak<RefCell<SceneNodeData3d>>>,
        inverse_bind: Vec<Mat4>,
    ) -> Self {
        let n = joints.len();
        Skin3d {
            joints,
            inverse_bind,
            palette: vec![Mat4::IDENTITY; n],
            palette_buffer: None,
            palette_capacity: 0,
        }
    }

    /// The number of joints influencing this skin.
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    /// The current joint-matrix palette (world-space, column-major `Mat4`s). Used
    /// by the path tracer to CPU-skin the gathered geometry.
    pub(crate) fn palette(&self) -> &[Mat4] {
        &self.palette
    }

    /// Uploads the current palette to the GPU storage buffer, (re)allocating it by
    /// powers of two if it grew. Called each frame after the palette is recomputed,
    /// before any render pass consumes it.
    pub(crate) fn upload(&mut self) {
        if self.palette.is_empty() {
            return;
        }
        let ctxt = Context::get();
        if self.palette_buffer.is_none() || self.palette.len() > self.palette_capacity {
            let cap = self.palette.len().next_power_of_two().max(1);
            self.palette_buffer = Some(ctxt.create_buffer(&wgpu::BufferDescriptor {
                label: Some("skin_palette_buffer"),
                size: (cap * std::mem::size_of::<[[f32; 4]; 4]>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.palette_capacity = cap;
        }
        ctxt.write_buffer(
            self.palette_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&self.palette),
        );
    }

    /// The joint-palette GPU storage buffer, if it has been uploaded. Only consumed
    /// by the native deform path; the web build uploads the palette but never reads
    /// the buffer (skinned meshes fall back to the rest shape there).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn palette_buffer(&self) -> Option<&wgpu::Buffer> {
        self.palette_buffer.as_ref()
    }
}

/// The shading model used by the path tracer for an object's surface.
///
/// The rasterizer always uses its metallic-roughness PBR shader; this only
/// selects which lobe set the path tracer's unified BSDF evaluates.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Bsdf {
    /// Opaque metallic-roughness PBR (the default).
    Opaque,
    /// Dielectric glass: smooth or rough refraction governed by `ior`/`roughness`.
    Glass,
    /// Pure conductor: reflection only, tinted by `specular_tint`.
    Metal,
    /// Emitter; shaded as opaque but contributes light via its emissive color.
    Emissive,
}

impl Bsdf {
    /// The WGSL `bsdf_type` tag for this model.
    pub(crate) fn tag(self) -> u32 {
        match self {
            Bsdf::Opaque => 0,
            Bsdf::Glass => 1,
            Bsdf::Metal => 2,
            Bsdf::Emissive => 3,
        }
    }
}

/// How a surface's alpha channel is interpreted when shading.
///
/// The common alpha-blending modes. The default, [`Blend`], keeps
/// kiss3d's historical behavior: a surface is treated as transparent (and routed
/// through the order-independent transparency pass) exactly when its color alpha
/// is below `1.0`.
///
/// [`Blend`]: AlphaMode::Blend
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AlphaMode {
    /// Fully opaque: the alpha channel is ignored and the surface always renders
    /// in the opaque pass.
    Opaque,
    /// Alpha masking / cutout: fragments whose alpha is below the cutoff are
    /// discarded; everything else is opaque. Good for foliage and chain-link.
    Mask(f32),
    /// Standard (straight) alpha blending through the order-independent
    /// transparency pass when alpha `< 1.0`.
    Blend,
    /// Premultiplied alpha blending (color is already multiplied by alpha) through
    /// the order-independent transparency pass when alpha `< 1.0`.
    Premultiplied,
}

impl Default for AlphaMode {
    fn default() -> Self {
        AlphaMode::Blend
    }
}

impl AlphaMode {
    /// Whether a surface with this mode and `color_alpha` renders in the
    /// transparent (OIT) pass rather than the opaque pass.
    pub(crate) fn is_transparent(self, color_alpha: f32) -> bool {
        matches!(self, AlphaMode::Blend | AlphaMode::Premultiplied) && color_alpha < 1.0
    }

    /// `(mode_code, cutoff)` packed for the shader. Codes: 0 opaque, 1 mask,
    /// 2 blend, 3 premultiplied.
    pub(crate) fn shader_params(self) -> (u32, f32) {
        match self {
            AlphaMode::Opaque => (0, 0.0),
            AlphaMode::Mask(c) => (1, c),
            AlphaMode::Blend => (2, 0.0),
            AlphaMode::Premultiplied => (3, 0.0),
        }
    }
}

/// How parallax mapping marches the height field.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParallaxMethod {
    /// Parallax-occlusion mapping: linear search + interpolation. Cheaper.
    Occlusion,
    /// Relief mapping: linear search + `max_steps` binary-search refinements.
    /// Sharper than occlusion at the cost of extra samples; more steps give a
    /// crisper crossing.
    Relief { max_steps: u32 },
}

impl Default for ParallaxMethod {
    fn default() -> Self {
        ParallaxMethod::Occlusion
    }
}

impl ParallaxMethod {
    /// Encodes the method into the shader's `parallax.w` slot: `0` selects
    /// occlusion, any positive value selects relief with that many binary-search
    /// steps (clamped to the shader's hard loop cap of 64).
    pub(crate) fn code(self) -> f32 {
        match self {
            ParallaxMethod::Occlusion => 0.0,
            ParallaxMethod::Relief { max_steps } => max_steps.clamp(1, 64) as f32,
        }
    }
}

/// Monotonic counter handing out a unique default segmentation id to each new
/// object. Starts at 1 so that 0 stays reserved for "background" (empty pixels)
/// in the segmentation auxiliary render output.
static NEXT_SEGMENTATION_ID: AtomicU32 = AtomicU32::new(1);

/// Returns a fresh, process-unique segmentation id for a newly created object.
fn next_segmentation_id() -> u32 {
    NEXT_SEGMENTATION_ID.fetch_add(1, Ordering::Relaxed)
}

/// Rendering properties and state for a scene object.
///
/// Contains material, texture, color, and rendering settings for a 3D object.
/// This data is used by the rendering pipeline to determine how the object should be drawn.
pub struct ObjectData3d {
    material: Rc<RefCell<Box<dyn Material3d + 'static>>>,
    texture: Arc<Texture>,
    color: Color,
    lines_color: Option<Color>,
    points_color: Option<Color>,
    wlines: f32,
    wpoints: f32,
    lines_use_perspective: bool,
    points_use_perspective: bool,
    draw_surface: bool,
    cull: bool,
    /// Integer object identifier written to the segmentation auxiliary output.
    /// Auto-assigned to a process-unique value on creation; user-overridable.
    segmentation_id: u32,
    user_data: Box<dyn Any + 'static>,
    /// Render-layer bitmask. The object is drawn by a camera only when this
    /// shares a bit with the camera's mask. Defaults to layer 0 (`1`).
    render_layers: u32,
    // PBR material properties
    metallic: f32,
    roughness: f32,
    emissive: Color,
    alpha_mode: AlphaMode,
    // Path-tracer BSDF properties (ignored by the rasterizer).
    bsdf: Bsdf,
    ior: f32,
    transmission: f32,
    specular_tint: Color,
    subsurface: f32,
    subsurface_radius: f32,
    // Extended PBR surface properties, honored by BOTH the rasterizer and the
    // path tracer where applicable.
    /// Dielectric specular intensity remap in `[0, 1]`: `F0 = 0.16 *
    /// reflectance^2` for non-metals (standard dielectric convention). `0.5`
    /// reproduces the classic `0.04`.
    reflectance: f32,
    /// Clearcoat layer strength in `[0, 1]` (a second, smooth specular lobe).
    clearcoat: f32,
    /// Roughness of the clearcoat layer in `[0, 1]`.
    clearcoat_roughness: f32,
    /// Anisotropy strength in `[-1, 1]`: stretches the specular highlight along
    /// the tangent (`> 0`) or bitangent (`< 0`).
    anisotropy: f32,
    /// Rotation of the anisotropy direction around the surface normal, in radians.
    anisotropy_rotation: f32,
    /// Per-object screen-space-reflection properties. `Some` (the default) makes
    /// the object receive SSR; `None` opts it out. Ignored when SSR is disabled.
    ssr: Option<crate::renderer::SsrMaterial>,
    /// Planar reflector (mirror): when set, this surface shows a reflected view of
    /// the scene blended over its PBR shading. `None` (the default) is a normal
    /// surface. See [`Object3d::set_reflector`].
    reflector: Option<crate::renderer::Reflector>,
    // PBR texture maps
    normal_map: Option<Arc<Texture>>,
    metallic_roughness_map: Option<Arc<Texture>>,
    ao_map: Option<Arc<Texture>>,
    emissive_map: Option<Arc<Texture>>,
    /// Height/displacement map for parallax mapping (grayscale; brighter = higher).
    height_map: Option<Arc<Texture>>,
    /// Parallax displacement scale (surface depth in UV units). `0` disables it.
    parallax_scale: f32,
    /// Maximum number of parallax search layers (more = sharper, costlier).
    parallax_layers: f32,
    /// Parallax search method (occlusion vs relief).
    parallax_method: ParallaxMethod,
    /// Skeletal skinning binding, present only on skinned glTF meshes. When set,
    /// the object is drawn with the GPU skinning (deform) pipeline.
    skin: Option<Skin3d>,
    /// Current morph-target weights (one per target), driven by animation or set
    /// manually. Empty when the mesh has no morph targets. Matches the target count
    /// of the object's [`GpuMesh3d`].
    morph_weights: Vec<f32>,
    /// Per-object GPU deform resources (control uniform + cached deform bind group),
    /// lazily built when the object is skinned or morphed. `None` on the web, where
    /// the deform path is unavailable (see [`crate::builtin::deform`]).
    deform: Option<crate::builtin::deform::DeformGpu>,
    /// Cached albedo-texture bind group for the shadow transmittance pass (so a
    /// translucent caster's shadow is tinted by its texture). Lazily built;
    /// rebuilt when `texture` changes (`cached_shadow_tex_ptr`).
    shadow_tex_bind_group: Option<wgpu::BindGroup>,
    cached_shadow_tex_ptr: usize,
}

impl ObjectData3d {
    /// Returns a reference to this object's texture.
    ///
    /// # Returns
    /// A reference-counted texture
    #[inline]
    pub fn texture(&self) -> &Arc<Texture> {
        &self.texture
    }

    /// Returns the base color of this object.
    ///
    /// # Returns
    /// RGBA color with components in range [0.0, 1.0]
    #[inline]
    pub fn color(&self) -> Color {
        self.color
    }

    /// Returns the line width used for wireframe rendering.
    ///
    /// # Returns
    /// Line width in pixels
    #[inline]
    pub fn lines_width(&self) -> f32 {
        self.wlines
    }

    /// Returns the color used for wireframe line rendering.
    ///
    /// # Returns
    /// `Some(color)` if a custom line color is set, `None` to use the object's base color
    #[inline]
    pub fn lines_color(&self) -> Option<Color> {
        self.lines_color
    }

    /// Returns the point size used for point cloud rendering.
    ///
    /// # Returns
    /// Point size in pixels
    #[inline]
    pub fn points_size(&self) -> f32 {
        self.wpoints
    }

    /// Returns the color used for point rendering.
    ///
    /// # Returns
    /// `Some(color)` if a custom point color is set, `None` to use the object's base color
    #[inline]
    pub fn points_color(&self) -> Option<Color> {
        self.points_color
    }

    /// Checks if wireframe lines use perspective projection.
    ///
    /// # Returns
    /// `true` if wireframe lines scale with distance (perspective), `false` for constant screen-space width
    #[inline]
    pub fn lines_use_perspective(&self) -> bool {
        self.lines_use_perspective
    }

    /// Checks if points use perspective projection.
    ///
    /// # Returns
    /// `true` if points scale with distance (perspective), `false` for constant screen-space size
    #[inline]
    pub fn points_use_perspective(&self) -> bool {
        self.points_use_perspective
    }

    /// Checks if surface rendering is enabled for this object.
    ///
    /// # Returns
    /// `true` if surfaces are rendered, `false` if only wireframe/points are rendered
    #[inline]
    pub fn surface_rendering_active(&self) -> bool {
        self.draw_surface
    }

    /// Checks if backface culling is enabled for this object.
    ///
    /// # Returns
    /// `true` if backface culling is enabled
    #[inline]
    pub fn backface_culling_enabled(&self) -> bool {
        self.cull
    }

    /// Returns the integer segmentation/object id of this object.
    ///
    /// This id is what the segmentation auxiliary render output writes into the
    /// per-pixel mask. It defaults to a process-unique value and can be changed
    /// with [`Object3d::set_segmentation_id`]. The value `0` is reserved for the
    /// background (pixels not covered by any object).
    #[inline]
    pub fn segmentation_id(&self) -> u32 {
        self.segmentation_id
    }

    /// Returns a reference to user-defined data attached to this object.
    ///
    /// Use the `Any` trait's downcasting methods to recover the actual data type.
    ///
    /// # Returns
    /// A reference to the user data as `&dyn Any`
    #[inline]
    pub fn user_data(&self) -> &dyn Any {
        &*self.user_data
    }

    /// Whether this object is a skinned mesh (driven by a [`Skin3d`]).
    #[inline]
    pub fn has_skin(&self) -> bool {
        self.skin.is_some()
    }

    /// The skeletal skinning binding, if this is a skinned mesh.
    #[inline]
    pub fn skin(&self) -> Option<&Skin3d> {
        self.skin.as_ref()
    }

    /// Mutable access to the skinning binding (used to refresh the palette).
    #[inline]
    pub(crate) fn skin_mut(&mut self) -> Option<&mut Skin3d> {
        self.skin.as_mut()
    }

    /// Attaches a skeletal skinning binding, marking this object as skinned.
    #[inline]
    pub(crate) fn set_skin(&mut self, skin: Skin3d) {
        self.skin = Some(skin);
    }

    /// The current morph-target weights (one per target), or an empty slice when the
    /// mesh has no morph targets.
    #[inline]
    pub fn morph_weights(&self) -> &[f32] {
        &self.morph_weights
    }

    /// Sets the morph-target weights (one per target). Typically driven by an
    /// [`AnimationPlayer`](crate::scene::AnimationPlayer), but can be set manually to
    /// pose blend shapes. Takes effect on the next frame's deform update.
    #[inline]
    pub fn set_morph_weights(&mut self, weights: &[f32]) {
        self.morph_weights.clear();
        self.morph_weights.extend_from_slice(weights);
    }

    /// The number of morph targets this object expects (the length of its weight
    /// vector). `0` when the mesh is not morphable.
    #[inline]
    pub fn morph_target_count(&self) -> usize {
        self.morph_weights.len()
    }

    /// Whether this object is deformable (skinned and/or morphed) and therefore drawn
    /// with the deform pipeline.
    #[inline]
    pub(crate) fn is_deformable(&self) -> bool {
        self.skin.is_some() || !self.morph_weights.is_empty()
    }

    /// The per-frame deform bind group (group 4 color / group 2 shadow), or `None`
    /// when the object isn't deformable or the deform path is unavailable (web).
    #[inline]
    pub(crate) fn deform_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.deform.as_ref().and_then(|d| d.bind_group())
    }

    /// Refreshes this object's GPU deform state for the current frame: writes the
    /// control uniform (skin flag + current morph weights) and (re)builds the deform
    /// bind group over the palette + skin/morph storage buffers. Native-only; a no-op
    /// when the object isn't deformable. Called once per frame from
    /// [`SceneNode3d::update_deformations`](crate::scene::SceneNode3d::update_deformations)
    /// after the joint palette has been uploaded.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn update_deform(&mut self, mesh: &GpuMesh3d) {
        use crate::builtin::deform::{DeformControl, DeformGpu};

        // Skinning applies only once the palette has been uploaded this frame.
        let has_skin = mesh.has_skin_vertices()
            && self
                .skin
                .as_ref()
                .and_then(|s| s.palette_buffer())
                .is_some();
        let has_morph = mesh.has_morph() && !self.morph_weights.is_empty();
        if !has_skin && !has_morph {
            return;
        }

        // `ensure_*_on_gpu` take `&self` (interior RwLock mutability), so the skin and
        // morph buffer borrows can be held simultaneously to build one bind group.
        let (joints, weights) = match has_skin {
            true => mesh
                .ensure_skin_on_gpu()
                .map_or((None, None), |(j, w)| (Some(j), Some(w))),
            false => (None, None),
        };
        let (morph_pos, morph_nrm) = match has_morph {
            true => mesh
                .ensure_morph_on_gpu()
                .map_or((None, None), |(p, n)| (Some(p), n)),
            false => (None, None),
        };
        let palette = if has_skin {
            self.skin.as_ref().and_then(|s| s.palette_buffer())
        } else {
            None
        };

        let mut ctrl = DeformControl::default();
        ctrl.set_weights(if has_morph { &self.morph_weights } else { &[] });
        ctrl.num_vertices = mesh.morph_vertex_count() as u32;
        ctrl.has_skin = has_skin as u32;
        ctrl.has_morph_normals = (has_morph && mesh.has_morph_normals()) as u32;

        let deform = self.deform.get_or_insert_with(DeformGpu::new);
        deform.update(&ctrl, palette, joints, weights, morph_pos, morph_nrm);
    }

    /// Returns (lazily building) the albedo-texture bind group for the shadow
    /// transmittance pass, for the given group layout. Rebuilt only when the
    /// object's texture changes.
    pub(crate) fn shadow_tex_bind_group(
        &mut self,
        layout: &wgpu::BindGroupLayout,
    ) -> &wgpu::BindGroup {
        let ptr = Arc::as_ptr(&self.texture) as usize;
        if self.shadow_tex_bind_group.is_none() || self.cached_shadow_tex_ptr != ptr {
            let ctxt = Context::get();
            self.shadow_tex_bind_group = Some(ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow_tex_bind_group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.texture.sampler),
                    },
                ],
            }));
            self.cached_shadow_tex_ptr = ptr;
        }
        self.shadow_tex_bind_group.as_ref().unwrap()
    }

    /// Returns the metallic factor of this object.
    ///
    /// # Returns
    /// Metallic factor in range [0.0, 1.0] where 0.0 is dielectric and 1.0 is metal
    #[inline]
    pub fn metallic(&self) -> f32 {
        self.metallic
    }

    /// Returns the roughness factor of this object.
    ///
    /// # Returns
    /// Roughness factor in range [0.0, 1.0] where 0.0 is smooth and 1.0 is rough
    #[inline]
    pub fn roughness(&self) -> f32 {
        self.roughness
    }

    /// Returns the emissive color of this object.
    ///
    /// # Returns
    /// RGBA emissive color with components typically in range [0.0, 1.0] or higher for HDR
    #[inline]
    pub fn emissive(&self) -> Color {
        self.emissive
    }

    /// Returns this object's alpha blending mode.
    #[inline]
    pub fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    /// Returns this object's render-layer bitmask.
    #[inline]
    pub fn render_layers(&self) -> u32 {
        self.render_layers
    }

    /// Returns the path-tracer BSDF model for this object.
    #[inline]
    pub fn bsdf(&self) -> Bsdf {
        self.bsdf
    }

    /// Returns the index of refraction (used by the glass/dielectric BSDF).
    #[inline]
    pub fn ior(&self) -> f32 {
        self.ior
    }

    /// Returns the transmission (specular-transmittance) factor in `[0, 1]`.
    #[inline]
    pub fn transmission(&self) -> f32 {
        self.transmission
    }

    /// Returns the specular tint color (multiplies the specular/conductor lobe).
    #[inline]
    pub fn specular_tint(&self) -> Color {
        self.specular_tint
    }

    /// Returns the subsurface/translucency factor in `[0, 1]`.
    #[inline]
    pub fn subsurface(&self) -> f32 {
        self.subsurface
    }

    /// Returns the subsurface scattering radius (world units).
    #[inline]
    pub fn subsurface_radius(&self) -> f32 {
        self.subsurface_radius
    }

    /// Returns the dielectric specular reflectance remap in `[0, 1]`.
    #[inline]
    pub fn reflectance(&self) -> f32 {
        self.reflectance
    }

    /// Returns this object's per-object SSR properties (`None` = opted out).
    #[inline]
    pub fn ssr(&self) -> Option<crate::renderer::SsrMaterial> {
        self.ssr
    }

    /// Returns this object's planar reflector, if any.
    #[inline]
    pub fn reflector(&self) -> Option<&crate::renderer::Reflector> {
        self.reflector.as_ref()
    }

    /// Mutable access to this object's planar reflector, if any.
    #[inline]
    pub fn reflector_mut(&mut self) -> Option<&mut crate::renderer::Reflector> {
        self.reflector.as_mut()
    }

    /// Returns the clearcoat layer strength in `[0, 1]`.
    #[inline]
    pub fn clearcoat(&self) -> f32 {
        self.clearcoat
    }

    /// Returns the clearcoat layer roughness in `[0, 1]`.
    #[inline]
    pub fn clearcoat_roughness(&self) -> f32 {
        self.clearcoat_roughness
    }

    /// Returns the anisotropy strength in `[-1, 1]`.
    #[inline]
    pub fn anisotropy(&self) -> f32 {
        self.anisotropy
    }

    /// Returns the anisotropy direction rotation around the normal, in radians.
    #[inline]
    pub fn anisotropy_rotation(&self) -> f32 {
        self.anisotropy_rotation
    }

    /// Returns a reference to this object's normal map texture.
    ///
    /// # Returns
    /// `Some` if a normal map is set, `None` otherwise
    #[inline]
    pub fn normal_map(&self) -> Option<&Arc<Texture>> {
        self.normal_map.as_ref()
    }

    /// Returns a reference to this object's metallic-roughness map texture.
    ///
    /// The texture follows glTF convention: B channel = metallic, G channel = roughness.
    ///
    /// # Returns
    /// `Some` if a metallic-roughness map is set, `None` otherwise
    #[inline]
    pub fn metallic_roughness_map(&self) -> Option<&Arc<Texture>> {
        self.metallic_roughness_map.as_ref()
    }

    /// Returns a reference to this object's ambient occlusion map texture.
    ///
    /// # Returns
    /// `Some` if an AO map is set, `None` otherwise
    #[inline]
    pub fn ao_map(&self) -> Option<&Arc<Texture>> {
        self.ao_map.as_ref()
    }

    /// Returns a reference to this object's emissive map texture.
    ///
    /// # Returns
    /// `Some` if an emissive map is set, `None` otherwise
    #[inline]
    pub fn emissive_map(&self) -> Option<&Arc<Texture>> {
        self.emissive_map.as_ref()
    }

    /// Returns a reference to this object's height/displacement map (parallax).
    #[inline]
    pub fn height_map(&self) -> Option<&Arc<Texture>> {
        self.height_map.as_ref()
    }

    /// Returns the parallax displacement scale (`0` disables parallax mapping).
    #[inline]
    pub fn parallax_scale(&self) -> f32 {
        self.parallax_scale
    }

    /// Returns the maximum number of parallax search layers.
    #[inline]
    pub fn parallax_layers(&self) -> f32 {
        self.parallax_layers
    }

    /// Returns the parallax search method.
    #[inline]
    pub fn parallax_method(&self) -> ParallaxMethod {
        self.parallax_method
    }
}

/// Data for a single instance in instanced rendering.
///
/// When rendering multiple copies of the same mesh with different transformations
/// and colors (instancing), each instance is defined by this data.
///
/// # Example
/// ```no_run
/// # use kiss3d::scene::InstanceData3d;
/// # use kiss3d::color::{Color, RED, LIME, YELLOW};
/// # use glamx::{Vec3, Mat3};
/// let instance = InstanceData3d {
///     position: Vec3::new(1.0, 0.0, 0.0),
///     deformation: Mat3::IDENTITY,
///     color: RED,
///     lines_color: Some(LIME),  // Green wireframe
///     lines_width: Some(2.0),  // 2px wireframe
///     points_color: Some(YELLOW),  // Yellow points
///     points_size: Some(5.0),  // 5px points
/// };
/// ```
pub struct InstanceData3d {
    /// The position offset for this instance.
    pub position: Vec3,
    /// The 3x3 deformation matrix (scale, rotation, shear) for this instance.
    pub deformation: Mat3,
    /// The RGBA color for this instance.
    pub color: Color,
    /// The RGBA wireframe color for this instance. None = use object's wireframe color.
    pub lines_color: Option<Color>,
    /// The wireframe line width in pixels for this instance. None = use object's wireframe width.
    pub lines_width: Option<f32>,
    /// The RGBA point color for this instance. None = use object's point color.
    pub points_color: Option<Color>,
    /// The point size in pixels for this instance. None = use object's point size.
    pub points_size: Option<f32>,
}

impl Default for InstanceData3d {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            deformation: Mat3::IDENTITY,
            color: crate::color::WHITE,
            lines_color: None,  // Use object's wireframe color
            lines_width: None,  // Use object's wireframe width
            points_color: None, // Use object's point color
            points_size: None,  // Use object's point size
        }
    }
}

/// Sentinel value for lines_width indicating "use object's value".
pub const LINES_WIDTH_USE_OBJECT: f32 = -1.0;
/// Sentinel value for lines_color indicating "use object's value" (alpha = 0).
pub const LINES_COLOR_USE_OBJECT: Color = Color::new(0.0, 0.0, 0.0, 0.0);
/// Sentinel value for points_size indicating "use object's value".
pub const POINTS_SIZE_USE_OBJECT: f32 = -1.0;
/// Sentinel value for points_color indicating "use object's value" (alpha = 0).
pub const POINTS_COLOR_USE_OBJECT: Color = Color::new(0.0, 0.0, 0.0, 0.0);

/// GPU buffer for instanced rendering data.
///
/// Contains GPU-allocated buffers for positions, deformations, colors,
/// wireframe settings, and point settings of all instances to be rendered.
pub struct InstancesBuffer3d {
    /// GPU buffer of instance positions.
    pub positions: GPUVec<Vec3>,
    /// GPU buffer of instance deformation matrices (stored as 3 column vectors).
    pub deformations: GPUVec<Vec3>,
    /// GPU buffer of instance colors.
    pub colors: GPUVec<[f32; 4]>,
    /// GPU buffer of instance wireframe colors. Alpha = 0 means use object's color.
    pub lines_colors: GPUVec<[f32; 4]>,
    /// GPU buffer of instance wireframe line widths. Negative means use object's width.
    pub lines_widths: GPUVec<f32>,
    /// GPU buffer of instance point colors. Alpha = 0 means use object's color.
    pub points_colors: GPUVec<[f32; 4]>,
    /// GPU buffer of instance point sizes. Negative means use object's size.
    pub points_sizes: GPUVec<f32>,
}

/// Helper function to convert Color to [f32; 4] for GPU buffers.
#[inline]
pub(crate) fn color_to_array(color: Color) -> [f32; 4] {
    [color.r, color.g, color.b, color.a]
}

impl Default for InstancesBuffer3d {
    fn default() -> Self {
        InstancesBuffer3d {
            positions: GPUVec::new(
                vec![Vec3::ZERO],
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            deformations: GPUVec::new(
                vec![Vec3::X, Vec3::Y, Vec3::Z],
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            colors: GPUVec::new(
                vec![[1.0; 4]],
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            lines_colors: GPUVec::new(
                vec![color_to_array(LINES_COLOR_USE_OBJECT)], // Use object's wireframe color by default
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            lines_widths: GPUVec::new(
                vec![LINES_WIDTH_USE_OBJECT], // Use object's wireframe width by default
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            points_colors: GPUVec::new(
                vec![color_to_array(POINTS_COLOR_USE_OBJECT)], // Use object's point color by default
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            points_sizes: GPUVec::new(
                vec![POINTS_SIZE_USE_OBJECT], // Use object's point size by default
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
        }
    }
}

impl InstancesBuffer3d {
    /// Checks if there are no instances.
    ///
    /// # Returns
    /// `true` if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of instances in the buffer.
    ///
    /// # Returns
    /// The number of instances
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Checks if any instance has a specific wireframe width set (not using object's default).
    ///
    /// # Returns
    /// `true` if at least one instance has a specific wireframe width (>= 0)
    pub fn any_instance_has_wireframe(&self) -> bool {
        if let Some(widths) = self.lines_widths.data() {
            widths.iter().any(|&w| w >= 0.0)
        } else {
            false
        }
    }

    /// Checks if all instances use the object's wireframe width (all have sentinel value).
    ///
    /// # Returns
    /// `true` if all instances use object's wireframe width
    pub fn all_use_object_wireframe(&self) -> bool {
        if let Some(widths) = self.lines_widths.data() {
            widths.iter().all(|&w| w < 0.0)
        } else {
            true
        }
    }
}

/// A renderable 3D object in the scene.
///
/// `Object` combines a mesh with rendering properties (material, texture, color).
/// It's the primary interface for manipulating an object's appearance and geometry.
pub struct Object3d {
    // TODO: should Mesh and Object be merged?
    // (thus removing the need of ObjectData at all.)
    data: ObjectData3d,
    instances: Rc<RefCell<InstancesBuffer3d>>,
    mesh: Rc<RefCell<GpuMesh3d>>,
    /// Per-object GPU data for the material (uniform buffers, etc.)
    gpu_data: Box<dyn GpuData>,
}

impl Object3d {
    #[doc(hidden)]
    pub fn new(
        mesh: Rc<RefCell<GpuMesh3d>>,
        color: Color,
        texture: Arc<Texture>,
        material: Rc<RefCell<Box<dyn Material3d + 'static>>>,
    ) -> Object3d {
        // Create per-object GPU data from the material
        let gpu_data = material.borrow().create_gpu_data();

        let user_data = ();
        let data = ObjectData3d {
            color,
            lines_color: None,
            points_color: None,
            texture,
            wlines: 0.0,
            wpoints: 0.0,
            lines_use_perspective: true,
            points_use_perspective: true,
            draw_surface: true,
            cull: true,
            segmentation_id: next_segmentation_id(),
            material,
            user_data: Box::new(user_data),
            render_layers: 1, // layer 0

            // PBR defaults (backward compatible with Blinn-Phong appearance)
            metallic: 0.0,
            roughness: 0.5,
            emissive: crate::color::BLACK,
            alpha_mode: AlphaMode::default(),
            // Path-tracer BSDF defaults: opaque dielectric.
            bsdf: Bsdf::Opaque,
            ior: 1.5,
            transmission: 0.0,
            specular_tint: crate::color::WHITE,
            subsurface: 0.0,
            subsurface_radius: 0.0,
            reflectance: 0.5,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            ssr: Some(crate::renderer::SsrMaterial::default()),
            reflector: None,
            normal_map: None,
            metallic_roughness_map: None,
            ao_map: None,
            emissive_map: None,
            height_map: None,
            parallax_scale: 0.1,
            parallax_layers: 16.0,
            parallax_method: ParallaxMethod::Occlusion,
            skin: None,
            morph_weights: Vec::new(),
            deform: None,
            shadow_tex_bind_group: None,
            cached_shadow_tex_ptr: 0,
        };
        let instances = Rc::new(RefCell::new(InstancesBuffer3d::default()));

        Object3d {
            data,
            instances,
            mesh,
            gpu_data,
        }
    }

    #[doc(hidden)]
    pub fn prepare(
        &mut self,
        transform: Pose3,
        scale: Vec3,
        pass: usize,
        camera: &mut dyn Camera3d,
        lights: &LightCollection,
        viewport_width: u32,
        viewport_height: u32,
    ) {
        self.data.material.borrow_mut().prepare(
            pass,
            transform,
            scale,
            camera,
            lights,
            &self.data,
            &mut *self.gpu_data,
            viewport_width,
            viewport_height,
        );
    }

    #[doc(hidden)]
    pub fn render(
        &mut self,
        transform: Pose3,
        scale: Vec3,
        pass: usize,
        camera: &mut dyn Camera3d,
        lights: &LightCollection,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext,
    ) {
        // Skip materials that don't participate in the transparent (OIT) pass — its
        // render targets differ from the opaque pass, so a material whose pipeline
        // targets the opaque format would be incompatible there. Built-in materials
        // that implement OIT opt in via `renders_in_transparent_phase`.
        if context.phase == RenderPhase::Transparent
            && !self.data.material.borrow().renders_in_transparent_phase()
        {
            return;
        }
        // Render-layer filtering: skip objects the camera's layer mask excludes.
        if self.data.render_layers & context.render_layers == 0 {
            return;
        }
        self.data.material.borrow_mut().render(
            pass,
            transform,
            scale,
            camera,
            lights,
            &self.data,
            &mut self.mesh.borrow_mut(),
            &mut self.instances.borrow_mut(),
            &mut *self.gpu_data,
            render_pass,
            context,
        );
    }

    /// Whether this object contributes surface geometry to the shadow pre-pass.
    #[doc(hidden)]
    pub fn casts_shadows(&self) -> bool {
        self.data.surface_rendering_active()
    }

    /// Draws this object's surface geometry into the shadow depth pass.
    ///
    /// Sets `base_pipeline` (or, for a deformable caster when `deform_pipeline` is
    /// supplied, the deformed depth pipeline plus the object's deform bind group at
    /// group 2 — joint palette + skin streams + morph deltas) and binds the
    /// per-object model group (group 1) via `model_offset`. Group 0 (view) is bound
    /// once by the caller and persists across the pipeline switch (compatible
    /// layout). No material state is touched.
    /// `transmittance_tex` is `Some(layout)` only for the colored-transmittance
    /// pass; then this also binds the object's albedo texture (so the shadow tint
    /// follows the texture) and the UV stream, at the group/slot the transmittance
    /// pipeline expects (texture group 2 + UV slot 3 plain; group 3 + slot 3
    /// deformed). The depth pass passes `None` and binds neither.
    #[doc(hidden)]
    pub fn render_depth_only(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'_>,
        base_pipeline: &wgpu::RenderPipeline,
        deform_pipeline: Option<&wgpu::RenderPipeline>,
        transmittance_tex: Option<&wgpu::BindGroupLayout>,
        model_bind_group: &wgpu::BindGroup,
        model_offset: u32,
    ) {
        if !self.data.surface_rendering_active() {
            return;
        }

        let mesh = self.mesh.borrow();
        let mut instances = self.instances.borrow_mut();

        let num_instances = instances.len();
        instances.positions.load_to_gpu();
        instances.deformations.load_to_gpu();

        mesh.coords().write().unwrap().load_to_gpu();
        mesh.faces().write().unwrap().load_to_gpu();

        let coords_buffer = mesh.coords().read().unwrap();
        let faces_buffer = mesh.faces().read().unwrap();

        let coords_buf = match coords_buffer.buffer() {
            Some(b) => b,
            None => return,
        };
        let faces_buf = match faces_buffer.buffer() {
            Some(b) => b,
            None => return,
        };
        let inst_positions_buf = match instances.positions.buffer() {
            Some(b) => b,
            None => return,
        };
        let inst_deformations_buf = match instances.deformations.buffer() {
            Some(b) => b,
            None => return,
        };

        // Use the deformed depth pipeline only when it exists (native) and the
        // object's deform bind group was built this frame (skinned and/or morphed);
        // otherwise the base pipeline draws the rest shape.
        let use_deform = deform_pipeline.is_some() && self.data.deform_bind_group().is_some();

        // The transmittance pass also samples the albedo texture (UVs needed).
        let uvs_guard;
        let uvs_buf: Option<&wgpu::Buffer> = if transmittance_tex.is_some() {
            mesh.uvs().write().unwrap().load_to_gpu();
            uvs_guard = mesh.uvs().read().unwrap();
            uvs_guard.buffer()
        } else {
            None
        };

        if use_deform {
            render_pass.set_pipeline(deform_pipeline.unwrap());
            render_pass.set_bind_group(1, model_bind_group, &[model_offset]);
            // Group 2: the object's deform bind group (built in update_deformations).
            render_pass.set_bind_group(2, self.data.deform_bind_group().unwrap(), &[]);
        } else {
            render_pass.set_pipeline(base_pipeline);
            render_pass.set_bind_group(1, model_bind_group, &[model_offset]);
        }

        // Albedo texture for the transmittance pass (group 3 deformed / group 2 not).
        if let (Some(tex_layout), Some(uvs)) = (transmittance_tex, uvs_buf) {
            let tex_group = if use_deform { 3 } else { 2 };
            let tex_bg = self.data.shadow_tex_bind_group(tex_layout);
            render_pass.set_bind_group(tex_group, tex_bg, &[]);
            // UV stream at slot 3 (the deformed layout has no joints/weights buffers).
            render_pass.set_vertex_buffer(3, uvs.slice(..));
        }

        render_pass.set_vertex_buffer(0, coords_buf.slice(..));
        render_pass.set_vertex_buffer(1, inst_positions_buf.slice(..));
        render_pass.set_vertex_buffer(2, inst_deformations_buf.slice(..));
        render_pass.set_index_buffer(faces_buf.slice(..), VERTEX_INDEX_FORMAT);
        render_pass.draw_indexed(0..mesh.num_indices(), 0, 0..num_instances as u32);
    }

    /// Gets the data of this object.
    #[inline]
    pub fn data(&self) -> &ObjectData3d {
        &self.data
    }

    /// Gets the data of this object.
    #[inline]
    pub fn data_mut(&mut self) -> &mut ObjectData3d {
        &mut self.data
    }

    /// Whether this object is a skinned mesh.
    #[inline]
    pub fn has_skin(&self) -> bool {
        self.data.has_skin()
    }

    /// Attaches a skeletal skinning binding (used by the glTF loader).
    #[inline]
    pub(crate) fn set_skin(&mut self, skin: Skin3d) {
        self.data.set_skin(skin);
    }

    /// Gets the instances of this object.
    #[inline]
    pub fn instances(&self) -> &Rc<RefCell<InstancesBuffer3d>> {
        &self.instances
    }

    pub fn set_instances(&mut self, instances: &[InstanceData3d]) {
        let mut pos_data: Vec<_> = self
            .instances
            .borrow_mut()
            .positions
            .data_mut()
            .take()
            .unwrap_or_default();
        let mut col_data: Vec<_> = self
            .instances
            .borrow_mut()
            .colors
            .data_mut()
            .take()
            .unwrap_or_default();
        let mut def_data: Vec<_> = self
            .instances
            .borrow_mut()
            .deformations
            .data_mut()
            .take()
            .unwrap_or_default();
        let mut lines_col_data: Vec<_> = self
            .instances
            .borrow_mut()
            .lines_colors
            .data_mut()
            .take()
            .unwrap_or_default();
        let mut lines_width_data: Vec<_> = self
            .instances
            .borrow_mut()
            .lines_widths
            .data_mut()
            .take()
            .unwrap_or_default();
        let mut points_col_data: Vec<_> = self
            .instances
            .borrow_mut()
            .points_colors
            .data_mut()
            .take()
            .unwrap_or_default();
        let mut points_size_data: Vec<_> = self
            .instances
            .borrow_mut()
            .points_sizes
            .data_mut()
            .take()
            .unwrap_or_default();

        pos_data.clear();
        col_data.clear();
        def_data.clear();
        lines_col_data.clear();
        lines_width_data.clear();
        points_col_data.clear();
        points_size_data.clear();

        pos_data.extend(instances.iter().map(|i| i.position));
        col_data.extend(instances.iter().map(|i| color_to_array(i.color)));
        def_data.extend(instances.iter().flat_map(|i| {
            [
                i.deformation.x_axis,
                i.deformation.y_axis,
                i.deformation.z_axis,
            ]
        }));
        lines_col_data.extend(
            instances
                .iter()
                .map(|i| color_to_array(i.lines_color.unwrap_or(LINES_COLOR_USE_OBJECT))),
        );
        lines_width_data.extend(
            instances
                .iter()
                .map(|i| i.lines_width.unwrap_or(LINES_WIDTH_USE_OBJECT)),
        );
        points_col_data.extend(
            instances
                .iter()
                .map(|i| color_to_array(i.points_color.unwrap_or(POINTS_COLOR_USE_OBJECT))),
        );
        points_size_data.extend(
            instances
                .iter()
                .map(|i| i.points_size.unwrap_or(POINTS_SIZE_USE_OBJECT)),
        );

        *self.instances.borrow_mut().positions.data_mut() = Some(pos_data);
        *self.instances.borrow_mut().colors.data_mut() = Some(col_data);
        *self.instances.borrow_mut().deformations.data_mut() = Some(def_data);
        *self.instances.borrow_mut().lines_colors.data_mut() = Some(lines_col_data);
        *self.instances.borrow_mut().lines_widths.data_mut() = Some(lines_width_data);
        *self.instances.borrow_mut().points_colors.data_mut() = Some(points_col_data);
        *self.instances.borrow_mut().points_sizes.data_mut() = Some(points_size_data);
    }

    /// Enables or disables backface culling for this object.
    #[inline]
    pub fn enable_backface_culling(&mut self, active: bool) {
        self.data.cull = active;
    }

    /// Attaches user-defined data to this object.
    #[inline]
    pub fn set_user_data(&mut self, user_data: Box<dyn Any + 'static>) {
        self.data.user_data = user_data;
    }

    /// Sets the integer segmentation/object id of this object.
    ///
    /// This id is written by the segmentation auxiliary render output. Assigning
    /// the same id to several objects groups them into a single segmentation
    /// mask (e.g. all parts of one robot link). Avoid `0`, which is reserved for
    /// the background.
    #[inline]
    pub fn set_segmentation_id(&mut self, id: u32) {
        self.data.segmentation_id = id;
    }

    /// Returns the integer segmentation/object id of this object.
    #[inline]
    pub fn segmentation_id(&self) -> u32 {
        self.data.segmentation_id
    }

    /// Gets the material of this object.
    #[inline]
    pub fn material(&self) -> Rc<RefCell<Box<dyn Material3d + 'static>>> {
        self.data.material.clone()
    }

    /// Sets the material of this object.
    #[inline]
    pub fn set_material(&mut self, material: Rc<RefCell<Box<dyn Material3d + 'static>>>) {
        // Create new GPU data for the new material
        self.gpu_data = material.borrow().create_gpu_data();
        self.data.material = material;
    }

    /// Sets the width of the lines drawn for this object.
    ///
    /// If `use_perspective` is true, the width is in world units and scales with distance.
    /// If `use_perspective` is false, the width is in screen pixels and stays constant.
    #[inline]
    pub fn set_lines_width(&mut self, width: f32, use_perspective: bool) {
        self.data.wlines = width;
        self.data.lines_use_perspective = use_perspective;
    }

    /// Returns the width of the lines drawn for this object.
    #[inline]
    pub fn lines_width(&self) -> f32 {
        self.data.wlines
    }

    /// Sets the color of the lines drawn for this object.
    #[inline]
    pub fn set_lines_color(&mut self, color: Option<Color>) {
        self.data.lines_color = color
    }

    /// Returns the color of the lines drawn for this object.
    #[inline]
    pub fn lines_color(&self) -> Option<Color> {
        self.data.lines_color
    }

    /// Sets the size of the points drawn for this object.
    ///
    /// If `use_perspective` is true, the size is in world units and scales with distance.
    /// If `use_perspective` is false, the size is in screen pixels and stays constant.
    #[inline]
    pub fn set_points_size(&mut self, size: f32, use_perspective: bool) {
        self.data.wpoints = size;
        self.data.points_use_perspective = use_perspective;
    }

    /// Returns the size of the points drawn for this object.
    #[inline]
    pub fn points_size(&self) -> f32 {
        self.data.wpoints
    }

    /// Sets the color of the points drawn for this object.
    #[inline]
    pub fn set_points_color(&mut self, color: Option<Color>) {
        self.data.points_color = color
    }

    /// Returns the color of the points drawn for this object.
    #[inline]
    pub fn points_color(&self) -> Option<Color> {
        self.data.points_color
    }

    /// Activate or deactivate the rendering of this object surface.
    #[inline]
    pub fn set_surface_rendering_activation(&mut self, active: bool) {
        self.data.draw_surface = active
    }

    /// Activate or deactivate the rendering of this object surface.
    #[inline]
    pub fn surface_rendering_activation(&self) -> bool {
        self.data.draw_surface
    }

    /// This object's mesh.
    #[inline]
    pub fn mesh(&self) -> &Rc<RefCell<GpuMesh3d>> {
        &self.mesh
    }

    /// Mutably access the object's vertices.
    #[inline(always)]
    pub fn modify_vertices<F: FnMut(&mut Vec<Vec3>)>(&mut self, f: &mut F) {
        let bmesh = self.mesh.borrow_mut();
        let _ = bmesh.coords().write().unwrap().data_mut().as_mut().map(f);
    }

    /// Access the object's vertices.
    #[inline(always)]
    pub fn read_vertices<F: FnMut(&[Vec3])>(&self, f: &mut F) {
        let bmesh = self.mesh.borrow();
        let _ = bmesh
            .coords()
            .read()
            .unwrap()
            .data()
            .as_ref()
            .map(|coords| f(&coords[..]));
    }

    /// Recomputes the normals of this object's mesh.
    #[inline]
    pub fn recompute_normals(&mut self) {
        self.mesh.borrow_mut().recompute_normals();
    }

    /// Mutably access the object's normals.
    #[inline(always)]
    pub fn modify_normals<F: FnMut(&mut Vec<Vec3>)>(&mut self, f: &mut F) {
        let bmesh = self.mesh.borrow_mut();
        let _ = bmesh.normals().write().unwrap().data_mut().as_mut().map(f);
    }

    /// Access the object's normals.
    #[inline(always)]
    pub fn read_normals<F: FnMut(&[Vec3])>(&self, f: &mut F) {
        let bmesh = self.mesh.borrow();
        let _ = bmesh
            .normals()
            .read()
            .unwrap()
            .data()
            .as_ref()
            .map(|normals| f(&normals[..]));
    }

    /// Mutably access the object's faces.
    #[inline(always)]
    pub fn modify_faces<F: FnMut(&mut Vec<[VertexIndex; 3]>)>(&mut self, f: &mut F) {
        let bmesh = self.mesh.borrow_mut();
        let _ = bmesh.faces().write().unwrap().data_mut().as_mut().map(f);
    }

    /// Access the object's faces.
    #[inline(always)]
    pub fn read_faces<F: FnMut(&[[VertexIndex; 3]])>(&self, f: &mut F) {
        let bmesh = self.mesh.borrow();
        let _ = bmesh
            .faces()
            .read()
            .unwrap()
            .data()
            .as_ref()
            .map(|faces| f(&faces[..]));
    }

    /// Mutably access the object's texture coordinates.
    #[inline(always)]
    pub fn modify_uvs<F: FnMut(&mut Vec<Vec2>)>(&mut self, f: &mut F) {
        let bmesh = self.mesh.borrow_mut();
        let _ = bmesh.uvs().write().unwrap().data_mut().as_mut().map(f);
    }

    /// Access the object's texture coordinates.
    #[inline(always)]
    pub fn read_uvs<F: FnMut(&[Vec2])>(&self, f: &mut F) {
        let bmesh = self.mesh.borrow();
        let _ = bmesh
            .uvs()
            .read()
            .unwrap()
            .data()
            .as_ref()
            .map(|uvs| f(&uvs[..]));
    }

    /// Sets the color of the object.
    ///
    /// Colors components must be on the range `[0.0, 1.0]`.
    #[inline]
    pub fn set_color(&mut self, color: Color) {
        self.data.color = color;
    }

    /// Sets the texture of the object.
    ///
    /// The texture is loaded from a file and registered by the global `TextureManager`.
    ///
    /// # Arguments
    ///   * `path` - relative path of the texture on the disk
    #[inline]
    pub fn set_texture_from_file(&mut self, path: &Path, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));

        self.set_texture(texture)
    }

    /// Sets the texture of the object.
    ///
    /// The texture must already have been registered as `name`.
    #[inline]
    pub fn set_texture_with_name(&mut self, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| {
            tm.get(name).unwrap_or_else(|| {
                panic!("Invalid attempt to use the unregistered texture: {}", name)
            })
        });

        self.set_texture(texture)
    }

    /// Sets the texture of the object.
    #[inline]
    pub fn set_texture(&mut self, texture: Arc<Texture>) {
        self.data.texture = texture
    }

    // === PBR Material Properties ===

    /// Sets the metallic factor of this object.
    ///
    /// # Arguments
    /// * `metallic` - Metallic factor clamped to [0.0, 1.0] where 0.0 is dielectric and 1.0 is metal
    #[inline]
    pub fn set_metallic(&mut self, metallic: f32) {
        self.data.metallic = metallic.clamp(0.0, 1.0);
    }

    /// Sets this object's screen-space-reflection properties.
    ///
    /// `Some(SsrMaterial { .. })` makes the object receive SSR with those
    /// properties (per-object intensity, infinite-thick, Fresnel and distance
    /// attenuation); `None` opts the object out of SSR entirely. Objects receive
    /// SSR by default. Only effective while SSR is enabled on the window
    /// ([`Window::set_ssr_enabled`](crate::window::Window::set_ssr_enabled)); the
    /// global march-quality knobs live in
    /// [`Window::ssr_settings_mut`](crate::window::Window::ssr_settings_mut).
    #[inline]
    pub fn set_ssr(&mut self, ssr: Option<crate::renderer::SsrMaterial>) {
        self.data.ssr = ssr;
    }

    /// Makes this surface a planar reflector (mirror).
    ///
    /// `Some(Reflector { .. })` renders the reflected scene into the reflector's
    /// texture each frame and blends it over this object's normal PBR shading (so a
    /// reflective floor can still have its base color, textures, roughness, etc.);
    /// `None` (the default) is a normal surface. The reflection plane is this
    /// object's world transform applied to its origin and the reflector's
    /// object-space normal (default +Z, see [`Reflector::with_local_normal`]).
    /// Best on a flat surface (e.g. a [`SceneNode3d::add_quad`] quad — or use the
    /// [`SceneNode3d::add_reflector`] convenience).
    ///
    /// [`Reflector`]: crate::renderer::Reflector
    /// [`Reflector::with_local_normal`]: crate::renderer::Reflector::with_local_normal
    /// [`SceneNode3d::add_quad`]: crate::scene::SceneNode3d::add_quad
    /// [`SceneNode3d::add_reflector`]: crate::scene::SceneNode3d::add_reflector
    #[inline]
    pub fn set_reflector(&mut self, reflector: Option<crate::renderer::Reflector>) {
        self.data.reflector = reflector;
    }

    /// This object's planar reflector, if any.
    #[inline]
    pub fn reflector(&self) -> Option<&crate::renderer::Reflector> {
        self.data.reflector.as_ref()
    }

    /// Mutable access to this object's planar reflector, if any.
    #[inline]
    pub fn reflector_mut(&mut self) -> Option<&mut crate::renderer::Reflector> {
        self.data.reflector.as_mut()
    }

    /// Sets the roughness factor of this object.
    ///
    /// # Arguments
    /// * `roughness` - Roughness factor clamped to [0.0, 1.0] where 0.0 is smooth and 1.0 is rough
    #[inline]
    pub fn set_roughness(&mut self, roughness: f32) {
        self.data.roughness = roughness.clamp(0.0, 1.0);
    }

    /// Sets the emissive color of this object.
    ///
    /// Objects with emissive color appear to glow. Values above 1.0 can be used for HDR.
    ///
    /// # Arguments
    /// * `color` - RGBA emissive color
    #[inline]
    pub fn set_emissive(&mut self, color: Color) {
        self.data.emissive = color;
    }

    /// Sets this object's render-layer bitmask.
    ///
    /// A camera renders this object only when its layer mask (see
    /// [`Camera3d::render_layers`](crate::camera::Camera3d::render_layers))
    /// shares at least one bit with `layers`. Objects start on layer 0 (mask
    /// `1`). Use this to show different objects to different cameras, e.g. an
    /// overlay/editor layer.
    #[inline]
    pub fn set_render_layers(&mut self, layers: u32) {
        self.data.render_layers = layers;
    }

    /// Returns this object's render-layer bitmask.
    #[inline]
    pub fn render_layers(&self) -> u32 {
        self.data.render_layers
    }

    /// Sets how this object's alpha is interpreted (see [`AlphaMode`]).
    ///
    /// - [`AlphaMode::Opaque`] ignores alpha (always opaque).
    /// - [`AlphaMode::Mask`] discards fragments below the cutoff (cutout).
    /// - [`AlphaMode::Blend`] (default) / [`AlphaMode::Premultiplied`] route the
    ///   surface through the order-independent transparency pass when its color
    ///   alpha is below `1.0`.
    #[inline]
    pub fn set_alpha_mode(&mut self, alpha_mode: AlphaMode) {
        self.data.alpha_mode = alpha_mode;
    }

    // === Path-tracer BSDF Properties ===

    /// Selects the path-tracer BSDF model for this object (rasterizer unaffected).
    #[inline]
    pub fn set_bsdf(&mut self, bsdf: Bsdf) {
        self.data.bsdf = bsdf;
    }

    /// Sets the index of refraction used by the glass/dielectric BSDF.
    ///
    /// Typical values: 1.0 (air), 1.33 (water), 1.5 (glass), 2.4 (diamond).
    #[inline]
    pub fn set_ior(&mut self, ior: f32) {
        self.data.ior = ior.max(1.0);
    }

    /// Sets the transmission (specular-transmittance) factor in `[0, 1]`.
    ///
    /// A value above zero lets light pass through the surface (refraction for the
    /// glass BSDF, diffuse/specular transmission otherwise).
    #[inline]
    pub fn set_transmission(&mut self, transmission: f32) {
        self.data.transmission = transmission.clamp(0.0, 1.0);
    }

    /// Sets the specular tint color, which multiplies the specular/conductor lobe.
    #[inline]
    pub fn set_specular_tint(&mut self, color: Color) {
        self.data.specular_tint = color;
    }

    /// Sets a cheap subsurface/translucency factor in `[0, 1]` and its radius.
    ///
    /// The factor blends the diffuse lobe toward a wrap-lit translucent look; the
    /// radius is reserved for a future diffusion approximation.
    #[inline]
    pub fn set_subsurface(&mut self, factor: f32, radius: f32) {
        self.data.subsurface = factor.clamp(0.0, 1.0);
        self.data.subsurface_radius = radius.max(0.0);
    }

    // === Extended PBR surface properties (rasterizer + path tracer) ===

    /// Sets the dielectric specular reflectance in `[0, 1]`.
    ///
    /// For non-metals the normal-incidence reflectance becomes `F0 = 0.16 *
    /// reflectance^2`. The default `0.5` reproduces the common `0.04`; raise it
    /// for shinier dielectrics (e.g. gemstones), lower it for matte surfaces.
    #[inline]
    pub fn set_reflectance(&mut self, reflectance: f32) {
        self.data.reflectance = reflectance.clamp(0.0, 1.0);
    }

    /// Sets the clearcoat layer strength and roughness, both in `[0, 1]`.
    ///
    /// Clearcoat adds a thin, smooth dielectric specular layer on top of the base
    /// material (car paint, lacquer, varnish). `strength` of `0` disables it.
    #[inline]
    pub fn set_clearcoat(&mut self, strength: f32, roughness: f32) {
        self.data.clearcoat = strength.clamp(0.0, 1.0);
        self.data.clearcoat_roughness = roughness.clamp(0.0, 1.0);
    }

    /// Sets the specular anisotropy: `strength` in `[-1, 1]` stretches the
    /// highlight along the tangent (positive) or bitangent (negative), and
    /// `rotation` (radians) rotates the anisotropy direction around the normal.
    ///
    /// Useful for brushed metal, hair, and vinyl records.
    #[inline]
    pub fn set_anisotropy(&mut self, strength: f32, rotation: f32) {
        self.data.anisotropy = strength.clamp(-1.0, 1.0);
        self.data.anisotropy_rotation = rotation;
    }

    // === PBR Texture Maps ===

    /// Sets the normal map texture from a file.
    ///
    /// Normal maps add surface detail without additional geometry.
    ///
    /// # Arguments
    /// * `path` - Path to the normal map image file
    /// * `name` - Name to register the texture under
    #[inline]
    pub fn set_normal_map_from_file(&mut self, path: &Path, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));
        self.set_normal_map(texture);
    }

    /// Sets the normal map texture.
    #[inline]
    pub fn set_normal_map(&mut self, texture: Arc<Texture>) {
        self.data.normal_map = Some(texture);
    }

    /// Clears the normal map.
    #[inline]
    pub fn clear_normal_map(&mut self) {
        self.data.normal_map = None;
    }

    /// Sets the metallic-roughness map texture from a file.
    ///
    /// Follows glTF convention: B channel = metallic, G channel = roughness.
    ///
    /// # Arguments
    /// * `path` - Path to the metallic-roughness map image file
    /// * `name` - Name to register the texture under
    #[inline]
    pub fn set_metallic_roughness_map_from_file(&mut self, path: &Path, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));
        self.set_metallic_roughness_map(texture);
    }

    /// Sets the metallic-roughness map texture.
    ///
    /// Follows glTF convention: B channel = metallic, G channel = roughness.
    #[inline]
    pub fn set_metallic_roughness_map(&mut self, texture: Arc<Texture>) {
        self.data.metallic_roughness_map = Some(texture);
    }

    /// Clears the metallic-roughness map.
    #[inline]
    pub fn clear_metallic_roughness_map(&mut self) {
        self.data.metallic_roughness_map = None;
    }

    /// Sets the ambient occlusion map texture from a file.
    ///
    /// AO maps add subtle shadows in crevices and corners.
    ///
    /// # Arguments
    /// * `path` - Path to the AO map image file
    /// * `name` - Name to register the texture under
    #[inline]
    pub fn set_ao_map_from_file(&mut self, path: &Path, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));
        self.set_ao_map(texture);
    }

    /// Sets the ambient occlusion map texture.
    #[inline]
    pub fn set_ao_map(&mut self, texture: Arc<Texture>) {
        self.data.ao_map = Some(texture);
    }

    /// Clears the ambient occlusion map.
    #[inline]
    pub fn clear_ao_map(&mut self) {
        self.data.ao_map = None;
    }

    /// Sets the emissive map texture from a file.
    ///
    /// The emissive map is multiplied by the emissive color.
    ///
    /// # Arguments
    /// * `path` - Path to the emissive map image file
    /// * `name` - Name to register the texture under
    #[inline]
    pub fn set_emissive_map_from_file(&mut self, path: &Path, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));
        self.set_emissive_map(texture);
    }

    /// Sets the emissive map texture.
    #[inline]
    pub fn set_emissive_map(&mut self, texture: Arc<Texture>) {
        self.data.emissive_map = Some(texture);
    }

    /// Clears the emissive map.
    #[inline]
    pub fn clear_emissive_map(&mut self) {
        self.data.emissive_map = None;
    }

    /// Sets the height/displacement map used for parallax mapping from a file.
    ///
    /// The map is grayscale (brighter = higher). Parallax shifts the texture
    /// coordinates per fragment along the view direction to fake surface depth.
    #[inline]
    pub fn set_height_map_from_file(&mut self, path: &Path, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));
        self.set_height_map(texture);
    }

    /// Sets the height/displacement map used for parallax mapping.
    #[inline]
    pub fn set_height_map(&mut self, texture: Arc<Texture>) {
        self.data.height_map = Some(texture);
    }

    /// Clears the height map (disables parallax mapping for this object).
    #[inline]
    pub fn clear_height_map(&mut self) {
        self.data.height_map = None;
    }

    /// Sets the parallax displacement scale (surface depth in UV units). `0`
    /// disables parallax even when a height map is set; typical values are small
    /// (e.g. `0.03`–`0.1`).
    #[inline]
    pub fn set_parallax_scale(&mut self, scale: f32) {
        self.data.parallax_scale = scale.max(0.0);
    }

    /// Sets the maximum number of parallax search layers (clamped to `[1, 64]`).
    /// More layers give sharper relief at steep angles at a higher cost; a low
    /// count (1–2) gives a chunky, thick-sliced look.
    #[inline]
    pub fn set_parallax_layers(&mut self, layers: f32) {
        self.data.parallax_layers = layers.clamp(1.0, 64.0);
    }

    /// Sets the parallax search method (occlusion vs relief).
    #[inline]
    pub fn set_parallax_method(&mut self, method: ParallaxMethod) {
        self.data.parallax_method = method;
    }
}
