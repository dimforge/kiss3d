//! CPU-side extraction of the scene graph into flat, GPU-ready ray-tracing data.
//!
//! A single walk of the scene graph bakes every object's world transform into its
//! geometry and produces flat arrays of vertices, triangles, materials and lights,
//! together with a content hash used to detect when the GPU scene must be rebuilt.

use bytemuck::{Pod, Zeroable};
use glamx::Vec3;

use crate::light::{CollectedLight, LightCollection, LightType};
use crate::scene::SceneNode3d;

/// Light type tag matching the WGSL `RtLight.light_type` convention.
pub const RT_LIGHT_POINT: u32 = 0;
/// Light type tag matching the WGSL `RtLight.light_type` convention.
pub const RT_LIGHT_DIRECTIONAL: u32 = 1;
/// Light type tag matching the WGSL `RtLight.light_type` convention.
pub const RT_LIGHT_SPOT: u32 = 2;

/// A single vertex (position + shading normal), padded to a 32-byte std430 layout.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct RtVertex {
    /// World-space position.
    pub position: [f32; 3],
    pub _pad0: f32,
    /// World-space shading normal.
    pub normal: [f32; 3],
    pub _pad1: f32,
}

/// A triangle as three vertex indices plus the index of its material.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct RtTriangle {
    /// Index of the first vertex.
    pub v0: u32,
    /// Index of the second vertex.
    pub v1: u32,
    /// Index of the third vertex.
    pub v2: u32,
    /// Index into the material table.
    pub material_id: u32,
}

/// PBR material parameters, padded to a 48-byte std430 layout.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct RtMaterial {
    /// Base (albedo) color, RGBA.
    pub base_color: [f32; 4],
    /// Emissive color, RGB (alpha unused).
    pub emissive: [f32; 4],
    /// Metallic factor in `[0, 1]`.
    pub metallic: f32,
    /// Roughness factor in `[0, 1]`.
    pub roughness: f32,
    pub _pad: [f32; 2],
}

/// A light in world space, padded to a 64-byte std430 layout.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct RtLight {
    /// World-space position (point/spot).
    pub position: [f32; 3],
    /// One of [`RT_LIGHT_POINT`], [`RT_LIGHT_DIRECTIONAL`], [`RT_LIGHT_SPOT`].
    pub light_type: u32,
    /// World-space direction (directional/spot).
    pub direction: [f32; 3],
    /// Intensity multiplier.
    pub intensity: f32,
    /// Light color, RGB.
    pub color: [f32; 3],
    /// Maximum distance the light affects (point/spot).
    pub attenuation_radius: f32,
    /// Cosine of the inner cone angle (spot).
    pub inner_cone_cos: f32,
    /// Cosine of the outer cone angle (spot).
    pub outer_cone_cos: f32,
    pub _pad: [f32; 2],
}

/// The flattened scene ready to be uploaded to the GPU.
#[derive(Default)]
pub struct RtScene {
    /// All vertices of every object, with world transforms baked in.
    pub vertices: Vec<RtVertex>,
    /// All triangles, indexing into [`Self::vertices`].
    pub triangles: Vec<RtTriangle>,
    /// One material per object.
    pub materials: Vec<RtMaterial>,
    /// Lights collected from the scene tree.
    pub lights: Vec<RtLight>,
    /// Global ambient intensity (drives the sky term in the kernel).
    pub ambient: f32,
    /// Content hash used to detect changes that require a GPU rebuild.
    pub hash: u64,
}

impl RtScene {
    /// Returns `true` if the scene contains no triangles.
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }
}

/// FNV-1a hasher accumulator.
struct Fnv(u64);

impl Fnv {
    #[inline]
    fn new() -> Self {
        Fnv(0xcbf29ce484222325)
    }

    #[inline]
    fn write_u32(&mut self, v: u32) {
        for b in v.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    #[inline]
    fn write_f32(&mut self, v: f32) {
        // Normalize -0.0 to 0.0 so equal values hash equally.
        let bits = if v == 0.0 { 0 } else { v.to_bits() };
        self.write_u32(bits);
    }

    #[inline]
    fn write_vec3(&mut self, v: Vec3) {
        self.write_f32(v.x);
        self.write_f32(v.y);
        self.write_f32(v.z);
    }
}

/// Walks the scene graph and builds an [`RtScene`].
///
/// `lights` must already have been populated for the current frame (e.g. by the
/// scene's `prepare` pass, which also propagates world transforms). The ambient
/// term is taken from the light collection.
///
/// World transforms are baked into vertex positions; normals are transformed by
/// the inverse-transpose (rotation combined with the reciprocal of the non-uniform
/// scale) so they stay correct under anisotropic scaling.
pub fn gather(scene: &SceneNode3d, lights: &LightCollection) -> RtScene {
    let mut out = RtScene {
        ambient: lights.ambient,
        ..Default::default()
    };
    let mut hasher = Fnv::new();

    scene.apply_to_scene_nodes_recursive(&mut |node| {
        // `world_pose`/`world_scale` borrow the node data internally, so fetch
        // them before taking the immutable `data()` borrow below.
        let pose = node.world_pose();
        let scale = node.world_scale();
        let data = node.data();

        let Some(obj) = data.object() else {
            return;
        };
        if !obj.data().surface_rendering_active() {
            return;
        }

        let mesh = obj.mesh().borrow();
        let coords_lock = mesh.coords().read().unwrap();
        let faces_lock = mesh.faces().read().unwrap();
        let normals_lock = mesh.normals().read().unwrap();

        let (Some(coords), Some(faces)) = (coords_lock.data().as_ref(), faces_lock.data().as_ref())
        else {
            return;
        };
        if coords.is_empty() || faces.is_empty() {
            return;
        }
        let normals = normals_lock.data().as_ref();

        let material_id = out.materials.len() as u32;
        let odata = obj.data();
        let color = odata.color();
        let emissive = odata.emissive();
        out.materials.push(RtMaterial {
            base_color: [color.r, color.g, color.b, color.a],
            emissive: [emissive.r, emissive.g, emissive.b, 1.0],
            metallic: odata.metallic(),
            roughness: odata.roughness(),
            _pad: [0.0; 2],
        });

        // Inverse scale for the normal transform (guard against zero components).
        let inv_scale = Vec3::new(
            if scale.x != 0.0 { 1.0 / scale.x } else { 0.0 },
            if scale.y != 0.0 { 1.0 / scale.y } else { 0.0 },
            if scale.z != 0.0 { 1.0 / scale.z } else { 0.0 },
        );

        let base_vertex = out.vertices.len() as u32;
        for (i, &local_pos) in coords.iter().enumerate() {
            let world_pos = pose.rotation * (local_pos * scale) + pose.translation;
            let local_n = normals.and_then(|n| n.get(i)).copied().unwrap_or(Vec3::Y);
            let world_n = (pose.rotation * (local_n * inv_scale)).normalize_or(Vec3::Y);
            out.vertices.push(RtVertex {
                position: [world_pos.x, world_pos.y, world_pos.z],
                _pad0: 0.0,
                normal: [world_n.x, world_n.y, world_n.z],
                _pad1: 0.0,
            });
        }

        for f in faces {
            out.triangles.push(RtTriangle {
                v0: base_vertex + f[0],
                v1: base_vertex + f[1],
                v2: base_vertex + f[2],
                material_id,
            });
        }

        hash_object(
            &mut hasher,
            pose,
            scale,
            color,
            emissive,
            odata.metallic(),
            odata.roughness(),
            coords.len(),
            faces.len(),
        );
    });

    // Lights also influence the rendered image; fold them into the hash so a
    // moved/added light resets accumulation via a rebuild.
    for cl in &lights.lights {
        out.lights.push(collected_to_rt(cl));
        hash_light(&mut hasher, cl);
    }
    hasher.write_f32(lights.ambient);

    out.hash = hasher.0;
    out
}

/// Computes the same content hash as [`gather`] without building the (expensive)
/// vertex/triangle/material arrays. Used every frame to detect whether the GPU
/// scene must be rebuilt; only on a change does the full [`gather`] run.
pub fn scene_hash(scene: &SceneNode3d, lights: &LightCollection) -> u64 {
    let mut hasher = Fnv::new();

    scene.apply_to_scene_nodes_recursive(&mut |node| {
        let pose = node.world_pose();
        let scale = node.world_scale();
        let data = node.data();

        let Some(obj) = data.object() else {
            return;
        };
        if !obj.data().surface_rendering_active() {
            return;
        }

        let mesh = obj.mesh().borrow();
        let ncoords = mesh.coords().read().unwrap().len();
        let nfaces = mesh.faces().read().unwrap().len();
        if ncoords == 0 || nfaces == 0 {
            return;
        }

        let odata = obj.data();
        hash_object(
            &mut hasher,
            pose,
            scale,
            odata.color(),
            odata.emissive(),
            odata.metallic(),
            odata.roughness(),
            ncoords,
            nfaces,
        );
    });

    for cl in &lights.lights {
        hash_light(&mut hasher, cl);
    }
    hasher.write_f32(lights.ambient);

    hasher.0
}

/// Hashes the cheap-but-discriminating bits of an object: world transform,
/// material and element counts. Per-vertex deformation is intentionally not
/// hashed (too costly); callers mutating vertices in place use
/// [`RayTracer::mark_dirty`](crate::renderer::RayTracer::mark_dirty).
#[allow(clippy::too_many_arguments)]
fn hash_object(
    h: &mut Fnv,
    pose: glamx::Pose3,
    scale: Vec3,
    color: crate::color::Color,
    emissive: crate::color::Color,
    metallic: f32,
    roughness: f32,
    ncoords: usize,
    nfaces: usize,
) {
    h.write_vec3(pose.translation);
    h.write_f32(pose.rotation.x);
    h.write_f32(pose.rotation.y);
    h.write_f32(pose.rotation.z);
    h.write_f32(pose.rotation.w);
    h.write_vec3(scale);
    h.write_f32(metallic);
    h.write_f32(roughness);
    for c in [color.r, color.g, color.b, color.a, emissive.r, emissive.g, emissive.b] {
        h.write_f32(c);
    }
    h.write_u32(ncoords as u32);
    h.write_u32(nfaces as u32);
}

fn hash_light(h: &mut Fnv, cl: &CollectedLight) {
    h.write_vec3(cl.world_position);
    h.write_vec3(cl.world_direction);
    h.write_vec3(cl.color);
    h.write_f32(cl.intensity);
}

fn collected_to_rt(cl: &CollectedLight) -> RtLight {
    let (light_type, attenuation_radius, inner_cone_cos, outer_cone_cos) = match cl.light_type {
        LightType::Point { attenuation_radius } => (RT_LIGHT_POINT, attenuation_radius, 1.0, 0.0),
        LightType::Directional(_) => (RT_LIGHT_DIRECTIONAL, 0.0, 1.0, 0.0),
        LightType::Spot {
            inner_cone_angle,
            outer_cone_angle,
            attenuation_radius,
        } => (
            RT_LIGHT_SPOT,
            attenuation_radius,
            inner_cone_angle.cos(),
            outer_cone_angle.cos(),
        ),
    };

    RtLight {
        position: [cl.world_position.x, cl.world_position.y, cl.world_position.z],
        light_type,
        direction: [cl.world_direction.x, cl.world_direction.y, cl.world_direction.z],
        intensity: cl.intensity,
        color: [cl.color.x, cl.color.y, cl.color.z],
        attenuation_radius,
        inner_cone_cos,
        outer_cone_cos,
        _pad: [0.0; 2],
    }
}
