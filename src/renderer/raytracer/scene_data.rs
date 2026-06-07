//! CPU-side extraction of the scene graph into flat, GPU-ready ray-tracing data.
//!
//! A single walk of the scene graph bakes every object's world transform into its
//! geometry and produces flat arrays of vertices, triangles, materials and lights,
//! together with a content hash used to detect when the GPU scene must be rebuilt.

use bytemuck::{Pod, Zeroable};
use glamx::{Mat3, Mat4, Vec3};

use crate::light::{CollectedLight, LightCollection, LightType};
use crate::scene::{InstancesBuffer3d, SceneNode3d};

/// Light type tag matching the WGSL `RtLight.light_type` convention.
pub const RT_LIGHT_POINT: u32 = 0;
/// Light type tag matching the WGSL `RtLight.light_type` convention.
pub const RT_LIGHT_DIRECTIONAL: u32 = 1;
/// Light type tag matching the WGSL `RtLight.light_type` convention.
pub const RT_LIGHT_SPOT: u32 = 2;

/// A single vertex, padded to a 32-byte std430 layout.
///
/// The UV is packed into the `w` lanes of the position/normal vec4s so the WGSL
/// `RtVertex` stays two `vec4<f32>`s (`position.w = u`, `normal.w = v`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct RtVertex {
    /// World-space position.
    pub position: [f32; 3],
    /// Texture coordinate U.
    pub u: f32,
    /// World-space shading normal.
    pub normal: [f32; 3],
    /// Texture coordinate V.
    pub v: f32,
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

/// One emissive triangle baked into world space for next-event estimation toward
/// area lights. Positions and emission are stored directly (not as indices) so
/// emitter sampling is independent of each backend's geometry layout. 64-byte
/// std430 layout (4 x vec4).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct RtEmitter {
    /// World-space position of the first vertex.
    pub p0: [f32; 3],
    pub _pad0: f32,
    /// World-space position of the second vertex.
    pub p1: [f32; 3],
    pub _pad1: f32,
    /// World-space position of the third vertex.
    pub p2: [f32; 3],
    pub _pad2: f32,
    /// Emission radiance (RGB).
    pub emission: [f32; 3],
    pub _pad3: f32,
}

/// Unified Disney-style material parameters, std430 96-byte layout (6 × vec4).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RtMaterial {
    /// Base (albedo) color, RGBA.
    pub base_color: [f32; 4],
    /// Emissive color, RGB (alpha unused).
    pub emissive: [f32; 4],
    /// Metallic factor in `[0, 1]`.
    pub metallic: f32,
    /// Roughness factor in `[0, 1]`.
    pub roughness: f32,
    /// Index of refraction (glass/dielectric).
    pub ior: f32,
    /// Transmission / specular-transmittance factor in `[0, 1]`.
    pub transmission: f32,
    /// Specular tint, RGB (multiplies the specular/conductor lobe).
    pub specular_tint: [f32; 3],
    /// BSDF model tag (see [`Bsdf::tag`](crate::scene::Bsdf)).
    pub bsdf_type: u32,
    /// Subsurface / translucency factor in `[0, 1]`.
    pub subsurface: f32,
    /// Subsurface scattering radius (world units).
    pub subsurface_radius: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    /// Texture-array layer index for the albedo map (-1 = none).
    pub albedo_tex: i32,
    /// Texture-array layer index for the normal map (-1 = none).
    pub normal_tex: i32,
    /// Texture-array layer index for the metallic-roughness map (-1 = none).
    pub mr_tex: i32,
    /// Texture-array layer index for the emissive map (-1 = none).
    pub emissive_tex: i32,
}

impl Default for RtMaterial {
    fn default() -> Self {
        RtMaterial {
            base_color: [1.0, 1.0, 1.0, 1.0],
            emissive: [0.0, 0.0, 0.0, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            ior: 1.5,
            transmission: 0.0,
            specular_tint: [1.0, 1.0, 1.0],
            bsdf_type: 0,
            subsurface: 0.0,
            subsurface_radius: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
            albedo_tex: -1,
            normal_tex: -1,
            mr_tex: -1,
            emissive_tex: -1,
        }
    }
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
    /// Sphere radius for soft shadows (0 = delta point/spot light).
    pub radius: f32,
    pub _pad: f32,
}

/// One instance for the two-level (compute) acceleration structure: a reference
/// to a shared bottom-level mesh plus the transforms placing it in the world.
/// 144-byte std430 layout (2 mat4 + a uvec4 tail).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RtInstance {
    /// World→object transform: brings a world-space ray into mesh-local space for
    /// bottom-level traversal (direction left un-normalized so `t` is preserved).
    pub world_to_object: [[f32; 4]; 4],
    /// Object→world transform: maps a local hit back to world space (positions,
    /// geometric normal, triangle area).
    pub object_to_world: [[f32; 4]; 4],
    /// Index into `mesh_ranges`. CPU-side use only (world-AABB computation and
    /// looking up the offsets below); the compute kernel reads the offsets directly.
    pub mesh_id: u32,
    /// Index into the material table.
    pub material_id: u32,
    /// Compute backend: base of this instance's mesh in the merged `bvh` node buffer
    /// (TLAS-node count + the mesh's BLAS-node base). Inlined here so the kernel needs
    /// no separate per-mesh descriptor buffer. Unused by the hardware backend.
    pub node_offset: u32,
    /// Compute backend: base of this instance's mesh triangles in the reordered
    /// triangle buffer. Unused by the hardware backend.
    pub tri_offset: u32,
}

/// GPU mesh descriptor for the two-level path: where this mesh's bottom-level BVH
/// nodes and triangles live in the concatenated buffers. 16-byte std430 layout.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct RtMeshDesc {
    /// Base index of this mesh's BVH nodes in the concatenated `blas_nodes`.
    pub node_offset: u32,
    /// Base index of this mesh's triangles in the concatenated `mesh_triangles`.
    pub tri_offset: u32,
    pub _pad: [u32; 2],
}

/// CPU-side per-mesh ranges produced by [`gather`] and consumed by
/// `gpu_scene` to build each mesh's bottom-level BVH.
#[derive(Copy, Clone, Debug, Default)]
pub struct RtMeshRange {
    /// First vertex of this mesh in `mesh_vertices`.
    pub vert_start: u32,
    /// Vertex count (for the local AABB).
    pub vert_count: u32,
    /// First triangle of this mesh in `mesh_triangles`.
    pub tri_start: u32,
    /// Triangle count.
    pub tri_count: u32,
}

/// The scene ready to be uploaded to the GPU, in a two-level (instanced) layout:
/// each unique mesh is stored once in *local* space (`mesh_vertices`/
/// `mesh_triangles`, delimited by `mesh_ranges`) and placed by `instances` with
/// per-instance transforms + materials. The compute backend builds a CPU
/// two-level BVH from this; the hardware backend builds one BLAS per mesh plus a
/// TLAS of instances. `emitters` are baked into world space for light sampling.
#[derive(Default)]
pub struct RtScene {
    /// Local-space vertices of every unique mesh, concatenated.
    pub mesh_vertices: Vec<RtVertex>,
    /// Triangles of every unique mesh (vertex indices are global into
    /// [`Self::mesh_vertices`]); `material_id` is unused (material comes from the
    /// instance). Concatenated per mesh, in gather order.
    pub mesh_triangles: Vec<RtTriangle>,
    /// Per-mesh vertex/triangle ranges into the two `mesh_*` arrays.
    pub mesh_ranges: Vec<RtMeshRange>,
    /// Instances referencing meshes with per-instance transforms + material.
    pub instances: Vec<RtInstance>,
    /// One material per object.
    pub materials: Vec<RtMaterial>,
    /// Lights collected from the scene tree.
    pub lights: Vec<RtLight>,
    /// Emissive triangles baked into world space, for next-event estimation toward
    /// area lights (built per emissive instance in [`gather`]).
    pub emitters: Vec<RtEmitter>,
    /// Source GPU textures, one per material-array layer, that `gpu_scene`
    /// blits into the path tracer's `texture_2d_array`. Layer index matches the
    /// `*_tex` fields of [`RtMaterial`].
    pub textures: Vec<std::sync::Arc<crate::resource::Texture>>,
    /// Global ambient intensity (drives the sky term in the kernel).
    pub ambient: f32,
    /// Content hash used to detect changes that require a GPU rebuild.
    pub hash: u64,
}

impl RtScene {
    /// Returns `true` if the scene contains no triangles.
    pub fn is_empty(&self) -> bool {
        self.mesh_triangles.is_empty()
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
/// Produces the two-level (instanced) representation used by both backends: each
/// unique mesh once in local space, plus per-instance transforms + materials. See
/// [`RtScene`].
pub fn gather(scene: &SceneNode3d, lights: &LightCollection) -> RtScene {
    let mut out = RtScene {
        ambient: lights.ambient,
        ..Default::default()
    };
    let mut hasher = Fnv::new();

    scene.apply_to_visible_scene_nodes_recursive(&mut |node| {
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
        let uvs_lock = mesh.uvs().read().unwrap();

        let (Some(coords), Some(faces)) = (coords_lock.data().as_ref(), faces_lock.data().as_ref())
        else {
            return;
        };
        if coords.is_empty() || faces.is_empty() {
            return;
        }
        let normals = normals_lock.data().as_ref();
        let uvs = uvs_lock.data().as_ref();

        let odata = obj.data();
        let color = odata.color();
        let emissive = odata.emissive();
        let tint = odata.specular_tint();

        // Whether this object is a skinned mesh whose palette is ready: if so its
        // geometry is CPU-skinned into world space below.
        let use_skin = odata.has_skin()
            && mesh.has_skin_vertices()
            && odata.skin().is_some_and(|s| !s.palette().is_empty());

        // CPU morph targets: the path tracer has no GPU morph either, so blend the
        // weighted position/normal deltas into the base mesh here (before any
        // skinning), mirroring the deform vertex shader. `local_pos`/`local_nrm`
        // then return the morphed value, or the base value when not morphed.
        let morph_weights = odata.morph_weights();
        let morphed: Option<(Vec<Vec3>, Option<Vec<Vec3>>)> =
            if mesh.has_morph() && morph_weights.iter().any(|&w| w != 0.0) {
                cpu_morph(&mesh, coords, normals, morph_weights)
            } else {
                None
            };
        let local_pos = |i: usize| -> Vec3 {
            match &morphed {
                Some((p, _)) => p[i],
                None => coords[i],
            }
        };
        let local_nrm = |i: usize| -> Vec3 {
            match &morphed {
                Some((_, Some(n))) => n[i],
                _ => normals.and_then(|n| n.get(i)).copied().unwrap_or(Vec3::Y),
            }
        };

        // A non-instanced object still carries one default instance (identity
        // deformation, zero offset, white color); an object with `set_instances`
        // carries one entry per copy. The path tracer bakes a separate world-space
        // copy of the geometry per instance (it has no hardware instancing). Skip
        // objects with zero instances — the rasterizer would draw nothing either.
        let instances = obj.instances().borrow();
        let num_instances = instances.len();
        if num_instances == 0 {
            return;
        }
        let inst_positions = instances.positions.data().as_ref();
        let inst_deformations = instances.deformations.data().as_ref();
        let inst_colors = instances.colors.data().as_ref();

        // Register any PBR maps into the texture-array layer list once for the
        // object (all its instances share the same maps), returning the assigned
        // layer index (or -1 when the object has no such map).
        let mut push_tex = |tex: Option<&std::sync::Arc<crate::resource::Texture>>| -> i32 {
            match tex {
                Some(t) => {
                    let idx = out.textures.len() as i32;
                    out.textures.push(t.clone());
                    idx
                }
                None => -1,
            }
        };
        // The base color map lives on the object's primary `texture` when set to
        // something other than the default white texture is hard to detect here,
        // so we treat the explicitly-set PBR maps as the texture sources and use
        // the primary texture as the albedo map.
        let albedo_tex = push_tex(Some(odata.texture()));
        let normal_tex = push_tex(odata.normal_map());
        let mr_tex = push_tex(odata.metallic_roughness_map());
        let emissive_tex = push_tex(odata.emissive_map());

        // Material shared by every instance, save for the instance-tinted base
        // color filled in per instance below.
        let base_material = RtMaterial {
            base_color: [color.r, color.g, color.b, color.a],
            emissive: [emissive.r, emissive.g, emissive.b, 1.0],
            metallic: odata.metallic(),
            roughness: odata.roughness(),
            ior: odata.ior(),
            transmission: odata.transmission(),
            specular_tint: [tint.r, tint.g, tint.b],
            bsdf_type: odata.bsdf().tag(),
            subsurface: odata.subsurface(),
            subsurface_radius: odata.subsurface_radius(),
            _pad0: 0.0,
            _pad1: 0.0,
            albedo_tex,
            normal_tex,
            mr_tex,
            emissive_tex,
        };

        let emissive_obj = emissive.r + emissive.g + emissive.b > 1.0e-4;
        let emission = [emissive.r, emissive.g, emissive.b];

        // Per-instance object→world transform, matching `default.wgsl`'s vertex
        // shader (T·R·deform·S) so the path tracer and rasterizer agree:
        //   world = inst_pos + WorldPose * (deform * (scale ⊙ local)).
        let instance_transform = |inst: usize| -> Mat4 {
            let inst_pos = inst_positions
                .and_then(|p| p.get(inst))
                .copied()
                .unwrap_or(Vec3::ZERO);
            let deform = match inst_deformations {
                Some(d) if d.len() >= inst * 3 + 3 => {
                    Mat3::from_cols(d[inst * 3], d[inst * 3 + 1], d[inst * 3 + 2])
                }
                _ => Mat3::IDENTITY,
            };
            Mat4::from_translation(pose.translation + inst_pos)
                * Mat4::from_quat(pose.rotation)
                * Mat4::from_mat3(deform)
                * Mat4::from_scale(scale)
        };
        // Instance color multiplies the object color (rasterizer `vert_color * color`).
        let instance_material = |inst: usize| -> RtMaterial {
            let inst_color = inst_colors
                .and_then(|c| c.get(inst))
                .copied()
                .unwrap_or([1.0; 4]);
            let mut mat = base_material;
            mat.base_color = [
                color.r * inst_color[0],
                color.g * inst_color[1],
                color.b * inst_color[2],
                color.a * inst_color[3],
            ];
            mat
        };

        {
            // Store the mesh once in LOCAL space; place copies via instances. Both
            // backends use this representation (compute builds a CPU two-level BVH;
            // the hardware backend builds one BLAS per mesh + a TLAS of instances).
            // A skinned mesh is deformed per-frame by its joint palette. The path
            // tracer has no GPU skinning, so we CPU-skin the vertices into WORLD
            // space here and place the mesh with an identity instance transform.
            // (The skinned rasterizer ignores the mesh node + instance transforms
            // too, so a single world-space copy matches it.)
            let skinned_verts: Option<Vec<(Vec3, Vec3)>> = if use_skin {
                let skin = odata.skin().unwrap();
                let palette = skin.palette();
                let joints_lock = mesh.skin_joints().unwrap().read().unwrap();
                let weights_lock = mesh.skin_weights().unwrap().read().unwrap();
                match (joints_lock.data().as_ref(), weights_lock.data().as_ref()) {
                    (Some(joints), Some(weights)) => Some(
                        (0..coords.len())
                            .map(|i| {
                                let local_n = local_nrm(i);
                                let m = skin_matrix(palette, joints[i], weights[i]);
                                let wp = m.transform_point3(local_pos(i));
                                let wn = Mat3::from_mat4(m) * local_n;
                                let wn = if wn.length_squared() > 1.0e-12 {
                                    wn.normalize()
                                } else {
                                    local_n
                                };
                                (wp, wn)
                            })
                            .collect(),
                    ),
                    _ => None,
                }
            } else {
                None
            };
            // A successfully-skinned mesh is already in world space.
            let skinned = skinned_verts.is_some();

            let mesh_id = out.mesh_ranges.len() as u32;
            let vert_start = out.mesh_vertices.len() as u32;
            for i in 0..coords.len() {
                let (pos, normal) = match &skinned_verts {
                    Some(sv) => sv[i],
                    None => (local_pos(i), local_nrm(i)),
                };
                let uv = uvs
                    .and_then(|u| u.get(i))
                    .copied()
                    .unwrap_or(glamx::Vec2::ZERO);
                out.mesh_vertices.push(RtVertex {
                    position: [pos.x, pos.y, pos.z],
                    u: uv.x,
                    normal: [normal.x, normal.y, normal.z],
                    v: uv.y,
                });
            }
            let tri_start = out.mesh_triangles.len() as u32;
            for f in faces {
                out.mesh_triangles.push(RtTriangle {
                    v0: vert_start + f[0],
                    v1: vert_start + f[1],
                    v2: vert_start + f[2],
                    material_id: 0, // unused; material comes from the instance
                });
            }
            out.mesh_ranges.push(RtMeshRange {
                vert_start,
                vert_count: coords.len() as u32,
                tri_start,
                tri_count: faces.len() as u32,
            });

            // Skinned meshes are baked in world space → a single identity instance.
            // Non-skinned meshes keep their per-instance world placements.
            let instance_count = if skinned { 1 } else { num_instances };
            for inst in 0..instance_count {
                let m = if skinned {
                    Mat4::IDENTITY
                } else {
                    instance_transform(inst)
                };
                let material_id = out.materials.len() as u32;
                out.materials.push(instance_material(inst));
                out.instances.push(RtInstance {
                    world_to_object: m.inverse().to_cols_array_2d(),
                    object_to_world: m.to_cols_array_2d(),
                    mesh_id,
                    material_id,
                    // Filled in by `GpuScene::build_compute` once the per-mesh node/
                    // triangle bases (and the TLAS-node count) are known.
                    node_offset: 0,
                    tri_offset: 0,
                });
                // Bake this instance's emissive triangles into world-space emitters.
                if emissive_obj {
                    for f in faces {
                        // Skinned vertices are already world-space; otherwise place
                        // the local vertex by the instance transform.
                        let world = |idx: u32| -> Vec3 {
                            match &skinned_verts {
                                Some(sv) => sv[idx as usize].0,
                                None => m.transform_point3(local_pos(idx as usize)),
                            }
                        };
                        let p0 = world(f[0]);
                        let p1 = world(f[1]);
                        let p2 = world(f[2]);
                        out.emitters.push(RtEmitter {
                            p0: p0.to_array(),
                            _pad0: 0.0,
                            p1: p1.to_array(),
                            _pad1: 0.0,
                            p2: p2.to_array(),
                            _pad2: 0.0,
                            emission,
                            _pad3: 0.0,
                        });
                    }
                }
            }
        }

        hash_object(&mut hasher, pose, scale, odata, coords.len(), faces.len());
        hash_instances(&mut hasher, &instances);
        hash_skin(&mut hasher, odata);
        hash_morph(&mut hasher, odata);
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

    scene.apply_to_visible_scene_nodes_recursive(&mut |node| {
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
        hash_object(&mut hasher, pose, scale, odata, ncoords, nfaces);
        hash_instances(&mut hasher, &obj.instances().borrow());
        hash_skin(&mut hasher, odata);
        hash_morph(&mut hasher, odata);
    });

    for cl in &lights.lights {
        hash_light(&mut hasher, cl);
    }
    hasher.write_f32(lights.ambient);

    hasher.0
}

/// Folds an object's instance data (count, per-instance offsets, deformations and
/// colors) into the change hash, so adding/moving/recoloring instances rebuilds the
/// GPU scene and resets accumulation. Must hash exactly the same bytes in [`gather`]
/// and [`scene_hash`].
/// The blended skinning matrix for one vertex: a weight-normalized combination of
/// its (up to four) joint palette matrices. Out-of-range joint indices fall back to
/// identity. Mirrors the skinned vertex shader so the path tracer and rasterizer
/// agree.
fn skin_matrix(palette: &[Mat4], joints: [u32; 4], weights: [f32; 4]) -> Mat4 {
    let wsum = weights[0] + weights[1] + weights[2] + weights[3];
    let inv = if wsum > 0.0 { 1.0 / wsum } else { 1.0 };
    let get = |j: u32| palette.get(j as usize).copied().unwrap_or(Mat4::IDENTITY);
    get(joints[0]) * (weights[0] * inv)
        + get(joints[1]) * (weights[1] * inv)
        + get(joints[2]) * (weights[2] * inv)
        + get(joints[3]) * (weights[3] * inv)
}

/// Folds a skinned object's joint-matrix palette into the change hash, so an
/// animated pose rebuilds the path-traced scene (and resets accumulation). Must
/// hash the same bytes in [`gather`] and [`scene_hash`]. No-op for non-skinned
/// objects.
fn hash_skin(h: &mut Fnv, odata: &crate::scene::ObjectData3d) {
    if let Some(skin) = odata.skin() {
        for m in skin.palette() {
            for c in m.to_cols_array() {
                h.write_f32(c);
            }
        }
    }
}

/// CPU-applies morph-target weights to the base mesh, returning blended local
/// positions and — when the mesh carries normal deltas — blended normals. Mirrors
/// the deform vertex shader so the path tracer matches the rasterizer. Returns
/// `None` if the morph deltas aren't CPU-resident.
fn cpu_morph(
    mesh: &crate::resource::GpuMesh3d,
    coords: &[Vec3],
    normals: Option<&Vec<Vec3>>,
    weights: &[f32],
) -> Option<(Vec<Vec3>, Option<Vec<Vec3>>)> {
    let nv = mesh.morph_vertex_count();
    let nt = mesh.morph_target_count().min(weights.len());
    let pos_lock = mesh.morph_positions()?.read().unwrap();
    let mpos = pos_lock.data().as_ref()?;
    let nrm_lock = mesh.morph_normals().map(|n| n.read().unwrap());
    let mnrm = nrm_lock.as_ref().and_then(|g| g.data().as_ref());

    let count = coords.len().min(nv);
    let mut positions = coords.to_vec();
    let mut out_normals = mnrm.is_some().then(|| match normals {
        Some(n) => n.clone(),
        None => vec![Vec3::Y; coords.len()],
    });

    for (t, &w) in weights.iter().take(nt).enumerate() {
        if w == 0.0 {
            continue;
        }
        let base = t * nv;
        for v in 0..count {
            let d = mpos[base + v];
            positions[v] += w * Vec3::new(d[0], d[1], d[2]);
        }
        if let (Some(mn), Some(on)) = (mnrm, out_normals.as_mut()) {
            for v in 0..count {
                let d = mn[base + v];
                on[v] += w * Vec3::new(d[0], d[1], d[2]);
            }
        }
    }
    Some((positions, out_normals))
}

/// Folds an object's current morph-target weights into the change hash, so a
/// morphing pose rebuilds the path-traced scene (and resets accumulation). Must
/// hash the same bytes in [`gather`] and [`scene_hash`]. No-op for non-morph objects.
fn hash_morph(h: &mut Fnv, odata: &crate::scene::ObjectData3d) {
    for &w in odata.morph_weights() {
        h.write_f32(w);
    }
}

fn hash_instances(h: &mut Fnv, instances: &InstancesBuffer3d) {
    h.write_u32(instances.len() as u32);
    if let Some(p) = instances.positions.data() {
        for v in p {
            h.write_vec3(*v);
        }
    }
    if let Some(d) = instances.deformations.data() {
        for v in d {
            h.write_vec3(*v);
        }
    }
    if let Some(c) = instances.colors.data() {
        for v in c {
            for x in v {
                h.write_f32(*x);
            }
        }
    }
}

/// Hashes the cheap-but-discriminating bits of an object: world transform,
/// material and element counts. Per-vertex deformation is intentionally not
/// hashed (too costly); callers mutating vertices in place use
/// [`RayTracer::mark_dirty`](crate::renderer::RayTracer::mark_dirty).
fn hash_object(
    h: &mut Fnv,
    pose: glamx::Pose3,
    scale: Vec3,
    odata: &crate::scene::ObjectData3d,
    ncoords: usize,
    nfaces: usize,
) {
    h.write_vec3(pose.translation);
    h.write_f32(pose.rotation.x);
    h.write_f32(pose.rotation.y);
    h.write_f32(pose.rotation.z);
    h.write_f32(pose.rotation.w);
    h.write_vec3(scale);
    h.write_f32(odata.metallic());
    h.write_f32(odata.roughness());
    let color = odata.color();
    let emissive = odata.emissive();
    let tint = odata.specular_tint();
    for c in [
        color.r, color.g, color.b, color.a, emissive.r, emissive.g, emissive.b,
    ] {
        h.write_f32(c);
    }
    // Path-tracer BSDF fields: a change must trigger a GPU rebuild (and reset).
    h.write_u32(odata.bsdf().tag());
    h.write_f32(odata.ior());
    h.write_f32(odata.transmission());
    for c in [tint.r, tint.g, tint.b] {
        h.write_f32(c);
    }
    h.write_f32(odata.subsurface());
    h.write_f32(odata.subsurface_radius());
    // Texture-map presence (pointers) so attaching/detaching a map rebuilds.
    let tex_id = |t: Option<&std::sync::Arc<crate::resource::Texture>>| -> u32 {
        t.map(|a| std::sync::Arc::as_ptr(a) as usize as u32)
            .unwrap_or(0)
    };
    h.write_u32(std::sync::Arc::as_ptr(odata.texture()) as usize as u32);
    h.write_u32(tex_id(odata.normal_map()));
    h.write_u32(tex_id(odata.metallic_roughness_map()));
    h.write_u32(tex_id(odata.emissive_map()));
    h.write_u32(ncoords as u32);
    h.write_u32(nfaces as u32);
}

fn hash_light(h: &mut Fnv, cl: &CollectedLight) {
    h.write_vec3(cl.world_position);
    h.write_vec3(cl.world_direction);
    h.write_vec3(cl.color);
    h.write_f32(cl.intensity);
    h.write_f32(cl.radius);
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
        position: [
            cl.world_position.x,
            cl.world_position.y,
            cl.world_position.z,
        ],
        light_type,
        direction: [
            cl.world_direction.x,
            cl.world_direction.y,
            cl.world_direction.z,
        ],
        intensity: cl.intensity,
        color: [cl.color.x, cl.color.y, cl.color.z],
        attenuation_radius,
        inner_cone_cos,
        outer_cone_cos,
        radius: cl.radius,
        _pad: 0.0,
    }
}
