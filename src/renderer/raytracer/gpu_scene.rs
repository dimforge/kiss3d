//! GPU-resident scene data for the path tracer.
//!
//! Holds the storage buffers shared by both backends (vertices, triangles,
//! materials, lights). The compute backend additionally stores a BVH node
//! buffer; the hardware backend additionally builds BLAS/TLAS acceleration
//! structures (behind the `hw_raytracer` feature).

use bytemuck::{Pod, Zeroable};
use glamx::{Mat4, Vec3};

use crate::context::Context;

use super::bvh::{self, BvhNode};
use super::scene_data::{
    RtEmitter, RtInstance, RtLight, RtMaterial, RtMeshDesc, RtMeshRange, RtScene, RtTriangle,
    RtVertex,
};
use super::tex_array::TexArray;
use super::RayBackend;

/// Storage buffers (and, for the hardware backend, acceleration structures)
/// describing the scene on the GPU.
pub struct GpuScene {
    /// World-space vertices (`array<RtVertex>`).
    pub vertices: wgpu::Buffer,
    /// Triangles. Compute: reordered to match BVH leaves. Hardware: original
    /// order, so `primitive_index` from a ray query indexes directly into it.
    pub triangles: wgpu::Buffer,
    /// Per-object materials (`array<RtMaterial>`).
    pub materials: wgpu::Buffer,
    /// Lights (`array<RtLight>`).
    pub lights: wgpu::Buffer,
    /// Emissive triangles for next-event estimation (`array<RtEmitter>`).
    pub emitters: wgpu::Buffer,
    /// Compute backend: the merged two-level BVH bound at binding 4 — top-level
    /// (TLAS) nodes followed by every mesh's bottom-level (BLAS) nodes. (Hardware
    /// backend: an unused dummy.)
    pub bvh: wgpu::Buffer,
    /// Hardware backend only: per-mesh descriptors (`array<RtMeshDesc>`) mapping a
    /// ray query's `primitive_index` to the global triangle table. The compute
    /// backend inlines these offsets into each [`RtInstance`] instead.
    #[cfg(feature = "hw_raytracer")]
    pub meshes: wgpu::Buffer,
    /// Compute backend: instances (`array<RtInstance>`), reordered so TLAS leaves
    /// are contiguous.
    pub instances: wgpu::Buffer,
    /// Packed PBR maps as a `texture_2d_array` (with a trailing white fallback).
    pub tex_array: TexArray,
    /// Number of triangles actually present (the buffer may be padded).
    pub num_triangles: u32,
    /// Number of lights actually present (the buffer may be padded).
    pub num_lights: u32,
    /// Number of emissive triangles actually present (the buffer may be padded).
    pub num_emitters: u32,
    /// Whether any material is translucent (`base_color.a < 1`). When false the
    /// kernel uses a cheap binary occlusion test for shadow rays; when true it
    /// accumulates colored transmittance through translucent occluders.
    pub has_translucent: bool,
    /// Content hash of the [`RtScene`] this was built from.
    pub hash: u64,

    /// Bottom-level acceleration structures, one per mesh (kept alive while
    /// referenced by the TLAS). Hardware backend only.
    #[cfg(feature = "hw_raytracer")]
    _blas: Vec<wgpu::Blas>,
    /// Top-level acceleration structure bound to the path-tracing pipeline.
    /// Hardware backend only.
    #[cfg(feature = "hw_raytracer")]
    pub tlas: Option<wgpu::Tlas>,
}

/// Uploads a slice as a buffer with the given usage, padding empty slices to one
/// element so the buffer is always bindable.
fn buffer_from<T: Pod + Zeroable>(
    label: &str,
    data: &[T],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let ctxt = Context::get();
    let fallback = [T::zeroed()];
    let slice = if data.is_empty() { &fallback[..] } else { data };
    ctxt.create_buffer_init(Some(label), bytemuck::cast_slice(slice), usage)
}

/// Local-space AABB of one mesh's vertices.
fn mesh_local_aabb(vertices: &[RtVertex], r: RtMeshRange) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for v in &vertices[r.vert_start as usize..(r.vert_start + r.vert_count) as usize] {
        let p = Vec3::from_array(v.position);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    if r.vert_count == 0 {
        (Vec3::ZERO, Vec3::ZERO)
    } else {
        (lo, hi)
    }
}

/// World-space AABB of a local AABB transformed by `m` (transforms all 8 corners).
fn transform_aabb(m: Mat4, lo: Vec3, hi: Vec3) -> (Vec3, Vec3) {
    let mut wlo = Vec3::splat(f32::INFINITY);
    let mut whi = Vec3::splat(f32::NEG_INFINITY);
    for i in 0..8 {
        let corner = Vec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        );
        let w = m.transform_point3(corner);
        wlo = wlo.min(w);
        whi = whi.max(w);
    }
    (wlo, whi)
}

impl GpuScene {
    /// Builds the GPU scene for the given backend.
    pub fn build(scene: &RtScene, backend: RayBackend) -> GpuScene {
        match backend {
            RayBackend::Software => Self::build_compute(scene),
            #[cfg(feature = "hw_raytracer")]
            RayBackend::Hardware => Self::build_hardware(scene),
        }
    }

    /// Builds the compute backend with a **two-level** acceleration structure: one
    /// bottom-level BVH per unique mesh (in local space, shared by all its
    /// instances) plus a top-level BVH over the instances. This keeps instanced
    /// scenes compact — geometry is stored once per mesh, not once per instance.
    fn build_compute(scene: &RtScene) -> GpuScene {
        // Bottom level: one BVH per mesh, concatenated. Node child/leaf indices
        // stay mesh-local; the shader adds the mesh's node/tri offsets.
        let mut blas_nodes: Vec<BvhNode> = Vec::new();
        let mut ordered_tris: Vec<RtTriangle> = Vec::new();
        let mut mesh_descs: Vec<RtMeshDesc> = Vec::with_capacity(scene.mesh_ranges.len());
        for r in &scene.mesh_ranges {
            let tris =
                &scene.mesh_triangles[r.tri_start as usize..(r.tri_start + r.tri_count) as usize];
            let (nodes, ordered) = bvh::build(&scene.mesh_vertices, tris);
            mesh_descs.push(RtMeshDesc {
                node_offset: blas_nodes.len() as u32,
                tri_offset: ordered_tris.len() as u32,
                _pad: [0; 2],
            });
            blas_nodes.extend_from_slice(&nodes);
            ordered_tris.extend_from_slice(&ordered);
        }

        // Top level: BVH over instance world-space AABBs (mesh local AABB
        // transformed by object→world). Reorder instances so leaves are contiguous.
        let inst_bounds: Vec<(Vec3, Vec3)> = scene
            .instances
            .iter()
            .map(|inst| {
                let r = scene.mesh_ranges[inst.mesh_id as usize];
                let (lo, hi) = mesh_local_aabb(&scene.mesh_vertices, r);
                transform_aabb(Mat4::from_cols_array_2d(&inst.object_to_world), lo, hi)
            })
            .collect();
        let (tlas_nodes, inst_order) = bvh::build_tlas(&inst_bounds);

        // Merge the top- and bottom-level BVHs into a single `bvh` node buffer
        // (TLAS first, then all BLAS nodes) so the compute stage binds one node
        // buffer instead of two — WebGPU only guarantees 8 storage buffers/stage.
        // The BLAS region starts at `tlas_count`, so each mesh's node base is
        // shifted by it.
        let tlas_count = tlas_nodes.len() as u32;
        let mut bvh_nodes = tlas_nodes;
        bvh_nodes.extend_from_slice(&blas_nodes);

        // Reorder instances to match the TLAS leaves and inline each instance's
        // node/triangle bases (from its mesh descriptor), so the kernel needs no
        // separate mesh-descriptor buffer.
        let instances: Vec<RtInstance> = inst_order
            .iter()
            .map(|&i| {
                let mut inst = scene.instances[i as usize];
                let desc = mesh_descs[inst.mesh_id as usize];
                inst.node_offset = tlas_count + desc.node_offset;
                inst.tri_offset = desc.tri_offset;
                inst
            })
            .collect();

        GpuScene {
            vertices: buffer_from::<RtVertex>(
                "rt_mesh_vertices",
                &scene.mesh_vertices,
                wgpu::BufferUsages::STORAGE,
            ),
            triangles: buffer_from::<RtTriangle>(
                "rt_mesh_triangles",
                &ordered_tris,
                wgpu::BufferUsages::STORAGE,
            ),
            materials: buffer_from::<RtMaterial>(
                "rt_materials",
                &scene.materials,
                wgpu::BufferUsages::STORAGE,
            ),
            lights: buffer_from::<RtLight>("rt_lights", &scene.lights, wgpu::BufferUsages::STORAGE),
            emitters: buffer_from::<RtEmitter>(
                "rt_emitters",
                &scene.emitters,
                wgpu::BufferUsages::STORAGE,
            ),
            bvh: buffer_from::<BvhNode>("rt_bvh", &bvh_nodes, wgpu::BufferUsages::STORAGE),
            // The compute backend inlines the per-mesh node/tri bases into instances
            // and does not bind this buffer; it is kept only for the hardware backend.
            #[cfg(feature = "hw_raytracer")]
            meshes: buffer_from::<RtMeshDesc>(
                "rt_meshes",
                &mesh_descs,
                wgpu::BufferUsages::STORAGE,
            ),
            instances: buffer_from::<RtInstance>(
                "rt_instances",
                &instances,
                wgpu::BufferUsages::STORAGE,
            ),
            tex_array: TexArray::build(&scene.textures),
            num_triangles: scene.mesh_triangles.len() as u32,
            num_lights: scene.lights.len() as u32,
            num_emitters: scene.emitters.len() as u32,
            has_translucent: scene.materials.iter().any(|m| m.base_color[3] < 1.0),
            hash: scene.hash,
            #[cfg(feature = "hw_raytracer")]
            _blas: Vec::new(),
            #[cfg(feature = "hw_raytracer")]
            tlas: None,
        }
    }

    /// Builds the hardware backend with **TLAS instancing**: one bottom-level
    /// acceleration structure per unique mesh (local space), referenced by a
    /// top-level acceleration structure with one instance per copy, each carrying
    /// its object→world transform and a custom index = its index into the
    /// `instances` storage buffer (where the hit shader reads mesh + material).
    /// `primitive_index` from a ray query is the triangle within its mesh's BLAS;
    /// the shader maps it to the global table via the mesh's `tri_offset`.
    #[cfg(feature = "hw_raytracer")]
    fn build_hardware(scene: &RtScene) -> GpuScene {
        use wgpu::{
            AccelerationStructureFlags, AccelerationStructureGeometryFlags,
            AccelerationStructureUpdateMode, BlasGeometries, BlasGeometrySizeDescriptors,
            BlasTriangleGeometry, BlasTriangleGeometrySizeDescriptor, CreateBlasDescriptor,
            CreateTlasDescriptor, TlasInstance,
        };

        let ctxt = Context::get();

        // Storage buffers the hit shader reads: local mesh vertices/triangles, the
        // mesh table (tri offsets), and instances (mesh + material per copy).
        let mut verts = scene.mesh_vertices.clone();
        if verts.len() < 3 {
            verts.resize(3, RtVertex::default());
        }
        let mesh_descs: Vec<RtMeshDesc> = scene
            .mesh_ranges
            .iter()
            .map(|r| RtMeshDesc {
                node_offset: 0,
                tri_offset: r.tri_start,
                _pad: [0; 2],
            })
            .collect();
        // BLAS index buffer: every mesh triangle's 3 vertex indices (global into
        // `verts`), concatenated in gather order. Each mesh's BLAS reads its own
        // sub-range via `first_index`.
        let indices: Vec<u32> = scene
            .mesh_triangles
            .iter()
            .flat_map(|t| [t.v0, t.v1, t.v2])
            .collect();
        let indices = if indices.is_empty() {
            vec![0u32, 0, 0]
        } else {
            indices
        };
        let vertex_count = verts.len() as u32;

        let vertices = buffer_from::<RtVertex>(
            "rt_mesh_vertices",
            &verts,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        );
        let index_buffer = buffer_from::<u32>(
            "rt_blas_indices",
            &indices,
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
        );
        let triangles = buffer_from::<RtTriangle>(
            "rt_mesh_triangles",
            &scene.mesh_triangles,
            wgpu::BufferUsages::STORAGE,
        );
        let materials = buffer_from::<RtMaterial>(
            "rt_materials",
            &scene.materials,
            wgpu::BufferUsages::STORAGE,
        );
        let lights =
            buffer_from::<RtLight>("rt_lights", &scene.lights, wgpu::BufferUsages::STORAGE);
        let emitters =
            buffer_from::<RtEmitter>("rt_emitters", &scene.emitters, wgpu::BufferUsages::STORAGE);
        let meshes =
            buffer_from::<RtMeshDesc>("rt_meshes", &mesh_descs, wgpu::BufferUsages::STORAGE);
        let instances_buf = buffer_from::<RtInstance>(
            "rt_instances",
            &scene.instances,
            wgpu::BufferUsages::STORAGE,
        );
        let tex_array = TexArray::build(&scene.textures);
        // Unused by the hardware pipeline (it traverses the TLAS/BLAS objects), but
        // the storage-buffer fields are always present.
        let bvh = buffer_from::<BvhNode>(
            "rt_bvh_unused",
            &[BvhNode::default()],
            wgpu::BufferUsages::STORAGE,
        );

        // One BLAS per mesh. A mesh with `tri_count` triangles reads index range
        // [tri_start*3, (tri_start+tri_count)*3) from the shared index buffer.
        let mesh_sizes: Vec<BlasTriangleGeometrySizeDescriptor> = scene
            .mesh_ranges
            .iter()
            .map(|r| BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count,
                index_format: Some(wgpu::IndexFormat::Uint32),
                index_count: Some((r.tri_count.max(1)) * 3),
                flags: AccelerationStructureGeometryFlags::OPAQUE,
            })
            .collect();
        // Degenerate fallback for an empty scene so the buffers/AS stay valid.
        let mesh_sizes = if mesh_sizes.is_empty() {
            vec![BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count,
                index_format: Some(wgpu::IndexFormat::Uint32),
                index_count: Some(3),
                flags: AccelerationStructureGeometryFlags::OPAQUE,
            }]
        } else {
            mesh_sizes
        };
        let first_indices: Vec<u32> = if scene.mesh_ranges.is_empty() {
            vec![0]
        } else {
            scene.mesh_ranges.iter().map(|r| r.tri_start * 3).collect()
        };

        let blases: Vec<wgpu::Blas> = mesh_sizes
            .iter()
            .map(|sz| {
                ctxt.device.create_blas(
                    &CreateBlasDescriptor {
                        label: Some("rt_blas"),
                        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                        update_mode: AccelerationStructureUpdateMode::Build,
                    },
                    BlasGeometrySizeDescriptors::Triangles {
                        descriptors: vec![sz.clone()],
                    },
                )
            })
            .collect();

        // One TLAS instance per scene instance, transform = object→world (3x4
        // row-major), custom index = instance index into the `instances` buffer.
        let n_instances = scene.instances.len().max(1) as u32;
        let mut tlas = ctxt.device.create_tlas(&CreateTlasDescriptor {
            label: Some("rt_tlas"),
            max_instances: n_instances,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        });
        if scene.instances.is_empty() {
            tlas[0] = Some(TlasInstance::new(&blases[0], identity_3x4(), 0, 0xFF));
        } else {
            for (i, inst) in scene.instances.iter().enumerate() {
                let blas = &blases[inst.mesh_id as usize];
                tlas[i] = Some(TlasInstance::new(
                    blas,
                    transform_3x4(&inst.object_to_world),
                    i as u32,
                    0xFF,
                ));
            }
        }

        // Build every BLAS and the TLAS in one submission.
        let stride = std::mem::size_of::<RtVertex>() as wgpu::BufferAddress;
        let blas_entries: Vec<wgpu::BlasBuildEntry> = blases
            .iter()
            .zip(mesh_sizes.iter())
            .zip(first_indices.iter())
            .map(|((blas, sz), &first_index)| wgpu::BlasBuildEntry {
                blas,
                geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                    size: sz,
                    vertex_buffer: &vertices,
                    first_vertex: 0,
                    vertex_stride: stride,
                    index_buffer: Some(&index_buffer),
                    first_index: Some(first_index),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            })
            .collect();

        let mut encoder = ctxt.create_command_encoder(Some("rt_as_build"));
        encoder.build_acceleration_structures(blas_entries.iter(), std::iter::once(&tlas));
        ctxt.submit(std::iter::once(encoder.finish()));

        GpuScene {
            vertices,
            triangles,
            materials,
            lights,
            emitters,
            bvh,
            meshes,
            instances: instances_buf,
            tex_array,
            num_triangles: scene.mesh_triangles.len() as u32,
            num_lights: scene.lights.len() as u32,
            num_emitters: scene.emitters.len() as u32,
            has_translucent: scene.materials.iter().any(|m| m.base_color[3] < 1.0),
            hash: scene.hash,
            _blas: blases,
            tlas: Some(tlas),
        }
    }
}

/// Identity object→world transform as a 3x4 row-major matrix (wgpu TLAS format).
#[cfg(feature = "hw_raytracer")]
fn identity_3x4() -> [f32; 12] {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}

/// Converts a column-major 4x4 transform to the 3x4 row-major form a wgpu
/// `TlasInstance` expects (drops the implicit `[0,0,0,1]` bottom row).
#[cfg(feature = "hw_raytracer")]
fn transform_3x4(m: &[[f32; 4]; 4]) -> [f32; 12] {
    // m[col][row], column-major. Row-major 3x4: row r = (m[0][r], m[1][r], m[2][r], m[3][r]).
    [
        m[0][0], m[1][0], m[2][0], m[3][0], //
        m[0][1], m[1][1], m[2][1], m[3][1], //
        m[0][2], m[1][2], m[2][2], m[3][2], //
    ]
}
