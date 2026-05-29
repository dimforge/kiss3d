//! GPU-resident scene data for the path tracer.
//!
//! Holds the storage buffers shared by both backends (vertices, triangles,
//! materials, lights). The compute backend additionally stores a BVH node
//! buffer; the hardware backend additionally builds BLAS/TLAS acceleration
//! structures (behind the `raytracing` feature).

use bytemuck::{Pod, Zeroable};

use crate::context::Context;

use super::bvh::{self, BvhNode};
use super::scene_data::{RtLight, RtMaterial, RtScene, RtTriangle, RtVertex};
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
    /// Flattened BVH nodes (`array<BvhNode>`); meaningful for the compute backend.
    pub bvh: wgpu::Buffer,
    /// Number of triangles actually present (the buffer may be padded).
    pub num_triangles: u32,
    /// Number of lights actually present (the buffer may be padded).
    pub num_lights: u32,
    /// Content hash of the [`RtScene`] this was built from.
    pub hash: u64,

    /// Bottom-level acceleration structure (kept alive while referenced by the
    /// TLAS). Hardware backend only.
    #[cfg(feature = "raytracing")]
    _blas: Option<wgpu::Blas>,
    /// Top-level acceleration structure bound to the path-tracing pipeline.
    /// Hardware backend only.
    #[cfg(feature = "raytracing")]
    pub tlas: Option<wgpu::Tlas>,
}

/// Uploads a slice as a buffer with the given usage, padding empty slices to one
/// element so the buffer is always bindable.
fn buffer_from<T: Pod + Zeroable>(label: &str, data: &[T], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    let ctxt = Context::get();
    let fallback = [T::zeroed()];
    let slice = if data.is_empty() { &fallback[..] } else { data };
    ctxt.create_buffer_init(Some(label), bytemuck::cast_slice(slice), usage)
}

impl GpuScene {
    /// Builds the GPU scene for the given backend.
    pub fn build(scene: &RtScene, backend: RayBackend) -> GpuScene {
        match backend {
            RayBackend::Compute => Self::build_compute(scene),
            #[cfg(feature = "raytracing")]
            RayBackend::HardwareRayQuery => Self::build_hardware(scene),
        }
    }

    fn build_compute(scene: &RtScene) -> GpuScene {
        let (nodes, ordered_triangles) = bvh::build(&scene.vertices, &scene.triangles);
        GpuScene {
            vertices: buffer_from::<RtVertex>("rt_vertices", &scene.vertices, wgpu::BufferUsages::STORAGE),
            triangles: buffer_from::<RtTriangle>("rt_triangles", &ordered_triangles, wgpu::BufferUsages::STORAGE),
            materials: buffer_from::<RtMaterial>("rt_materials", &scene.materials, wgpu::BufferUsages::STORAGE),
            lights: buffer_from::<RtLight>("rt_lights", &scene.lights, wgpu::BufferUsages::STORAGE),
            bvh: buffer_from::<BvhNode>("rt_bvh", &nodes, wgpu::BufferUsages::STORAGE),
            num_triangles: scene.triangles.len() as u32,
            num_lights: scene.lights.len() as u32,
            hash: scene.hash,
            #[cfg(feature = "raytracing")]
            _blas: None,
            #[cfg(feature = "raytracing")]
            tlas: None,
        }
    }

    /// Builds the hardware backend: storage buffers plus a single BLAS containing
    /// the whole (already world-space) scene, referenced by an identity TLAS
    /// instance. Triangles keep their original order so `primitive_index` indexes
    /// the triangle table directly. The acceleration structures are built and
    /// submitted immediately so they are ready before the first trace.
    #[cfg(feature = "raytracing")]
    fn build_hardware(scene: &RtScene) -> GpuScene {
        use wgpu::{
            AccelerationStructureFlags, AccelerationStructureGeometryFlags,
            AccelerationStructureUpdateMode, BlasGeometries, BlasGeometrySizeDescriptors,
            BlasTriangleGeometry, BlasTriangleGeometrySizeDescriptor, CreateBlasDescriptor,
            CreateTlasDescriptor, TlasInstance,
        };

        let ctxt = Context::get();

        // BLAS needs at least one (possibly degenerate) triangle. Pad an empty
        // scene with three zero vertices and a degenerate triangle.
        let mut verts = scene.vertices.clone();
        let mut indices: Vec<u32> = scene
            .triangles
            .iter()
            .flat_map(|t| [t.v0, t.v1, t.v2])
            .collect();
        if verts.len() < 3 {
            verts.resize(3, RtVertex::default());
        }
        if indices.is_empty() {
            indices = vec![0, 0, 0];
        }

        let vertex_count = verts.len() as u32;
        let index_count = indices.len() as u32;
        let triangle_count = index_count / 3;

        let vertices = buffer_from::<RtVertex>(
            "rt_vertices",
            &verts,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        );
        let index_buffer = buffer_from::<u32>(
            "rt_blas_indices",
            &indices,
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
        );
        let triangles = buffer_from::<RtTriangle>(
            "rt_triangles",
            &scene.triangles,
            wgpu::BufferUsages::STORAGE,
        );
        let materials =
            buffer_from::<RtMaterial>("rt_materials", &scene.materials, wgpu::BufferUsages::STORAGE);
        let lights = buffer_from::<RtLight>("rt_lights", &scene.lights, wgpu::BufferUsages::STORAGE);
        // Unused by the hardware pipeline, but the field is always present.
        let bvh = buffer_from::<BvhNode>("rt_bvh_unused", &[BvhNode::default()], wgpu::BufferUsages::STORAGE);

        let size_desc = BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(index_count),
            flags: AccelerationStructureGeometryFlags::OPAQUE,
        };

        let blas = ctxt.device.create_blas(
            &CreateBlasDescriptor {
                label: Some("rt_blas"),
                flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: AccelerationStructureUpdateMode::Build,
            },
            BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size_desc.clone()],
            },
        );

        let mut tlas = ctxt.device.create_tlas(&CreateTlasDescriptor {
            label: Some("rt_tlas"),
            max_instances: 1,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        });
        // Identity 3x4 row-major transform (geometry is already in world space).
        let identity = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
        ];
        tlas[0] = Some(TlasInstance::new(&blas, identity, 0, 0xFF));

        let mut encoder = ctxt.create_command_encoder(Some("rt_as_build"));
        encoder.build_acceleration_structures(
            std::iter::once(&wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                    size: &size_desc,
                    vertex_buffer: &vertices,
                    first_vertex: 0,
                    vertex_stride: std::mem::size_of::<RtVertex>() as wgpu::BufferAddress,
                    index_buffer: Some(&index_buffer),
                    first_index: Some(0),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            }),
            std::iter::once(&tlas),
        );
        ctxt.submit(std::iter::once(encoder.finish()));

        let _ = triangle_count;

        GpuScene {
            vertices,
            triangles,
            materials,
            lights,
            bvh,
            num_triangles: scene.triangles.len() as u32,
            num_lights: scene.lights.len() as u32,
            hash: scene.hash,
            _blas: Some(blas),
            tlas: Some(tlas),
        }
    }
}
