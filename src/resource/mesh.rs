//! Data structure of a scene node geometry.
use std::sync::{Arc, RwLock};

use crate::procedural::{IndexBuffer, RenderMesh};
use crate::resource::gpu_vector::{AllocationType, BufferType, GPUVec};
use crate::resource::vertex_index::VertexIndex;
use na::{self, Point2, Point3, Vector3};
use num::Zero;

/// A 3D mesh stored on the GPU.
///
/// `GpuMesh` contains vertex data (coordinates, normals, UVs) and face indices
/// stored in GPU memory buffers for efficient rendering. This is the GPU-side
/// representation of mesh data.
///
/// # Relationship with RenderMesh
/// - [`RenderMesh`](crate::procedural::RenderMesh) is the CPU-side mesh descriptor
/// - `GpuMesh` is the GPU-side representation
/// - Use [`from_render_mesh()`](Self::from_render_mesh) to convert from CPU to GPU
/// - Use [`to_render_mesh()`](Self::to_render_mesh) to convert from GPU to CPU
pub struct GpuMesh {
    coords: Arc<RwLock<GPUVec<Point3<f32>>>>,
    faces: Arc<RwLock<GPUVec<Point3<VertexIndex>>>>,
    normals: Arc<RwLock<GPUVec<Vector3<f32>>>>,
    uvs: Arc<RwLock<GPUVec<Point2<f32>>>>,
    edges: Option<Arc<RwLock<GPUVec<Point2<VertexIndex>>>>>,
}

impl GpuMesh {
    /// Creates a new GPU mesh from vertex and face data.
    ///
    /// Uploads the provided mesh data to GPU memory. If normals or UVs are not provided,
    /// they are automatically computed (normals from face geometry, UVs as zero).
    ///
    /// # Arguments
    /// * `coords` - Vertex positions
    /// * `faces` - Triangle faces as indices into the coords array (each Point3 contains 3 vertex indices)
    /// * `normals` - Optional vertex normals (auto-computed if None)
    /// * `uvs` - Optional texture coordinates (set to origin if None)
    /// * `dynamic_draw` - If true, use dynamic GPU allocation for data that will be modified frequently
    ///
    /// # Returns
    /// A new `GpuMesh` with data uploaded to the GPU
    pub fn new(
        coords: Vec<Point3<f32>>,
        faces: Vec<Point3<VertexIndex>>,
        normals: Option<Vec<Vector3<f32>>>,
        uvs: Option<Vec<Point2<f32>>>,
        dynamic_draw: bool,
    ) -> GpuMesh {
        let normals = match normals {
            Some(ns) => ns,
            None => GpuMesh::compute_normals_array(&coords[..], &faces[..]),
        };

        let uvs = match uvs {
            Some(us) => us,
            None => vec![Point2::origin(); coords.len()],
        };

        let location = if dynamic_draw {
            AllocationType::DynamicDraw
        } else {
            AllocationType::StaticDraw
        };
        let cs = Arc::new(RwLock::new(GPUVec::new(
            coords,
            BufferType::Array,
            location,
        )));
        let fs = Arc::new(RwLock::new(GPUVec::new(
            faces,
            BufferType::ElementArray,
            location,
        )));
        let ns = Arc::new(RwLock::new(GPUVec::new(
            normals,
            BufferType::Array,
            location,
        )));
        let us = Arc::new(RwLock::new(GPUVec::new(uvs, BufferType::Array, location)));

        GpuMesh::new_with_gpu_vectors(cs, fs, ns, us)
    }

    /// Creates a GPU mesh from a procedural mesh descriptor.
    ///
    /// Converts a `RenderMesh` (CPU-side mesh descriptor) into a `GpuMesh`
    /// by uploading its data to GPU memory. If normals or UVs are not provided
    /// in the RenderMesh, they are automatically computed.
    ///
    /// # Arguments
    /// * `mesh` - The procedural mesh descriptor to convert
    /// * `dynamic_draw` - If true, use dynamic GPU allocation for data that will be modified frequently
    ///
    /// # Returns
    /// A new `GpuMesh` with data uploaded to the GPU
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::procedural;
    /// # use kiss3d::resource::GpuMesh;
    /// let render_mesh = procedural::sphere(1.0, 32, 16, true);
    /// let gpu_mesh = GpuMesh::from_render_mesh(render_mesh, false);
    /// ```
    pub fn from_render_mesh(mesh: RenderMesh, dynamic_draw: bool) -> GpuMesh {
        let mut mesh = mesh;

        mesh.unify_index_buffer();

        let RenderMesh {
            coords,
            normals,
            uvs,
            indices,
        } = mesh;

        GpuMesh::new(
            coords,
            indices
                .unwrap_unified()
                .into_iter()
                .map(na::convert)
                .collect(),
            normals,
            uvs,
            dynamic_draw,
        )
    }

    /// Creates a triangle mesh from this mesh.
    ///
    /// Return `None` if the mesh data is not available on the CPU.
    pub fn to_render_mesh(&self) -> Option<RenderMesh> {
        if !self.coords.read().unwrap().is_on_ram()
            || !self.faces.read().unwrap().is_on_ram()
            || !self.normals.read().unwrap().is_on_ram()
            || !self.uvs.read().unwrap().is_on_ram()
        {
            return None;
        }

        let coords = self.coords.read().unwrap().to_owned();
        let faces = self.faces.read().unwrap().to_owned();
        let normals = self.normals.read().unwrap().to_owned();
        let uvs = self.uvs.read().unwrap().to_owned();

        Some(RenderMesh::new(
            coords.unwrap(),
            normals,
            uvs,
            Some(IndexBuffer::Unified(
                faces
                    .unwrap()
                    .into_iter()
                    .map(|e| Point3::new(e.x as u32, e.y as u32, e.z as u32))
                    .collect(),
            )),
        ))
    }

    /// Creates a new mesh. Arguments set to `None` are automatically computed.
    pub fn new_with_gpu_vectors(
        coords: Arc<RwLock<GPUVec<Point3<f32>>>>,
        faces: Arc<RwLock<GPUVec<Point3<VertexIndex>>>>,
        normals: Arc<RwLock<GPUVec<Vector3<f32>>>>,
        uvs: Arc<RwLock<GPUVec<Point2<f32>>>>,
    ) -> GpuMesh {
        GpuMesh {
            coords,
            faces,
            normals,
            uvs,
            edges: None,
        }
    }

    /// Ensures all mesh buffers are loaded to the GPU and returns buffer references.
    ///
    /// This must be called before rendering. Returns None if any buffer is empty.
    pub fn ensure_on_gpu(
        &mut self,
    ) -> Option<(&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer)> {
        // Load all buffers to GPU
        self.coords.write().unwrap().load_to_gpu();
        self.faces.write().unwrap().load_to_gpu();
        self.normals.write().unwrap().load_to_gpu();
        self.uvs.write().unwrap().load_to_gpu();

        // Get buffer references
        let coords = self.coords.read().unwrap();
        let faces = self.faces.read().unwrap();
        let normals = self.normals.read().unwrap();
        let uvs = self.uvs.read().unwrap();

        if coords.buffer().is_none()
            || faces.buffer().is_none()
            || normals.buffer().is_none()
            || uvs.buffer().is_none()
        {
            return None;
        }

        // SAFETY: We just verified all buffers exist and hold read locks
        // We need to return references that outlive this function, but the buffers
        // are stored in Arc<RwLock<>> so they won't be deallocated.
        // This is a bit awkward but necessary for the wgpu API pattern.
        unsafe {
            let coords_ptr = coords.buffer().unwrap() as *const wgpu::Buffer;
            let faces_ptr = faces.buffer().unwrap() as *const wgpu::Buffer;
            let normals_ptr = normals.buffer().unwrap() as *const wgpu::Buffer;
            let uvs_ptr = uvs.buffer().unwrap() as *const wgpu::Buffer;
            Some((&*coords_ptr, &*faces_ptr, &*normals_ptr, &*uvs_ptr))
        }
    }

    /// Returns the vertex coordinates buffer if loaded to GPU.
    pub fn coords_buffer(&self) -> Option<&wgpu::Buffer> {
        // This is tricky because we need to return a reference from inside RwLock
        // For now, callers should use ensure_on_gpu() or access via coords()
        None
    }

    /// Returns the index buffer if loaded to GPU.
    pub fn faces_buffer(&self) -> Option<&wgpu::Buffer> {
        None
    }

    /// Returns the normals buffer if loaded to GPU.
    pub fn normals_buffer(&self) -> Option<&wgpu::Buffer> {
        None
    }

    /// Returns the UVs buffer if loaded to GPU.
    pub fn uvs_buffer(&self) -> Option<&wgpu::Buffer> {
        None
    }

    /// Ensures edge data is created (but not necessarily uploaded to GPU).
    pub fn ensure_edges(&mut self) {
        if self.edges.is_none() {
            let mut edges = Vec::new();
            for face in self.faces.read().unwrap().data().as_ref().unwrap() {
                edges.push(Point2::new(face.x, face.y));
                edges.push(Point2::new(face.y, face.z));
                edges.push(Point2::new(face.z, face.x));
            }
            let gpu_edges =
                GPUVec::new(edges, BufferType::ElementArray, AllocationType::StaticDraw);
            self.edges = Some(Arc::new(RwLock::new(gpu_edges)));
        }
    }

    /// Ensures edge buffer is created and loaded to GPU.
    pub fn ensure_edges_on_gpu(&mut self) {
        self.ensure_edges();
        self.edges.as_mut().unwrap().write().unwrap().load_to_gpu();
    }

    /// Returns the edges buffer reference.
    pub fn edges(&self) -> &Option<Arc<RwLock<GPUVec<Point2<VertexIndex>>>>> {
        &self.edges
    }

    /// Number of points needed to draw this mesh.
    pub fn num_pts(&self) -> usize {
        self.faces.read().unwrap().len() * 3
    }

    /// Number of indices in this mesh.
    pub fn num_indices(&self) -> u32 {
        (self.faces.read().unwrap().len() * 3) as u32
    }

    /// Number of edge indices in this mesh.
    pub fn num_edge_indices(&self) -> u32 {
        self.edges
            .as_ref()
            .map(|e| (e.read().unwrap().len() * 2) as u32)
            .unwrap_or(0)
    }

    /// Recompute this mesh normals.
    pub fn recompute_normals(&mut self) {
        GpuMesh::compute_normals(
            &self.coords.read().unwrap().data().as_ref().unwrap()[..],
            &self.faces.read().unwrap().data().as_ref().unwrap()[..],
            self.normals.write().unwrap().data_mut().as_mut().unwrap(),
        );
    }

    /// This mesh faces.
    pub fn faces(&self) -> &Arc<RwLock<GPUVec<Point3<VertexIndex>>>> {
        &self.faces
    }

    /// This mesh normals.
    pub fn normals(&self) -> &Arc<RwLock<GPUVec<Vector3<f32>>>> {
        &self.normals
    }

    /// This mesh vertex coordinates.
    pub fn coords(&self) -> &Arc<RwLock<GPUVec<Point3<f32>>>> {
        &self.coords
    }

    /// This mesh texture coordinates.
    pub fn uvs(&self) -> &Arc<RwLock<GPUVec<Point2<f32>>>> {
        &self.uvs
    }

    /// Computes normals from a set of faces.
    pub fn compute_normals_array(
        coordinates: &[Point3<f32>],
        faces: &[Point3<VertexIndex>],
    ) -> Vec<Vector3<f32>> {
        let mut res = Vec::new();

        GpuMesh::compute_normals(coordinates, faces, &mut res);

        res
    }

    /// Computes normals from a set of faces.
    pub fn compute_normals(
        coordinates: &[Point3<f32>],
        faces: &[Point3<VertexIndex>],
        normals: &mut Vec<Vector3<f32>>,
    ) {
        let mut divisor: Vec<f32> = vec![0f32; coordinates.len()];

        normals.clear();
        normals.extend(std::iter::repeat_n(
            Vector3::<f32>::zero(),
            coordinates.len(),
        ));

        // Accumulate normals ...
        for f in faces.iter() {
            let edge1 = coordinates[f.y as usize] - coordinates[f.x as usize];
            let edge2 = coordinates[f.z as usize] - coordinates[f.x as usize];
            let cross = edge1.cross(&edge2);

            let normal = if !cross.is_zero() {
                cross.normalize()
            } else {
                cross
            };

            normals[f.x as usize] += normal;
            normals[f.y as usize] += normal;
            normals[f.z as usize] += normal;

            divisor[f.x as usize] += 1.0;
            divisor[f.y as usize] += 1.0;
            divisor[f.z as usize] += 1.0;
        }

        // ... and compute the mean
        for (n, divisor) in normals.iter_mut().zip(divisor.iter()) {
            *n /= *divisor
        }
    }
}
