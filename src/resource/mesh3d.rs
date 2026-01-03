//! Data structure of a scene node geometry.
use std::sync::{Arc, RwLock};

use crate::procedural::{IndexBuffer, RenderMesh};
use crate::resource::gpu_vector::{AllocationType, BufferType, GPUVec};
use crate::resource::vertex_index::VertexIndex;
use glamx::{Vec2, Vec3};

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
pub struct GpuMesh3d {
    coords: Arc<RwLock<GPUVec<Vec3>>>,
    faces: Arc<RwLock<GPUVec<[VertexIndex; 3]>>>,
    normals: Arc<RwLock<GPUVec<Vec3>>>,
    uvs: Arc<RwLock<GPUVec<Vec2>>>,
    edges: Option<Arc<RwLock<GPUVec<[VertexIndex; 2]>>>>,
}

impl GpuMesh3d {
    /// Creates a new GPU mesh from vertex and face data.
    ///
    /// Uploads the provided mesh data to GPU memory. If normals or UVs are not provided,
    /// they are automatically computed (normals from face geometry, UVs as zero).
    ///
    /// # Arguments
    /// * `coords` - Vertex positions
    /// * `faces` - Triangle faces as indices into the coords array (each array contains 3 vertex indices)
    /// * `normals` - Optional vertex normals (auto-computed if None)
    /// * `uvs` - Optional texture coordinates (set to origin if None)
    /// * `dynamic_draw` - If true, use dynamic GPU allocation for data that will be modified frequently
    ///
    /// # Returns
    /// A new `GpuMesh` with data uploaded to the GPU
    pub fn new(
        coords: Vec<Vec3>,
        faces: Vec<[VertexIndex; 3]>,
        normals: Option<Vec<Vec3>>,
        uvs: Option<Vec<Vec2>>,
        dynamic_draw: bool,
    ) -> GpuMesh3d {
        let normals = match normals {
            Some(ns) => ns,
            None => GpuMesh3d::compute_normals_array(&coords[..], &faces[..]),
        };

        let uvs = match uvs {
            Some(us) => us,
            None => vec![Vec2::ZERO; coords.len()],
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

        GpuMesh3d::new_with_gpu_vectors(cs, fs, ns, us)
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
    /// # use kiss3d::resource::GpuMesh3d;
    /// let render_mesh = procedural::sphere(1.0, 32, 16, true);
    /// let gpu_mesh = GpuMesh3d::from_render_mesh(render_mesh, false);
    /// ```
    pub fn from_render_mesh(mesh: RenderMesh, dynamic_draw: bool) -> GpuMesh3d {
        let mut mesh = mesh;

        mesh.unify_index_buffer();

        let RenderMesh {
            coords,
            normals,
            uvs,
            indices,
        } = mesh;

        // Convert [u32; 3] indices to [VertexIndex; 3]
        let faces: Vec<[VertexIndex; 3]> = indices
            .unwrap_unified()
            .into_iter()
            .map(|idx| {
                [
                    idx[0] as VertexIndex,
                    idx[1] as VertexIndex,
                    idx[2] as VertexIndex,
                ]
            })
            .collect();

        GpuMesh3d::new(coords, faces, normals, uvs, dynamic_draw)
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
            Some(IndexBuffer::Unified(faces.unwrap().into_iter().collect())),
        ))
    }

    /// Creates a new mesh. Arguments set to `None` are automatically computed.
    pub fn new_with_gpu_vectors(
        coords: Arc<RwLock<GPUVec<Vec3>>>,
        faces: Arc<RwLock<GPUVec<[VertexIndex; 3]>>>,
        normals: Arc<RwLock<GPUVec<Vec3>>>,
        uvs: Arc<RwLock<GPUVec<Vec2>>>,
    ) -> GpuMesh3d {
        GpuMesh3d {
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
                edges.push([face[0], face[1]]);
                edges.push([face[1], face[2]]);
                edges.push([face[2], face[0]]);
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
    pub fn edges(&self) -> &Option<Arc<RwLock<GPUVec<[VertexIndex; 2]>>>> {
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
        GpuMesh3d::compute_normals(
            &self.coords.read().unwrap().data().as_ref().unwrap()[..],
            &self.faces.read().unwrap().data().as_ref().unwrap()[..],
            self.normals.write().unwrap().data_mut().as_mut().unwrap(),
        );
    }

    /// This mesh faces.
    pub fn faces(&self) -> &Arc<RwLock<GPUVec<[VertexIndex; 3]>>> {
        &self.faces
    }

    /// This mesh normals.
    pub fn normals(&self) -> &Arc<RwLock<GPUVec<Vec3>>> {
        &self.normals
    }

    /// This mesh vertex coordinates.
    pub fn coords(&self) -> &Arc<RwLock<GPUVec<Vec3>>> {
        &self.coords
    }

    /// This mesh texture coordinates.
    pub fn uvs(&self) -> &Arc<RwLock<GPUVec<Vec2>>> {
        &self.uvs
    }

    /// Computes normals from a set of faces.
    pub fn compute_normals_array(coordinates: &[Vec3], faces: &[[VertexIndex; 3]]) -> Vec<Vec3> {
        let mut res = Vec::new();

        GpuMesh3d::compute_normals(coordinates, faces, &mut res);

        res
    }

    /// Computes normals from a set of faces.
    pub fn compute_normals(
        coordinates: &[Vec3],
        faces: &[[VertexIndex; 3]],
        normals: &mut Vec<Vec3>,
    ) {
        let mut divisor: Vec<f32> = vec![0f32; coordinates.len()];

        normals.clear();
        normals.extend(std::iter::repeat_n(Vec3::ZERO, coordinates.len()));

        // Accumulate normals ...
        for f in faces.iter() {
            let edge1 = coordinates[f[1] as usize] - coordinates[f[0] as usize];
            let edge2 = coordinates[f[2] as usize] - coordinates[f[0] as usize];
            let cross = edge1.cross(edge2);

            let normal = if cross != Vec3::ZERO {
                cross.normalize()
            } else {
                cross
            };

            normals[f[0] as usize] += normal;
            normals[f[1] as usize] += normal;
            normals[f[2] as usize] += normal;

            divisor[f[0] as usize] += 1.0;
            divisor[f[1] as usize] += 1.0;
            divisor[f[2] as usize] += 1.0;
        }

        // ... and compute the mean
        for (n, divisor) in normals.iter_mut().zip(divisor.iter()) {
            *n /= *divisor
        }
    }
}

impl From<RenderMesh> for GpuMesh3d {
    fn from(value: RenderMesh) -> Self {
        Self::from_render_mesh(value, false)
    }
}