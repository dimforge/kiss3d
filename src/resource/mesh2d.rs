//! Data structure of a scene node geometry.

use std::sync::{Arc, RwLock};

use crate::resource::gpu_vector::{AllocationType, BufferType, GPUVec};
use crate::resource::vertex_index::VertexIndex;
use glamx::Vec2;

/// Aggregation of vertices, indices, and texture coordinates for 2D meshes.
///
/// It also contains the GPU location of those buffers.
pub struct GpuMesh2d {
    coords: Arc<RwLock<GPUVec<Vec2>>>,
    faces: Arc<RwLock<GPUVec<[VertexIndex; 3]>>>,
    uvs: Arc<RwLock<GPUVec<Vec2>>>,
    edges: Option<Arc<RwLock<GPUVec<[VertexIndex; 2]>>>>,
}

impl GpuMesh2d {
    /// Creates a new mesh.
    ///
    /// If the uvs are not given, they are automatically computed as origin.
    pub fn new(
        coords: Vec<Vec2>,
        faces: Vec<[VertexIndex; 3]>,
        uvs: Option<Vec<Vec2>>,
        dynamic_draw: bool,
    ) -> GpuMesh2d {
        let uvs = match uvs {
            Some(us) => us,
            None => std::iter::repeat_n(Vec2::ZERO, coords.len()).collect(),
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
        let us = Arc::new(RwLock::new(GPUVec::new(uvs, BufferType::Array, location)));

        GpuMesh2d::new_with_gpu_vectors(cs, fs, us)
    }

    /// Creates a new mesh. Arguments set to `None` are automatically computed.
    pub fn new_with_gpu_vectors(
        coords: Arc<RwLock<GPUVec<Vec2>>>,
        faces: Arc<RwLock<GPUVec<[VertexIndex; 3]>>>,
        uvs: Arc<RwLock<GPUVec<Vec2>>>,
    ) -> GpuMesh2d {
        GpuMesh2d {
            coords,
            faces,
            uvs,
            edges: None,
        }
    }

    /// Ensures all mesh buffers are loaded to the GPU.
    pub fn load_to_gpu(&mut self) {
        self.coords.write().unwrap().load_to_gpu();
        self.uvs.write().unwrap().load_to_gpu();
        self.faces.write().unwrap().load_to_gpu();
    }

    /// Creates and loads edge buffer to GPU.
    pub fn ensure_edges_on_gpu(&mut self) {
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

        self.edges.as_mut().unwrap().write().unwrap().load_to_gpu();
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

    /// This mesh faces.
    pub fn faces(&self) -> &Arc<RwLock<GPUVec<[VertexIndex; 3]>>> {
        &self.faces
    }

    /// This mesh vertex coordinates.
    pub fn coords(&self) -> &Arc<RwLock<GPUVec<Vec2>>> {
        &self.coords
    }

    /// This mesh texture coordinates.
    pub fn uvs(&self) -> &Arc<RwLock<GPUVec<Vec2>>> {
        &self.uvs
    }

    /// This mesh edges.
    pub fn edges(&self) -> &Option<Arc<RwLock<GPUVec<[VertexIndex; 2]>>>> {
        &self.edges
    }
}
