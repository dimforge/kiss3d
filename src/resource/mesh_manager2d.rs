//! A resource manager to load meshes.
use std::f32;

use glamx::Vec2;

use crate::resource::vertex_index::VertexIndex;
use crate::resource::GpuMesh2d;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

thread_local!(static KEY_MESH_MANAGER: RefCell<MeshManager2d> = RefCell::new(MeshManager2d::new()));

/// The mesh manager.
///
/// Upon construction, it contains:
///
/// It keeps a cache of already-loaded meshes. Note that this is only a cache, nothing more.
/// Thus, its usage is not required to load meshes.
pub struct MeshManager2d {
    meshes: HashMap<String, Rc<RefCell<GpuMesh2d>>>,
}

impl Default for MeshManager2d {
    fn default() -> Self {
        Self::new()
    }
}

impl MeshManager2d {
    /// Creates a new mesh manager.
    pub fn new() -> MeshManager2d {
        let mut res = MeshManager2d {
            meshes: HashMap::new(),
        };

        /*
         * Rectangle geometry.
         */
        let rect_vtx = vec![
            Vec2::new(0.5, 0.5),
            Vec2::new(-0.5, -0.5),
            Vec2::new(-0.5, 0.5),
            Vec2::new(0.5, -0.5),
        ];
        let rect_uvs = vec![
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 1.0),
        ];

        let rect_ids = vec![[0, 1, 2], [1, 0, 3]];
        let rect = GpuMesh2d::new(rect_vtx, rect_ids, Some(rect_uvs), false);
        res.add(Rc::new(RefCell::new(rect)), "rectangle");

        /*
         * Circle geometry.
         */
        let mut circle_vtx = vec![Vec2::ZERO];
        let mut circle_ids = Vec::new();
        let nsamples = 50;

        for i in 0..nsamples {
            let ang = (i as f32) / (nsamples as f32) * f32::consts::PI * 2.0;
            circle_vtx.push(Vec2::new(ang.cos(), ang.sin()) * 0.5);
            circle_ids.push([
                0,
                circle_vtx.len() as VertexIndex - 2,
                circle_vtx.len() as VertexIndex - 1,
            ]);
        }
        circle_ids.push([0, circle_vtx.len() as VertexIndex - 1, 1]);

        let circle = GpuMesh2d::new(circle_vtx, circle_ids, None, false);
        res.add(Rc::new(RefCell::new(circle)), "circle");

        res
    }

    /// Mutably applies a function to the mesh manager.
    pub fn get_global_manager<T, F: FnMut(&mut MeshManager2d) -> T>(mut f: F) -> T {
        KEY_MESH_MANAGER.with(|manager| f(&mut manager.borrow_mut()))
    }

    /// Get a mesh with the specified name. Returns `None` if the mesh is not registered.
    pub fn get(&mut self, name: &str) -> Option<Rc<RefCell<GpuMesh2d>>> {
        self.meshes.get(name).cloned()
    }

    /// Adds a mesh with the specified name to this cache.
    pub fn add(&mut self, mesh: Rc<RefCell<GpuMesh2d>>, name: &str) {
        let _ = self.meshes.insert(name.to_string(), mesh);
    }

    /// Removes a mesh from this cache.
    pub fn remove(&mut self, name: &str) {
        let _ = self.meshes.remove(name);
    }

    /// Resets the global mesh manager, releasing all GPU resources.
    ///
    /// This should be called before thread-local storage destruction begins
    /// to avoid TLS access order issues with wgpu internals.
    pub fn reset_global_manager() {
        KEY_MESH_MANAGER.with(|manager| {
            manager.borrow_mut().meshes.clear();
        });
    }
}
