//! A resource manager to load materials.

use crate::builtin::ObjectMaterial2d;
use crate::resource::Material2d;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

thread_local!(static KEY_MATERIAL_MANAGER: RefCell<MaterialManager2d> = RefCell::new(MaterialManager2d::new()));

/// The material manager.
///
/// Upon construction, it contains:
/// * the `object` material, used as the default to render objects.
/// * the `normals` material, used do display an object normals.
///
/// It keeps a cache of already-loaded materials. Note that this is only a cache, nothing more.
/// Thus, its usage is not required to load materials.
pub struct MaterialManager2d {
    default_material: Rc<RefCell<Box<dyn Material2d + 'static>>>,
    materials: HashMap<String, Rc<RefCell<Box<dyn Material2d + 'static>>>>,
}

impl Default for MaterialManager2d {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialManager2d {
    /// Creates a new material manager.
    pub fn new() -> MaterialManager2d {
        // load the default ObjectMaterial and the LineMaterial
        let mut materials = HashMap::new();

        let om = Rc::new(RefCell::new(
            Box::new(ObjectMaterial2d::new()) as Box<dyn Material2d + 'static>
        ));
        let _ = materials.insert("object".to_string(), om.clone());

        MaterialManager2d {
            default_material: om,
            materials,
        }
    }

    /// Mutably applies a function to the material manager.
    pub fn get_global_manager<T, F: FnMut(&mut MaterialManager2d) -> T>(mut f: F) -> T {
        KEY_MATERIAL_MANAGER.with(|manager| f(&mut manager.borrow_mut()))
    }

    /// Gets the default material to draw objects.
    pub fn get_default(&self) -> Rc<RefCell<Box<dyn Material2d + 'static>>> {
        self.default_material.clone()
    }

    /// Get a material with the specified name. Returns `None` if the material is not registered.
    pub fn get(&mut self, name: &str) -> Option<Rc<RefCell<Box<dyn Material2d + 'static>>>> {
        self.materials.get(name).cloned()
    }

    /// Adds a material with the specified name to this cache.
    pub fn add(&mut self, material: Rc<RefCell<Box<dyn Material2d + 'static>>>, name: &str) {
        let _ = self.materials.insert(name.to_string(), material);
    }

    /// Removes a mesh from this cache.
    pub fn remove(&mut self, name: &str) {
        let _ = self.materials.remove(name);
    }

    /// Prepares all materials for a new frame.
    ///
    /// This clears internal buffers and prepares for accumulating new uniform data.
    /// Should be called at the start of each frame before any objects are prepared.
    pub fn begin_frame(&mut self) {
        for material in self.materials.values() {
            material.borrow_mut().begin_frame();
        }
    }

    /// Flushes all accumulated uniform data to the GPU.
    ///
    /// This should be called after all objects have been prepared (via
    /// `Material2d::prepare()`) and before rendering. It uploads the batched
    /// uniform data in a single operation.
    pub fn flush(&mut self) {
        for material in self.materials.values() {
            material.borrow_mut().flush();
        }
    }

    /// Resets the global material manager, releasing all GPU resources.
    ///
    /// This should be called before thread-local storage destruction begins
    /// to avoid TLS access order issues with wgpu internals.
    pub fn reset_global_manager() {
        KEY_MATERIAL_MANAGER.with(|manager| {
            let mut manager = manager.borrow_mut();
            manager.materials.clear();
            // Recreate default material to satisfy type requirements
            // (but it will be unused since context is being reset)
        });
    }
}
