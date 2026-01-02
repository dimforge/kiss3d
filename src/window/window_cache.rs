use std::cell::RefCell;

use crate::resource::{MaterialManager3d, MeshManager3d, TextureManager};

#[derive(Default)]
/// Globally accessible cache of objects
pub(crate) struct WindowCache {
    pub(crate) mesh_manager: Option<MeshManager3d>,
    pub(crate) texture_manager: Option<TextureManager>,
    pub(crate) material_manager: Option<MaterialManager3d>,
}

thread_local!(pub(crate) static WINDOW_CACHE: RefCell<WindowCache>  = RefCell::new(WindowCache::default()));

impl WindowCache {
    /// Initialize resource managers
    pub fn populate() {
        WINDOW_CACHE.with(|cache| {
            cache.borrow_mut().mesh_manager = Some(MeshManager3d::new());
            cache.borrow_mut().texture_manager = Some(TextureManager::new());
            cache.borrow_mut().material_manager = Some(MaterialManager3d::new());
        });
    }

    /// Reset all cached managers, releasing GPU resources.
    ///
    /// This should be called before thread-local storage destruction begins
    /// to avoid TLS access order issues with wgpu internals.
    pub fn reset() {
        WINDOW_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.mesh_manager = None;
            cache.texture_manager = None;
            cache.material_manager = None;
        });
    }
}
