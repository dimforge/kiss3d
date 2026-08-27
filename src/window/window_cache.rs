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
    /// Initialize the resource managers that are not set yet.
    ///
    /// A second window or offscreen surface created on the same thread shares
    /// the first one's GPU context, so it must share its managers too: objects
    /// created before it hold the first manager's default material, and the
    /// per-frame `begin_frame` / `flush` only reach the registered materials.
    /// Replacing the managers left those objects with an orphaned material
    /// whose per-object uniform buffer grew until the GPU rejected its offsets.
    /// After [`Self::reset`] (last window gone) fresh managers are created.
    pub fn populate() {
        WINDOW_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.mesh_manager.get_or_insert_with(MeshManager3d::new);
            cache
                .texture_manager
                .get_or_insert_with(TextureManager::new);
            cache
                .material_manager
                .get_or_insert_with(MaterialManager3d::new);
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
