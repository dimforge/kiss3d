//! GPU resource managers

pub use crate::resource::framebuffer_manager::{
    FramebufferManager, OffscreenBuffers, RenderTarget,
};
pub use crate::resource::gpu_vector::{AllocationType, BufferType, GPUVec};
pub use crate::resource::material::{
    GpuData, Material, PlanarMaterial, PlanarRenderContext, RenderContext,
};
pub use crate::resource::material_manager::MaterialManager;
pub use crate::resource::mesh::GpuMesh;
pub use crate::resource::mesh_manager::MeshManager;
pub use crate::resource::planar_material_manager::PlanarMaterialManager;
pub use crate::resource::planar_mesh::PlanarMesh;
pub use crate::resource::planar_mesh_manager::PlanarMeshManager;
pub use crate::resource::texture_manager::{Texture, TextureManager, TextureWrapping};

mod framebuffer_manager;
mod gpu_vector;
pub mod material;
mod material_manager;
mod mesh;
mod mesh_manager;
mod planar_material_manager;
mod planar_mesh;
mod planar_mesh_manager;
mod texture_manager;
pub mod vertex_index;
