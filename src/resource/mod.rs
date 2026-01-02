//! GPU resource managers

pub use crate::resource::dynamic_buffer::DynamicUniformBuffer;
pub use crate::resource::framebuffer_manager::{
    FramebufferManager, OffscreenBuffers, RenderTarget,
};
pub use crate::resource::gpu_vector::{AllocationType, BufferType, GPUVec};
pub use crate::resource::material::{
    GpuData, Material3d, Material2d, RenderContext, RenderContext2d, RenderContext2dEncoder,
};
pub use crate::resource::material_manager3d::MaterialManager3d;
pub use crate::resource::material_manager2d::MaterialManager2d;
pub use crate::resource::mesh3d::GpuMesh3d;
pub use crate::resource::mesh2d::GpuMesh2d;
pub use crate::resource::mesh_manager3d::MeshManager3d;
pub use crate::resource::mesh_manager2d::MeshManager2d;
pub use crate::resource::texture_manager::{Texture, TextureManager, TextureWrapping};

mod dynamic_buffer;
mod framebuffer_manager;
mod gpu_vector;
pub mod material;
mod material_manager3d;
mod material_manager2d;
mod mesh3d;
mod mesh2d;
mod mesh_manager3d;
mod mesh_manager2d;
mod texture_manager;
pub mod vertex_index;
