//! Specify the type used for vertex indices, which default to `u16` for wasm compatibility
//! reasons. If you need more than 65535 vertices, enable the `vertex_index_u32` feature.
pub use inner::*;

#[cfg(not(feature = "vertex_index_u32"))]
mod inner {
    /// Defaults to `u16`. If you need more than 65535 vertices, enable the `vertex_index_u32` feature.
    pub type VertexIndex = u16;
    /// The wgpu IndexFormat for the vertex index type.
    pub const VERTEX_INDEX_FORMAT: wgpu::IndexFormat = wgpu::IndexFormat::Uint16;
}

#[cfg(feature = "vertex_index_u32")]
mod inner {
    /// The type used for vertex indices. The feature `vertex_index_u32` enables `u32` indices.
    pub type VertexIndex = u32;
    /// The wgpu IndexFormat for the vertex index type.
    pub const VERTEX_INDEX_FORMAT: wgpu::IndexFormat = wgpu::IndexFormat::Uint32;
}
