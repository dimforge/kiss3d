//! Built-in geometries, shaders and effects.

pub use self::aov::{
    AovKind, AovRenderer, DEPTH_AOV_FORMAT, NORMALS_AOV_FORMAT, SEGMENTATION_AOV_FORMAT,
};
pub use self::normals_material::{NormalsMaterial, NORMAL_FRAGMENT_SRC, NORMAL_VERTEX_SRC};
pub use self::object_material::{ObjectMaterial, OBJECT_FRAGMENT_SRC, OBJECT_VERTEX_SRC};
pub use self::uvs_material::{UvsMaterial, UVS_FRAGMENT_SRC, UVS_VERTEX_SRC};

pub use self::object_material2d::ObjectMaterial2d;
pub use self::shadow::{ShadowMapper, MAX_SHADOW_VIEWS};

mod aov;
mod normals_material;
mod object_material;
mod shadow;
mod uvs_material;

mod object_material2d;
