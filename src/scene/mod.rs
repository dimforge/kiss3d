//! Everything related to the scene graph.

pub use self::object::{
    InstanceData, InstancesBuffer, Object, ObjectData, LINES_COLOR_USE_OBJECT,
    LINES_WIDTH_USE_OBJECT, POINTS_COLOR_USE_OBJECT, POINTS_SIZE_USE_OBJECT,
};
pub use self::planar_object::{
    PlanarInstanceData, PlanarInstancesBuffer, PlanarObject, PlanarObjectData,
    PLANAR_LINES_COLOR_USE_OBJECT, PLANAR_LINES_WIDTH_USE_OBJECT, PLANAR_POINTS_COLOR_USE_OBJECT,
    PLANAR_POINTS_SIZE_USE_OBJECT,
};
pub use self::planar_scene_node::{PlanarSceneNode, PlanarSceneNodeData};
pub use self::scene_node::{SceneNode, SceneNodeData};

mod object;
mod planar_object;
mod planar_scene_node;
mod scene_node;
