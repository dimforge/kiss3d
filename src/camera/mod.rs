//! Camera trait with some common implementations.

pub use self::camera2d::Camera2d;
pub use self::camera3d::Camera3d;
pub use self::first_person3d::FirstPersonCamera3d;
pub use self::first_person_stereo3d::FirstPersonCamera3dStereo;
pub use self::fixed_view2d::FixedView2d;
pub use self::fixed_view3d::FixedView3d;
pub use self::orbit3d::OrbitCamera3d;
pub use self::sidescroll2d::PanZoomCamera2d;

mod camera2d;
mod camera3d;
mod first_person3d;
mod first_person_stereo3d;
mod fixed_view2d;
mod fixed_view3d;
mod orbit3d;
mod sidescroll2d;
