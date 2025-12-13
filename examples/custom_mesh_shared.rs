extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::light::Light;
use kiss3d::resource::{GpuMesh, MeshManager};
use kiss3d::window::Window;
use na::{Point3, UnitQuaternion, Vector3};
use std::cell::RefCell;
use std::rc::Rc;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: custom_mesh_shared").await;

    let a = Point3::new(-1.0, -1.0, 0.0);
    let b = Point3::new(1.0, -1.0, 0.0);
    let c = Point3::new(0.0, 1.0, 0.0);

    let vertices = vec![a, b, c];
    let indices = vec![Point3::new(0, 1, 2)];

    let mesh = Rc::new(RefCell::new(GpuMesh::new(
        vertices, indices, None, None, false,
    )));

    // XXX: it would be better to do: MeshManager::add(Rc....) directly.
    MeshManager::get_global_manager(|mm| mm.add(mesh.clone(), "custom_mesh"));

    let mut c1 = window
        .add_geom_with_name("custom_mesh", Vector3::new(1.0, 1.0, 1.0))
        .unwrap();
    let mut c2 = window
        .add_geom_with_name("custom_mesh", Vector3::new(1.0, 1.0, 1.0))
        .unwrap();

    c1.set_color(1.0, 0.0, 0.0);
    c2.set_color(0.0, 1.0, 0.0);

    c1.enable_backface_culling(false);
    c2.enable_backface_culling(false);

    window.set_light(Light::StickToCamera);

    let rot1 = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);
    let rot2 = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -0.014);

    while window.render().await {
        c1.prepend_to_local_rotation(&rot1);
        c2.prepend_to_local_rotation(&rot2);
    }
}
