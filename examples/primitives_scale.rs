extern crate kiss3d;
extern crate nalgebra as na;
extern crate rand;

use kiss3d::light::Light;
use kiss3d::window::Window;
use na::{Translation3, UnitQuaternion, Vector3};
use rand::random;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: primitives_scale").await;

    // NOTE: scaling is not possible.
    for i in 0usize..11 {
        let dim: f32 = random::<f32>() / 2.0;
        let dim2 = dim / 2.0;

        let offset = i as f32 * 1.0 - 5.0;

        let mut cu = window.add_cube(dim2, dim2, dim2);
        let mut sp = window.add_sphere(dim2);
        let mut co = window.add_cone(dim2, dim);
        let mut cy = window.add_cylinder(dim2, dim);
        let mut ca = window.add_capsule(dim2, dim);

        cu.append_translation(&Translation3::new(offset, 1.0, 0.0));
        sp.append_translation(&Translation3::new(offset, -1.0, 0.0));
        co.append_translation(&Translation3::new(offset, 2.0, 0.0));
        cy.append_translation(&Translation3::new(offset, -2.0, 0.0));
        ca.append_translation(&Translation3::new(offset, 0.0, 0.0));

        cu.set_color(random(), random(), random());
        sp.set_color(random(), random(), random());
        co.set_color(random(), random(), random());
        cy.set_color(random(), random(), random());
        ca.set_color(random(), random(), random());
    }

    window.set_light(Light::StickToCamera);

    let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);

    while window.render().await {
        // XXX: applying this to each object individually became complicated…
        window.scene_mut().append_rotation_wrt_center(&rot);
    }
}
