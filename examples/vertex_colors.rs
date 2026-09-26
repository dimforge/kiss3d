//! Per-vertex colours: a mesh that carries one colour per vertex, which the
//! default material multiplies into its base colour.
//!
//! This is the attribute the glTF loader fills in from `COLOR_0`, so a model
//! authored with painted vertices arrives already tinted. A mesh that carries
//! none is unaffected and costs no extra buffer.

use kiss3d::prelude::*;

/// A flat quad of half-extent `half`, wound counter-clockwise.
fn quad(half: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    (
        vec![
            Vec3::new(-half, -half, 0.0),
            Vec3::new(half, -half, 0.0),
            Vec3::new(half, half, 0.0),
            Vec3::new(-half, half, 0.0),
        ],
        vec![[0, 1, 2], [0, 2, 3]],
    )
}

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: vertex_colors").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(120.0))
        .set_position(Vec3::new(-2.0, 4.0, -6.0));

    // One colour per corner; the gradient across each triangle is the
    // rasterizer interpolating between them.
    let (vertices, indices) = quad(0.7);
    let mut colored = GpuMesh3d::new(vertices, indices, None, None, false);
    colored.set_colors(vec![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ]);
    let mut painted = scene
        .add_mesh(Rc::new(RefCell::new(colored)), Vec3::ONE)
        .enable_backface_culling(false);
    painted.set_position(Vec3::new(-0.9, 0.0, 0.0));

    // The same quad without colours, for comparison: `set_color` alone still
    // decides the whole surface.
    let (vertices, indices) = quad(0.5);
    let mut plain = scene
        .add_mesh(
            Rc::new(RefCell::new(GpuMesh3d::new(
                vertices, indices, None, None, false,
            ))),
            Vec3::ONE,
        )
        .set_color(ORANGE)
        .enable_backface_culling(false);
    plain.set_position(Vec3::new(0.9, 0.0, 0.0));

    let rot = Quat::from_axis_angle(Vec3::Y, 0.008);

    while window.render_3d(&mut scene, &mut camera).await {
        painted.rotate(rot);
        plain.rotate(rot);
    }
}
