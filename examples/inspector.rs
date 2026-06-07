#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!("The 'egui' feature must be enabled for this example to work.")
}

// The whole point of the built-in inspector: a varied scene, and a *single*
// line inside the render loop that overlays a panel to configure every
// rendering knob, toggle the path tracer on/off, and edit the scene tree.
#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    use kiss3d::prelude::*;
    use kiss3d::window::Inspector;

    let mut window = Window::new("Kiss3d: inspector").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(3.0, 2.5, 4.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();

    window.set_background_color(LIGHT_STEEL_BLUE);

    // A ground plane and a few objects with different materials, grouped so the
    // scene tree has some structure to explore.
    scene
        .add_quad(8.0, 8.0, 1, 1)
        .set_color(GRAY)
        .append_rotation(Quat::from_axis_angle(Vec3::X, -std::f32::consts::FRAC_PI_2))
        .set_position(Vec3::new(0.0, -1.5, 0.0));

    let mut shapes = scene.add_group();
    shapes
        .add_cube(1.0, 1.0, 1.0)
        .set_color(CRIMSON)
        .set_position(Vec3::new(-1.5, 0.0, 0.0));
    shapes
        .add_sphere(0.6)
        .set_color(WHITE)
        .set_metallic(1.0)
        .set_roughness(0.1)
        .set_position(Vec3::new(0.0, 0.1, 0.0));
    shapes
        .add_cone(0.6, 1.2)
        .set_color(GOLD)
        .set_position(Vec3::new(1.5, 0.0, 0.0));

    scene
        .add_point_light(100.0)
        .set_position(Vec3::new(3.0, 5.0, 3.0));
    scene.add_directional_light(Vec3::new(-1.0, -1.0, -0.5));

    // You own the inspector; keep it alive across frames so its UI state persists.
    let mut inspector = Inspector::new();

    while window.render_3d(&mut scene, &mut camera).await {
        // Draw the inspector overlay. Pass `Some(&mut scene_2d)` as the third
        // argument to also inspect a 2D scene.
        window.draw_inspector(&mut inspector, Some(&mut scene), None, None);
    }
}
