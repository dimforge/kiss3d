use kiss3d::prelude::*;
use kiss3d::window::DrmWindow;

// Shows a rotating cube on a display through DRM/KMS, with no window system (e.g. on a
// console-only Raspberry Pi). Run it from a text console, not from inside a desktop session.
// By default it picks the first `/dev/dri/card*` that can drive a display. Set `DRM_DEVICE`
// (e.g. `/dev/dri/card1`) to use a specific device.
#[kiss3d::main]
async fn main() {
    env_logger::init();
    let mut window = match std::env::var("DRM_DEVICE") {
        Ok(device) => DrmWindow::new(&device)
            .await
            .unwrap_or_else(|e| panic!("Failed to open {}: {}", device, e)),
        Err(_) => DrmWindow::new_auto()
            .await
            .unwrap_or_else(|e| panic!("Failed to find a DRM device: {}", e)),
    };
    // Shadow mapping dominates the frame time on small GPUs such as the Raspberry Pi 4.
    window.set_shadows_enabled(false);

    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 2.0, -2.0));

    let mut c = scene.add_cube(1.0, 1.0, 1.0).set_color(RED);
    let rot = Quat::from_axis_angle(Vec3::Y, 0.014);

    while window.render_3d(&mut scene, &mut camera).await {
        c.rotate(rot);
    }
}
