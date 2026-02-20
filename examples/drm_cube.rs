#[cfg(feature = "drm")]
use kiss3d::prelude::*;
use log::info;

#[cfg(feature = "drm")]
#[kiss3d::main]
async fn main() {
    env_logger::init();
    let mut window = DRMWindow::new("/dev/dri/card0", 1024, 600)
        .await
        .expect("Failed to create DRM window");
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 2.0, -2.0));

    let mut c = scene.add_cube(1.0, 1.0, 1.0).set_color(RED);

    let rot = Quat::from_axis_angle(Vec3::Y, 0.014);

    let mut frame_count = 0;
    while window.render_3d(&mut scene, &mut camera).await {
        c.rotate(rot);

        // Save screenshot every 24 frames
        if frame_count % 24 == 0 {
            let image = window.snap_image();
            let filename = format!("frame_{:04}.png", frame_count);
            image.save(&filename).expect("Failed to save screenshot");
            info!("Saved {}", filename);
        }

        frame_count += 1;
    }
}

#[cfg(not(feature = "drm"))]
#[kiss3d::main]
async fn main() {
    info!("This example is supposed to be run with the featuere 'drm' enabled.");
}
