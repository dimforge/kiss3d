/// This example demonstrates how to record a screencast of the 3D scene.
///
/// Requires the `recording` feature to be enabled:
/// ```
/// cargo run --example recording --features recording
/// ```
#[kiss3d::main]
#[cfg(feature = "recording")]
async fn main() {
    use kiss3d::prelude::*;

    let mut window = Window::new("Kiss3d: recording").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));
    let mut c = scene.add_cube(0.2, 0.2, 0.2).set_color(RED);

    // Option 1: Simple recording (every frame)
    // window.begin_recording();

    // Option 2: Record every 2nd frame to reduce file size
    let config = RecordingConfig::new().with_frame_skip(2);
    window.begin_recording_with_config(config);

    println!("Recording started (every 2nd frame)...");

    let rot = Quat::from_axis_angle(Vec3::Y, 0.02);

    // Record 90 frames (3 seconds at 30fps, or 1.5 seconds with frame_skip=2)
    #[allow(unused_variables)]
    for frame in 0..90 {
        c.rotate(&rot);

        // Demonstrate pause/resume at frame 30-60
        if frame == 30 {
            window.pause_recording();
            println!("Recording paused at frame 30...");
        }
        if frame == 60 {
            window.resume_recording();
            println!("Recording resumed at frame 60...");
        }

        if !window.render_3d(&mut scene, &mut camera).await {
            break;
        }

        if frame % 30 == 0 {
            println!("Frame {}...", frame);
        }
    }

    // Stop recording and save to file
    println!("Encoding video...");
    match window.end_recording("recording.mp4", 30) {
        Ok(()) => println!("Video saved to `recording.mp4`"),
        Err(e) => eprintln!("Failed to save video: {}", e),
    }
}

#[kiss3d::main]
#[cfg(not(feature = "recording"))]
async fn main() {
    eprintln!("Recording feature is not enabled!\nRun with: cargo run --example recording --features recording");
}
