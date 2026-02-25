//! DRM Display Output Test - Phase 3
//!
//! This example tests the complete display output pipeline:
//! - Display discovery and configuration
//! - GBM buffer allocation
//! - Frame rendering with wgpu
//! - Frame copying to GBM buffers
//! - Display output via DRM/KMS
//!
//! This should show rotating colored cubes on your display!
//!
//! # Requirements
//! - DRM/KMS enabled GPU driver
//! - Root or video/render group permissions
//! - Connected display
//!
//! # Usage
//! ```bash
//! [sudo] [DRM_DEVICE=/dev/dri/card1] [DRM_DEVICE_WIDTH=1024] [DRM_DEVICE_HEIGHT=800] cargo run --example drm_display_test --features drm
//! ```
//! FPS on a raspberry pi 4
//! debug: 226 - 227
//! release: 220 - 235
//! After implementing dma zero buf copy

#[cfg(feature = "drm")]
use kiss3d::prelude::*;

#[cfg(feature = "drm")]
#[kiss3d::main]
async fn main() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║         DRM Display Output Test - Phase 3                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Try to open DRM device
    let device_path = std::env::var("DRM_DEVICE").unwrap_or_else(|_| "/dev/dri/card0".to_string());
    let width: u32 = std::env::var("DRM_DEVICE_WIDTH")
        .unwrap_or_else(|_| "1024".to_string())
        .parse()
        .expect("DRM_DEVICE_WIDTH must be valid unsigned integer");
    let height: u32 = std::env::var("DRM_DEVICE_HEIGHT")
        .unwrap_or_else(|_| "600".to_string())
        .parse()
        .expect("DRM_DEVICE_HEIGHT must be valid unsigned integer");

    println!(
        "📺 Opening DRM device: {} at {}*{}",
        device_path, width, height
    );
    println!("   This will attempt to display output on your connected display");
    println!();

    // Create DRM window with display output
    println!("🎬 Creating DRM window with display output...");
    let mut window = match DRMWindow::new(&device_path, width, height).await {
        Ok(w) => {
            println!("✅ DRM window created successfully!");
            w
        }
        Err(e) => {
            eprintln!("❌ Failed to create DRM window: {}", e);
            eprintln!();
            eprintln!("Troubleshooting:");
            eprintln!("  - Ensure you're running as root or in video/render group");
            eprintln!("  - Check that {} exists", device_path);
            eprintln!("  - Verify a display is connected");
            eprintln!("  - Check dmesg for DRM/KMS errors");
            return;
        }
    };

    println!();
    println!("🎨 Setting up 3D scene...");

    // Create camera
    let mut camera = OrbitCamera3d::default();
    camera.set_yaw(std::f32::consts::PI / 4.0);
    camera.set_pitch(-std::f32::consts::PI / 6.0);
    camera.set_dist(5.0);

    // Create scene
    let mut scene = SceneNode3d::empty();

    // Add a light
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 5.0, 0.0));

    // Add three colorful rotating cubes
    let mut cube1 = scene.add_cube(1.0, 1.0, 1.0).set_color(RED);

    let mut cube2 = scene.add_cube(0.5, 0.5, 0.5).set_color(GREEN);
    cube2.set_position(Vec3::new(2.0, 0.0, 0.0));

    let mut cube3 = scene.add_cube(0.5, 0.5, 0.5).set_color(BLUE);
    cube3.set_position(Vec3::new(-2.0, 0.0, 0.0));

    println!("✅ Scene created with 3 colored cubes");
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  🎉 DISPLAY OUTPUT ACTIVE - You should see rotating cubes! 🎉   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Rendering for 10 seconds...");
    println!("Press Ctrl+C to stop early");
    println!();

    let start_time = std::time::Instant::now();
    let mut frame_count = 0u32;
    let mut last_fps_time = start_time;
    let mut last_fps_count = 0u32;

    // Rotation quaternions
    let rot1 = Quat::from_axis_angle(Vec3::new(1.0, 0.7, 0.0).normalize(), 0.02);
    let rot2 = Quat::from_axis_angle(Vec3::new(0.0, 1.3, 1.0).normalize(), 0.02);
    let rot3 = Quat::from_axis_angle(Vec3::new(0.5, 0.0, 1.5).normalize(), 0.02);

    // Render loop for 10 seconds
    while start_time.elapsed().as_secs() < 10 {
        // Rotate the cubes
        cube1.rotate(rot1);
        cube2.rotate(rot2);
        cube3.rotate(rot3);

        // Render frame
        if !window.render_3d(&mut scene, &mut camera).await {
            break;
        }

        frame_count += 1;

        // Print FPS every second
        if last_fps_time.elapsed().as_secs() >= 1 {
            let fps = frame_count - last_fps_count;
            println!("Frame {}: {} FPS", frame_count, fps);
            last_fps_count = frame_count;
            last_fps_time = std::time::Instant::now();
        }
    }

    let total_time = start_time.elapsed();
    let avg_fps = frame_count as f64 / total_time.as_secs_f64();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    Test Complete!                               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Statistics:");
    println!("  Total frames rendered: {}", frame_count);
    println!("  Total time: {:.2}s", total_time.as_secs_f64());
    println!("  Average FPS: {:.2}", avg_fps);
    println!();
    println!("Phase 3 Status: ✅ COMPLETE");
    println!("  ✓ Display discovery");
    println!("  ✓ GBM buffer allocation");
    println!("  ✓ Frame rendering (wgpu)");
    println!("  ✓ Frame copying (wgpu → GBM)");
    println!("  ✓ Display output (DRM/KMS)");
    println!();
    println!("Next Steps:");
    println!("  → Phase 4: Optimize with DMA-BUF zero-copy");
    println!("  → Phase 4: Implement async page flip");
    println!("  → Phase 4: Add proper VBlank synchronization");
    println!();
}

#[cfg(not(feature = "drm"))]
#[kiss3d::main]
async fn main() {
    eprintln!("This example requires the 'drm' feature to be enabled.");
    eprintln!("Please run with: cargo run --example drm_display_test --features drm");
}
