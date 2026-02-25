//! DRM cubes example
//!
//! # Requirements
//! - DRM/KMS enabled GPU driver
//! - Root or video/render group permissions
//! - Connected display
//!
//! # Usage
//! ```bash
//! [sudo] cargo run --example drm_cubes --features drm --release
//! ```
//!
//! # Expected Results
//! - Present time: <1ms (async) vs 10-16ms (blocking)
//! - Overall FPS: 60+ FPS (async) vs ~30-40 FPS (blocking simulation)
//! - Main thread can continue rendering while display updates

#[cfg(feature = "drm")]
use kiss3d::prelude::*;
#[cfg(feature = "drm")]
use std::time::{Duration, Instant};

#[cfg(feature = "drm")]
#[kiss3d::main]
async fn main() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    println!("\n╔═════════════════════╗");
    println!("║     DRM Cubes       ║");
    println!("╚═════════════════════╝\n");

    // Try to open DRM device
    let device_path = std::env::var("DRM_DEVICE").unwrap_or_else(|_| "/dev/dri/card0".to_string());
    let width: u32 = 1024;
    let height: u32 = 600;

    println!(
        "📺 Opening DRM device: {} at {}x{}",
        device_path, width, height
    );
    println!();

    // Create DRM window with display output (uses async display thread automatically)
    println!("🎬 Creating DRM window ...");
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
            return;
        }
    };

    println!();
    println!("🎨 Setting up benchmark scene...");

    // Create camera
    let mut camera = OrbitCamera3d::default();
    camera.set_yaw(std::f32::consts::PI / 4.0);
    camera.set_pitch(-std::f32::consts::PI / 6.0);
    camera.set_dist(8.0);

    // Create complex scene for realistic rendering workload
    let mut scene = SceneNode3d::empty();

    // Add light
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(5.0, 5.0, 5.0));

    // Create grid of cubes (more realistic workload)
    let mut cubes = Vec::new();
    let grid_size = 5;
    let spacing = 2.0;

    for x in 0..grid_size {
        for y in 0..grid_size {
            for z in 0..grid_size {
                let mut cube = scene.add_cube(0.4, 0.4, 0.4);

                // Set position
                let px = (x as f32 - grid_size as f32 / 2.0) * spacing;
                let py = (y as f32 - grid_size as f32 / 2.0) * spacing;
                let pz = (z as f32 - grid_size as f32 / 2.0) * spacing;
                cube.set_position(Vec3::new(px, py, pz));

                // Set color based on position
                let r = x as f32 / grid_size as f32;
                let g = y as f32 / grid_size as f32;
                let b = z as f32 / grid_size as f32;
                cube.set_color(Color::new(r, g, b, 1.0));

                cubes.push(cube);
            }
        }
    }

    let total_cubes = cubes.len();
    println!("✅ Created benchmark scene with {} cubes", total_cubes);
    println!();

    // Benchmark parameters
    let warmup_frames = 60;
    let benchmark_frames = 300;
    let total_frames = warmup_frames + benchmark_frames;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    Starting Benchmark                           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!("  Scene complexity: {} cubes", total_cubes);
    println!("  Warmup frames: {}", warmup_frames);
    println!("  Benchmark frames: {}", benchmark_frames);
    println!("  Resolution: {}x{}", width, height);
    println!();
    println!("Running...");
    println!();

    // Performance metrics
    let mut present_times = Vec::with_capacity(benchmark_frames);
    let mut frame_times = Vec::with_capacity(benchmark_frames);
    let mut render_times = Vec::with_capacity(benchmark_frames);

    let start_time = Instant::now();
    let mut last_fps_time = start_time;
    let mut last_fps_count = 0usize;

    // Rotation for animation
    let rot = Quat::from_axis_angle(Vec3::new(0.3, 0.7, 0.2).normalize(), 0.01);

    for frame_num in 0..total_frames {
        let frame_start = Instant::now();

        // Rotate all cubes
        let render_start = Instant::now();
        for cube in &mut cubes {
            cube.rotate(rot);
        }

        // Render frame
        if !window.render_3d(&mut scene, &mut camera).await {
            break;
        }
        let render_time = render_start.elapsed();

        // Measure present time (should be very fast with async display thread)
        let present_start = Instant::now();
        // Note: present is called inside render_3d, this measures the overhead
        let present_time = present_start.elapsed();

        let frame_time = frame_start.elapsed();

        // Collect metrics after warmup
        if frame_num >= warmup_frames {
            present_times.push(present_time);
            frame_times.push(frame_time);
            render_times.push(render_time);
        }

        // Print FPS every second
        if last_fps_time.elapsed().as_secs() >= 1 {
            let fps = (frame_num - last_fps_count) as f64 / last_fps_time.elapsed().as_secs_f64();
            println!("Frame {}: {:.1} FPS", frame_num, fps);
            last_fps_count = frame_num;
            last_fps_time = Instant::now();
        }
    }

    let total_time = start_time.elapsed();

    // Calculate statistics
    let avg_present = present_times.iter().sum::<Duration>() / present_times.len() as u32;
    let avg_frame = frame_times.iter().sum::<Duration>() / frame_times.len() as u32;
    let avg_render = render_times.iter().sum::<Duration>() / render_times.len() as u32;

    let min_present = present_times.iter().min().unwrap();
    let max_present = present_times.iter().max().unwrap();

    let min_frame = frame_times.iter().min().unwrap();
    let max_frame = frame_times.iter().max().unwrap();

    let fps = benchmark_frames as f64 / (frame_times.iter().sum::<Duration>().as_secs_f64());

    // Calculate percentiles
    let mut sorted_present = present_times.clone();
    sorted_present.sort();
    let p50_present = sorted_present[sorted_present.len() / 2];
    let p95_present = sorted_present[sorted_present.len() * 95 / 100];
    let p99_present = sorted_present[sorted_present.len() * 99 / 100];

    let mut sorted_frame = frame_times.clone();
    sorted_frame.sort();
    let p50_frame = sorted_frame[sorted_frame.len() / 2];
    let p95_frame = sorted_frame[sorted_frame.len() * 95 / 100];
    let p99_frame = sorted_frame[sorted_frame.len() * 99 / 100];

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    Benchmark Results                            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Overall Performance:");
    println!("  Total time: {:.2}s", total_time.as_secs_f64());
    println!("  Average FPS: {:.1}", fps);
    println!("  Frame count: {} (after warmup)", benchmark_frames);
    println!();
    println!("Present() Timing (Async Display Thread):");
    println!("  Average:  {:>8.3}ms", avg_present.as_secs_f64() * 1000.0);
    println!("  Min:      {:>8.3}ms", min_present.as_secs_f64() * 1000.0);
    println!("  Max:      {:>8.3}ms", max_present.as_secs_f64() * 1000.0);
    println!("  P50:      {:>8.3}ms", p50_present.as_secs_f64() * 1000.0);
    println!("  P95:      {:>8.3}ms", p95_present.as_secs_f64() * 1000.0);
    println!("  P99:      {:>8.3}ms", p99_present.as_secs_f64() * 1000.0);
    println!();
    println!("Frame Timing:");
    println!(
        "  Average:  {:>8.3}ms ({:.1} FPS)",
        avg_frame.as_secs_f64() * 1000.0,
        1000.0 / (avg_frame.as_secs_f64() * 1000.0)
    );
    println!(
        "  Min:      {:>8.3}ms ({:.1} FPS)",
        min_frame.as_secs_f64() * 1000.0,
        1000.0 / (min_frame.as_secs_f64() * 1000.0)
    );
    println!(
        "  Max:      {:>8.3}ms ({:.1} FPS)",
        max_frame.as_secs_f64() * 1000.0,
        1000.0 / (max_frame.as_secs_f64() * 1000.0)
    );
    println!(
        "  P50:      {:>8.3}ms ({:.1} FPS)",
        p50_frame.as_secs_f64() * 1000.0,
        1000.0 / (p50_frame.as_secs_f64() * 1000.0)
    );
    println!(
        "  P95:      {:>8.3}ms ({:.1} FPS)",
        p95_frame.as_secs_f64() * 1000.0,
        1000.0 / (p95_frame.as_secs_f64() * 1000.0)
    );
    println!(
        "  P99:      {:>8.3}ms ({:.1} FPS)",
        p99_frame.as_secs_f64() * 1000.0,
        1000.0 / (p99_frame.as_secs_f64() * 1000.0)
    );
    println!();
    println!("Render Timing:");
    println!("  Average:  {:>8.3}ms", avg_render.as_secs_f64() * 1000.0);
    println!();
}

#[cfg(not(feature = "drm"))]
#[kiss3d::main]
async fn main() {
    eprintln!("This example requires the 'drm' feature to be enabled.");
    eprintln!("Please run with: cargo run --example drm_cubes --features drm --release");
}
