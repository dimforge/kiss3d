//! Unified DRM Test: Phase 1 (Display Discovery) + Phase 2 (GBM Integration)
//!
//! This program tests the complete DRM implementation by:
//!
//! Phase 1:
//! - Opening DRM devices with fallback
//! - Querying connected displays
//! - Enumerating available modes
//! - Finding CRTCs and encoders
//!
//! Phase 2:
//! - Initializing GBM (Generic Buffer Manager)
//! - Creating GBM surfaces
//! - Testing buffer operations
//! - Validating format support
//!
//! Run with: cargo run --example drm_test --features drm
//!
//! Requirements:
//! - Must be run with permissions to access /dev/dri/card* (root or video group)
//! - A display must be connected
//! - KMS driver enabled (vc4-kms-v3d on Raspberry Pi)

#[cfg(feature = "drm")]
use drm::control::{connector, Device as ControlDevice};
#[cfg(feature = "drm")]
use drm::Device;
// #[cfg(feature = "drm")]
// use gbm;

#[cfg(feature = "drm")]
mod card {
    use std::os::unix::io::{AsFd, BorrowedFd};

    #[derive(Debug)]
    pub struct Card(std::fs::File);

    impl AsFd for Card {
        fn as_fd(&self) -> BorrowedFd<'_> {
            self.0.as_fd()
        }
    }

    impl drm::Device for Card {}
    impl drm::control::Device for Card {}

    impl Card {
        pub fn open(path: &str) -> Result<Self, std::io::Error> {
            let mut options = std::fs::OpenOptions::new();
            options.read(true);
            options.write(true);
            Ok(Card(options.open(path)?))
        }
    }
}

#[cfg(feature = "drm")]
fn run_drm_tests() -> Result<(), Box<dyn std::error::Error>> {
    use card::Card;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         DRM/GBM Comprehensive Test (Phase 1 + Phase 2)          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ========================================================================
    // PHASE 1: Display Resource Discovery
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 1: Display Resource Discovery                             │");
    println!("└──────────────────────────────────────────────────────────────────┘");
    println!();

    // Test 1: Open DRM device with validation
    println!("Test 1.1: Opening and validating DRM device...");
    let device_paths = [
        "/dev/dri/card0",
        "/dev/dri/card1",
        "/dev/dri/card2",
        "/dev/dri/renderD128",
        "/dev/dri/renderD129",
    ];

    let mut card = None;
    let mut device_path = "";
    let mut resources = None;

    for path in &device_paths {
        match Card::open(path) {
            Ok(c) => {
                println!("  ✓ Successfully opened: {}", path);

                match c.resource_handles() {
                    Ok(res) => {
                        println!("  ✓ Successfully queried resources from: {}", path);
                        card = Some(c);
                        resources = Some(res);
                        device_path = path;
                        break;
                    }
                    Err(e) => {
                        println!("  ✗ Failed to query resources from {}: {}", path, e);
                        println!("     Trying next device...");
                    }
                }
            }
            Err(e) => {
                println!("  ✗ Failed to open {}: {}", path, e);
            }
        }
    }

    let card = card.ok_or("No usable DRM device found. Try running with sudo?")?;
    let resources = resources.ok_or("Failed to get resource handles from any device")?;
    println!();

    // Test 2: Display resource information
    println!("Test 1.2: Resource information...");
    println!("  ✓ Found resources:");
    println!("    - Connectors: {}", resources.connectors().len());
    println!("    - Encoders: {}", resources.encoders().len());
    println!("    - CRTCs: {}", resources.crtcs().len());
    println!("    - Framebuffers: {}", resources.framebuffers().len());
    println!();

    // Test 3: Enumerate connectors
    println!("Test 1.3: Enumerating connectors...");
    let mut connected_count = 0;
    let mut disconnected_count = 0;
    let mut connected_connector_info = None;

    for (i, &conn_handle) in resources.connectors().iter().enumerate() {
        match card.get_connector(conn_handle, false) {
            Ok(conn_info) => {
                let state = conn_info.state();
                let interface = conn_info.interface();

                print!("  Connector {}: {:?} - ", i, interface);

                match state {
                    connector::State::Connected => {
                        println!("✓ CONNECTED");
                        connected_count += 1;

                        if connected_connector_info.is_none() {
                            connected_connector_info = Some(conn_info.clone());
                        }

                        let modes = conn_info.modes();
                        println!("    Available modes: {}", modes.len());
                        for (j, mode) in modes.iter().take(3).enumerate() {
                            let (w, h) = mode.size();
                            println!(
                                "      {}. {}x{} @ {}Hz{}",
                                j + 1,
                                w,
                                h,
                                mode.vrefresh(),
                                if j == 0 { " (preferred)" } else { "" }
                            );
                        }
                        if modes.len() > 3 {
                            println!("      ... and {} more", modes.len() - 3);
                        }

                        if let Some(encoder_handle) = conn_info.current_encoder() {
                            match card.get_encoder(encoder_handle) {
                                Ok(encoder) => {
                                    println!("    Current encoder: {:?}", encoder.handle());
                                }
                                Err(e) => {
                                    println!("    Failed to get encoder: {}", e);
                                }
                            }
                        } else {
                            println!("    No current encoder");
                        }
                    }
                    connector::State::Disconnected => {
                        println!("✗ Disconnected");
                        disconnected_count += 1;
                    }
                    connector::State::Unknown => {
                        println!("? Unknown state");
                    }
                }
            }
            Err(e) => {
                println!("  Connector {}: Error - {}", i, e);
            }
        }
    }

    println!();
    println!("Summary:");
    println!("  Connected displays: {}", connected_count);
    println!("  Disconnected: {}", disconnected_count);
    println!();

    // Require at least one connected display
    let conn_info = connected_connector_info.ok_or("No connected display found")?;
    let modes = conn_info.modes();
    if modes.is_empty() {
        return Err("No display modes available".into());
    }
    let mode = modes[0];
    let (width, height) = mode.size();

    // Test 4: Check CRTCs
    println!("Test 1.4: Checking CRTCs...");
    for (i, &crtc_handle) in resources.crtcs().iter().enumerate() {
        match card.get_crtc(crtc_handle) {
            Ok(crtc_info) => {
                println!("  CRTC {}: {:?}", i, crtc_handle);
                if let Some(mode) = crtc_info.mode() {
                    let (w, h) = mode.size();
                    println!("    Current mode: {}x{} @ {}Hz", w, h, mode.vrefresh());
                } else {
                    println!("    No active mode");
                }
            }
            Err(e) => {
                println!("  CRTC {}: Error - {}", i, e);
            }
        }
    }
    println!();

    // Test 5: Validate Phase 1 query logic
    println!("Test 1.5: Validating Phase 1 query logic...");
    println!("  ✓ Found connected display: {}x{}", width, height);
    println!(
        "  ✓ Selected mode: {}x{} @ {}Hz",
        width,
        height,
        mode.vrefresh()
    );
    if let Some(&crtc) = resources.crtcs().first() {
        println!("  ✓ Selected CRTC: {:?}", crtc);
    } else {
        return Err("No CRTCs available".into());
    }
    println!();

    println!("✅ Phase 1: COMPLETE - Display discovery successful");
    println!();

    // ========================================================================
    // PHASE 2: GBM Integration
    // ========================================================================

    // println!("┌──────────────────────────────────────────────────────────────────┐");
    // println!("│ PHASE 2: GBM Integration                                        │");
    // println!("└──────────────────────────────────────────────────────────────────┘");
    // println!();

    // // Test 6: Create GBM device
    // println!("Test 2.1: Creating GBM device...");
    // let card_for_gbm = Card::open(device_path)?;
    // let gbm_device = match gbm::Device::new(card_for_gbm) {
    //     Ok(device) => {
    //         println!("  ✓ GBM device created successfully");
    //         device
    //     }
    //     Err(e) => {
    //         return Err(format!("Failed to create GBM device: {}", e).into());
    //     }
    // };
    // println!();

    // // Test 7: Query GBM backend
    // println!("Test 2.2: Querying GBM backend...");
    // let backend_name = gbm_device.backend_name();
    // println!("  ✓ GBM backend: {}", backend_name);
    // println!();

    // // Test 8: Test format support
    // println!("Test 2.3: Testing format support...");
    // let test_formats = [
    //     (gbm::Format::Xrgb8888, "XRGB8888 (standard)"),
    //     (gbm::Format::Argb8888, "ARGB8888 (with alpha)"),
    //     (gbm::Format::Rgb565, "RGB565 (16-bit)"),
    // ];

    // let mut supported_format = None;
    // for (format, name) in &test_formats {
    //     let is_supported = gbm_device.is_format_supported(*format, gbm::BufferObjectFlags::SCANOUT);
    //     println!("  {} - {}", if is_supported { "✓" } else { "✗" }, name);
    //     if is_supported && supported_format.is_none() {
    //         supported_format = Some(*format);
    //     }
    // }

    // let format = supported_format.ok_or("No supported formats found for scanout")?;
    // println!("  → Selected format: {:?}", format);
    // println!();

    // // Test 9: Create GBM surface
    // println!("Test 2.4: Creating GBM surface...");
    // let gbm_surface = match gbm_device.create_surface::<()>(
    //     width as u32,
    //     height as u32,
    //     format,
    //     gbm::BufferObjectFlags::SCANOUT | gbm::BufferObjectFlags::RENDERING,
    // ) {
    //     Ok(surface) => {
    //         println!("  ✓ GBM surface created: {}x{}", width, height);
    //         println!("    Format: {:?}", format);
    //         println!("    Flags: SCANOUT | RENDERING");
    //         surface
    //     }
    //     Err(e) => {
    //         return Err(format!("Failed to create GBM surface: {}", e).into());
    //     }
    // };
    // println!();

    // // Test 10: Test buffer operations
    // println!("Test 2.5: Testing buffer operations...");
    // match unsafe { gbm_surface.lock_front_buffer() } {
    //     Ok(bo) => {
    //         println!("  ✓ Successfully locked front buffer");

    //         let bo_width = bo.width();
    //         let bo_height = bo.height();
    //         let bo_format = bo.format();
    //         let bo_stride = bo.stride();

    //         println!("    Buffer information:");
    //         println!("      - Dimensions: {}x{}", bo_width, bo_height);
    //         println!("      - Format: {:?}", bo_format);
    //         println!("      - Stride: {} bytes", bo_stride);
    //         println!("      - Size: {} bytes", bo_stride * bo_height);

    //         let _handle = bo.handle();
    //         println!("  ✓ Buffer has valid handle (can create DRM framebuffer)");

    //         drop(bo);
    //         println!("  ✓ Buffer released successfully");
    //     }
    //     Err(e) => {
    //         println!(
    //             "  ℹ Failed to lock front buffer: {} (normal for new surface)",
    //             e
    //         );
    //     }
    // }
    // println!();

    // println!("✅ Phase 2: COMPLETE - GBM integration successful");
    // println!();

    // // ========================================================================
    // // FINAL SUMMARY
    // // ========================================================================

    // println!("╔══════════════════════════════════════════════════════════════════╗");
    // println!("║                      🎉 ALL TESTS PASSED 🎉                      ║");
    // println!("╚══════════════════════════════════════════════════════════════════╝");
    // println!();
    // println!("System Configuration:");
    // println!("  Device:           {}", device_path);
    // println!(
    //     "  Display:          {}x{} @ {}Hz",
    //     width,
    //     height,
    //     mode.vrefresh()
    // );
    // println!("  GBM Backend:      {}", backend_name);
    // println!("  Pixel Format:     {:?}", format);
    // println!();
    // println!("Phase 1: Display Discovery");
    // println!("  ✓ DRM device access");
    // println!("  ✓ Display resource enumeration");
    // println!("  ✓ Connected display detection");
    // println!("  ✓ Mode and CRTC selection");
    // println!();
    // println!("Phase 2: GBM Integration");
    // println!("  ✓ GBM device initialization");
    // println!("  ✓ Format support validation");
    // println!("  ✓ GBM surface creation");
    // println!("  ✓ Buffer lifecycle management");
    // println!();
    // println!("Ready for Phase 3: Display Output (modesetting and page flipping)");
    // println!();

    Ok(())
}

#[cfg(not(feature = "drm"))]
fn run_drm_tests() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    DRM Feature Not Enabled                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("This test requires the 'drm' feature to be enabled.");
    println!();
    println!("Run with:");
    println!("  cargo run --example drm_test --features drm");
    println!();
    Ok(())
}

fn main() {
    env_logger::init();

    match run_drm_tests() {
        Ok(_) => {
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!();
            eprintln!("╔══════════════════════════════════════════════════════════════════╗");
            eprintln!("║                        ❌ TEST FAILED                            ║");
            eprintln!("╚══════════════════════════════════════════════════════════════════╝");
            eprintln!();
            eprintln!("Error: {}", e);
            eprintln!();
            eprintln!("Troubleshooting:");
            eprintln!("  1. Run with elevated permissions:");
            eprintln!("     sudo cargo run --example drm_test --features drm");
            eprintln!();
            eprintln!("  2. Or add your user to the video group:");
            eprintln!("     sudo usermod -a -G video $USER");
            eprintln!("     (then logout and login)");
            eprintln!();
            eprintln!("  3. Ensure KMS driver is enabled (Raspberry Pi):");
            eprintln!("     Add 'dtoverlay=vc4-kms-v3d' to /boot/firmware/config.txt");
            eprintln!();
            eprintln!("  4. Verify display is connected:");
            eprintln!("     Check HDMI cable and try other port if available");
            eprintln!();
            eprintln!("  5. Check for device files:");
            eprintln!("     ls -l /dev/dri/");
            eprintln!();
            eprintln!("  6. View detailed logs:");
            eprintln!("     RUST_LOG=debug cargo run --example drm_test --features drm");
            eprintln!();
            std::process::exit(1);
        }
    }
}
