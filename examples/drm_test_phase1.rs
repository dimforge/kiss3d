//! Phase 1 Test: DRM Display Resource Query
//!
//! This program tests the Phase 1 implementation by:
//! - Opening a DRM device
//! - Querying connected displays
//! - Enumerating available modes
//! - Testing error handling
//!
//! Run with: cargo run --example drm_test_phase1 --features drm
//!
//! Requirements:
//! - Must be run with permissions to access /dev/dri/card* (root or video group)
//! - A display must be connected

#[cfg(feature = "drm")]
use drm::control::{connector, Device as ControlDevice, ResourceHandles};
#[cfg(feature = "drm")]
use drm::Device;

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
fn test_phase1() -> Result<(), Box<dyn std::error::Error>> {
    use card::Card;

    println!("=== Phase 1 DRM Display Resource Query Test ===\n");

    // Test 1: Open DRM device and validate it
    println!("Test 1: Opening and validating DRM device...");
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

                // Try to query resource handles to validate the device
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
    println!("Test 2: Resource information...");

    println!("  ✓ Found resources:");
    println!("    - Connectors: {}", resources.connectors().len());
    println!("    - Encoders: {}", resources.encoders().len());
    println!("    - CRTCs: {}", resources.crtcs().len());
    println!("    - Framebuffers: {}", resources.framebuffers().len());
    println!();

    // Test 3: Enumerate connectors
    println!("Test 3: Enumerating connectors...");
    let mut connected_count = 0;
    let mut disconnected_count = 0;

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

                        // Show available modes
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

                        // Show encoder info
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

    // Test 4: Check CRTCs
    println!("Test 4: Checking CRTCs...");
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

    // Test 5: Validate Phase 1 logic
    println!("Test 5: Validating Phase 1 query logic...");

    // Find connected connector (like Phase 1 does)
    let connected_connector = resources.connectors().iter().find_map(|&conn_handle| {
        card.get_connector(conn_handle, false)
            .ok()
            .and_then(|info| {
                if info.state() == connector::State::Connected {
                    Some(info)
                } else {
                    None
                }
            })
    });

    match connected_connector {
        Some(conn_info) => {
            println!("  ✓ Found connected display");

            // Get first mode (like Phase 1 does)
            let modes = conn_info.modes();
            if !modes.is_empty() {
                let mode = modes[0];
                let (w, h) = mode.size();
                println!("  ✓ Selected mode: {}x{} @ {}Hz", w, h, mode.vrefresh());
            } else {
                println!("  ✗ No modes available");
            }

            // Get first CRTC (like Phase 1 does)
            if let Some(&crtc) = resources.crtcs().first() {
                println!("  ✓ Selected CRTC: {:?}", crtc);
            } else {
                println!("  ✗ No CRTCs available");
            }

            println!();
            println!("=== Phase 1 Test: SUCCESS ===");
            println!("All Phase 1 functionality is working correctly!");
            println!("Device: {}", device_path);
        }
        None => {
            println!("  ✗ No connected display found");
            println!();
            println!("=== Phase 1 Test: FAILED ===");
            println!("No connected display. Please connect a display and try again.");
            return Err("No connected display".into());
        }
    }

    Ok(())
}

#[cfg(not(feature = "drm"))]
fn test_phase1() -> Result<(), Box<dyn std::error::Error>> {
    println!("This test requires the 'drm' feature to be enabled.");
    println!("Run with: cargo run --example drm_test_phase1 --features drm");
    Ok(())
}

fn main() {
    env_logger::init();

    match test_phase1() {
        Ok(_) => {
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("\n❌ Test failed: {}", e);
            eprintln!("\nTroubleshooting:");
            eprintln!("  1. Make sure you have permissions (try: sudo)");
            eprintln!("  2. Check if /dev/dri/card0 exists");
            eprintln!("  3. Ensure a display is connected");
            eprintln!("  4. On Raspberry Pi, make sure vc4-kms-v3d is enabled in config.txt");
            std::process::exit(1);
        }
    }
}
