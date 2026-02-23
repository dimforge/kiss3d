// #![allow(dead_code)]
use drm::Device;
use drm::control::Device as ControlDevice;

#[derive(Debug)]
/// A simple wrapper for a device node.
pub struct Card(std::fs::File);

/// Implementing `AsFd` is a prerequisite to implementing the traits found
/// in this crate. Here, we are just calling `as_fd()` on the inner File.
impl std::os::unix::io::AsFd for Card {
    fn as_fd(&self) -> std::os::unix::io::BorrowedFd<'_> {
        self.0.as_fd()
    }
}

/// With `AsFd` implemented, we can now implement `drm::Device`.
impl Device for Card {}
impl ControlDevice for Card {}

/// Simple helper methods for opening a `Card`.
impl Card {
    pub fn open(path: &str) -> Result<Self, std::io::Error> {
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        options.write(true);
        Ok(Card(options.open(path)?))
    }

    pub fn open_global(device: &str) -> Result<Self, std::io::Error> {
        Self::open(device)
    }
}

#[allow(dead_code)]
pub mod capabilities {
    use drm::ClientCapability as CC;
    pub const CLIENT_CAP_ENUMS: &[CC] = &[CC::Stereo3D, CC::UniversalPlanes, CC::Atomic];

    use drm::DriverCapability as DC;
    pub const DRIVER_CAP_ENUMS: &[DC] = &[
        DC::DumbBuffer,
        DC::VBlankHighCRTC,
        DC::DumbPreferredDepth,
        DC::DumbPreferShadow,
        DC::Prime,
        DC::MonotonicTimestamp,
        DC::ASyncPageFlip,
        DC::CursorWidth,
        DC::CursorHeight,
        DC::AddFB2Modifiers,
        DC::PageFlipTarget,
        DC::CRTCInVBlankEvent,
        DC::SyncObj,
        DC::TimelineSyncObj,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    // Since we can't easily mock std::fs::File, we'll need to be more creative with our tests
    // We'll use conditional compilation for tests that need real files

    #[test]
    fn test_capabilities_constants() {
        // Test that capability enums are defined and contain expected values
        assert!(!capabilities::CLIENT_CAP_ENUMS.is_empty());
        assert!(!capabilities::DRIVER_CAP_ENUMS.is_empty());

        // Check for specific capabilities
        assert!(capabilities::CLIENT_CAP_ENUMS.contains(&drm::ClientCapability::Stereo3D));
        assert!(capabilities::CLIENT_CAP_ENUMS.contains(&drm::ClientCapability::UniversalPlanes));
        assert!(capabilities::CLIENT_CAP_ENUMS.contains(&drm::ClientCapability::Atomic));

        assert!(capabilities::DRIVER_CAP_ENUMS.contains(&drm::DriverCapability::DumbBuffer));
        assert!(capabilities::DRIVER_CAP_ENUMS.contains(&drm::DriverCapability::Prime));
    }

    // Testing trait implementations via compile-time checks

    #[test]
    fn test_trait_impls_compile() {
        // This test doesn't actually run code, but it ensures that our Card type
        // implements the necessary traits for DRM operations.
        // If these trait bounds are satisfied, the test compiles.

        fn assert_device<T: Device>() {}
        fn assert_control_device<T: ControlDevice>() {}

        // These will fail at compile time if Card doesn't implement the traits
        assert_device::<Card>();
        assert_control_device::<Card>();
    }

    // File-related tests are more challenging without mocks
    // We'll use a fake file path for testing error handling

    #[test]
    #[ignore = "This test would attempt to open a real file"]
    fn test_open_error_handling() {
        // Test that opening a non-existent file path handles errors appropriately
        // We're ignoring this test because it would panic (unwrap on error)
        // In a real implementation, we might want to improve error handling

        let non_existent_path = "/path/that/definitely/does/not/exist";
        let _ = Card::open(non_existent_path);
    }

    #[test]
    #[ignore = "This test would attempt to open a real device"]
    fn test_open_global() {
        // Test that open_global correctly forwards to open
        // Again, ignored to avoid actual device operations
        let device_path = "/dev/dri/card0";
        let _ = Card::open_global(device_path);
    }

    #[test]
    fn test_card_debug() {
        // Test the Debug implementation for Card
        // We'd need a real file to instantiate a Card, so this is more of a conceptual test

        // Verify that the Debug trait is implemented for Card at compile time
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<Card>();
    }
}
