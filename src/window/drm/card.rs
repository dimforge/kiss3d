use drm::control::Device as ControlDevice;
use drm::Device;

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_implements_the_drm_device_traits() {
        // Fails to compile if `Card` lacks the traits the display path relies on.
        fn assert_device<T: Device>() {}
        fn assert_control_device<T: ControlDevice>() {}

        assert_device::<Card>();
        assert_control_device::<Card>();
    }

    #[test]
    fn open_reports_a_missing_device_as_an_error() {
        let result = Card::open("/path/that/definitely/does/not/exist");

        assert!(result.is_err());
    }
}
