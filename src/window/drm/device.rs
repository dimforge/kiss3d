//! DRM device selection: find the first `/dev/dri/card*` node that can drive a display.
//!
//! A machine can expose several DRM devices, and only some of them can show a picture.
//! On a Raspberry Pi 4, for example, the display controller (vc4) and the 3D unit (v3d)
//! are separate nodes, and which one is `card0` depends on the kernel and device tree.
//! So the first node is not always the right one, and each candidate is checked.

use super::card::Card;
use super::error::DrmError;
use super::kms::{self, DisplayConfig};
use std::path::{Path, PathBuf};

const DRI_DIR: &str = "/dev/dri";

/// Opens `path` and runs the display checks in order: DRM master, capabilities, and
/// a connected display with a usable CRTC and mode.
///
/// The `Card` is dropped on failure, which closes the node and releases master.
pub fn open_device(path: &str) -> Result<(Card, DisplayConfig), DrmError> {
    let card = Card::open(path)?;
    kms::acquire_master(&card)?;
    kms::validate_device_capabilities(&card)?;
    let config = kms::query_display_resources(&card)?;
    Ok((card, config))
}

/// Tries the `card*` nodes of `/dev/dri` in order and returns the first usable one.
///
/// `renderD*` nodes are skipped: they are render-only and cannot set a display mode.
/// If no node works, the error lists why each candidate was rejected.
pub fn open_first_usable() -> Result<(Card, DisplayConfig), DrmError> {
    let candidates = list_card_paths(Path::new(DRI_DIR))?;
    select_first(candidates, open_device)
}

/// Core of the selection, with the open step injected so it can be tested.
fn select_first<T>(
    candidates: Vec<PathBuf>,
    mut open: impl FnMut(&str) -> Result<T, DrmError>,
) -> Result<T, DrmError> {
    let mut failures = Vec::new();
    for path in candidates {
        let path = path.to_string_lossy().into_owned();
        match open(&path) {
            Ok(opened) => {
                log::info!("Using DRM device {}", path);
                return Ok(opened);
            }
            Err(e) => {
                log::debug!("Skipping DRM device {}: {}", path, e);
                failures.push(format!("{}: {}", path, e));
            }
        }
    }
    Err(DrmError::NoUsableDevice(failures))
}

/// Lists `card<N>` nodes in `dir`, ordered by `N`.
fn list_card_paths(dir: &Path) -> Result<Vec<PathBuf>, DrmError> {
    let mut numbered: Vec<(u32, PathBuf)> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            card_index(&name).map(|n| (n, entry.path()))
        })
        .collect();
    numbered.sort_by_key(|(n, _)| *n);
    Ok(numbered.into_iter().map(|(_, p)| p).collect())
}

/// Returns `N` for a node named `card<N>`, and `None` for anything else
/// (`renderD128`, `card0-HDMI-A-1`, `version`, ...).
fn card_index(name: &str) -> Option<u32> {
    let digits = name.strip_prefix("card")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_index_accepts_card_nodes_only() {
        assert_eq!(card_index("card0"), Some(0));
        assert_eq!(card_index("card12"), Some(12));
        assert_eq!(card_index("renderD128"), None);
        assert_eq!(card_index("card"), None);
        assert_eq!(card_index("card0-HDMI-A-1"), None);
        assert_eq!(card_index("by-path"), None);
    }

    #[test]
    fn list_card_paths_orders_numerically_and_skips_other_nodes() {
        let dir = std::env::temp_dir().join(format!("kiss3d_drm_list_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["card10", "card2", "card0", "renderD128", "card1-DSI-1"] {
            std::fs::File::create(dir.join(name)).unwrap();
        }

        let names: Vec<String> = list_card_paths(&dir)
            .unwrap()
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(names, ["card0", "card2", "card10"]);
    }

    #[test]
    fn list_card_paths_reports_a_missing_directory() {
        assert!(list_card_paths(Path::new("/path/that/does/not/exist")).is_err());
    }

    fn paths(names: &[&str]) -> Vec<PathBuf> {
        names.iter().map(PathBuf::from).collect()
    }

    #[test]
    fn select_first_skips_failing_devices_and_stops_at_the_first_success() {
        let mut tried = Vec::new();
        let selected = select_first(paths(&["/a", "/b", "/c"]), |p| {
            tried.push(p.to_string());
            if p == "/b" {
                Ok(42)
            } else {
                Err(DrmError::NoConnectedDisplay)
            }
        })
        .unwrap();

        assert_eq!(selected, 42);
        assert_eq!(tried, ["/a", "/b"]);
    }

    #[test]
    fn select_first_lists_every_failure_when_none_is_usable() {
        let err = select_first::<()>(paths(&["/a", "/b"]), |_| Err(DrmError::NoConnectedDisplay))
            .err()
            .unwrap();

        match err {
            DrmError::NoUsableDevice(failures) => {
                assert_eq!(failures.len(), 2);
                assert!(failures[0].starts_with("/a: "));
                assert!(failures[1].starts_with("/b: "));
            }
            other => panic!("unexpected error: {}", other),
        }
    }

    #[test]
    fn select_first_without_candidates_is_an_error() {
        let err = select_first::<()>(Vec::new(), |_| Ok(())).err().unwrap();

        assert!(matches!(err, DrmError::NoUsableDevice(f) if f.is_empty()));
    }
}
