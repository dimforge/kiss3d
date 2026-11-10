//! KMS discovery: find a connected display, a CRTC that can drive it, and a mode.

use super::card::Card;
use super::error::DrmError;
use drm::control::{
    connector, crtc, Device as ControlDevice, Mode, ModeTypeFlags, ResourceHandles,
};
use drm::Device;

/// Display configuration discovered from the hardware.
pub struct DisplayConfig {
    pub connector: connector::Handle,
    pub crtc: crtc::Handle,
    pub mode: Mode,
    pub width: u32,
    pub height: u32,
}

/// Becomes DRM master, which setting the display mode requires.
///
/// The first process to open a device on a console normally is master already, and
/// this succeeds at once. It fails while another process (such as a compositor, or
/// another console session) holds the display.
pub fn acquire_master(card: &Card) -> Result<(), DrmError> {
    card.acquire_master_lock().map_err(|e| {
        DrmError::Init(format!(
            "Could not become DRM master ({}): another process, such as a compositor or \
             another console session, may be using the display",
            e
        ))
    })
}

/// Checks that the device supports the capabilities the display path uses.
///
/// Fails early with a clear message instead of a cryptic error later.
pub fn validate_device_capabilities(card: &Card) -> Result<(), DrmError> {
    use drm::DriverCapability;

    log::debug!("Validating DRM device capabilities");

    match card.get_driver_capability(DriverCapability::DumbBuffer) {
        Ok(value) if value > 0 => {
            log::debug!("DumbBuffer capability: supported");
        }
        Ok(_) => {
            return Err(DrmError::Init(
                "Device does not support dumb buffers (required for display output)".to_string(),
            ));
        }
        Err(e) => {
            // Some drivers do not report capabilities properly: do not fail.
            log::warn!(
                "Could not query DumbBuffer capability: {}. Proceeding anyway.",
                e
            );
        }
    }

    Ok(())
}

/// Queries the display resources and picks a connector, CRTC and mode.
pub fn query_display_resources(card: &Card) -> Result<DisplayConfig, DrmError> {
    log::info!("Querying DRM display resources");

    let resources = card
        .resource_handles()
        .map_err(|e| DrmError::Drm(format!("Failed to get resource handles: {}", e)))?;

    let connector_info = find_connected_connector(card, &resources)?;
    log::info!(
        "Found connected connector: {:?} (id: {:?})",
        connector_info.interface(),
        connector_info.handle()
    );

    let crtc = find_available_crtc(card, &connector_info, &resources)?;
    log::info!("Selected CRTC: {:?}", crtc);

    let mode = select_best_mode(&connector_info)?;
    let (width, height) = mode.size();
    log::info!(
        "Selected mode: {}x{} @ {}Hz",
        width,
        height,
        mode.vrefresh()
    );

    Ok(DisplayConfig {
        connector: connector_info.handle(),
        crtc,
        mode,
        width: width as u32,
        height: height as u32,
    })
}

/// Finds the first connected connector.
fn find_connected_connector(
    card: &Card,
    resources: &ResourceHandles,
) -> Result<connector::Info, DrmError> {
    for &conn_handle in resources.connectors() {
        let conn_info = card
            .get_connector(conn_handle, false)
            .map_err(|e| DrmError::Drm(format!("Failed to get connector info: {}", e)))?;

        if conn_info.state() == connector::State::Connected {
            return Ok(conn_info);
        }
    }

    Err(DrmError::NoConnectedDisplay)
}

/// Finds a CRTC to drive the given connector.
///
/// Prefers the CRTC that already drives it, so that nothing else has to move. Otherwise
/// it takes one of the CRTCs the connector's encoders can use (a bitmask of
/// `possible_crtcs`, which matters on embedded systems such as the Raspberry Pi where the
/// routing is fixed), avoiding CRTCs that drive another connector.
fn find_available_crtc(
    card: &Card,
    connector_info: &connector::Info,
    resources: &ResourceHandles,
) -> Result<crtc::Handle, DrmError> {
    log::debug!(
        "Finding compatible CRTC for connector {:?}",
        connector_info.handle()
    );

    if resources.crtcs().is_empty() {
        return Err(DrmError::Modeset(
            "No CRTCs available in system".to_string(),
        ));
    }

    if let Some(crtc_handle) = current_crtc(card, connector_info) {
        log::info!(
            "Using CRTC {:?} that already drives connector {:?}",
            crtc_handle,
            connector_info.handle()
        );
        return Ok(crtc_handle);
    }

    let mut candidates = Vec::new();
    for &encoder_handle in connector_info.encoders() {
        match card.get_encoder(encoder_handle) {
            Ok(encoder) => {
                for crtc_handle in resources.filter_crtcs(encoder.possible_crtcs()) {
                    if !candidates.contains(&crtc_handle) {
                        candidates.push(crtc_handle);
                    }
                }
            }
            Err(e) => log::warn!("Failed to get encoder {:?}: {}", encoder_handle, e),
        }
    }

    let in_use = crtcs_in_use(card, connector_info.handle(), resources);
    let crtc_handle = pick_crtc(&candidates, &in_use).ok_or_else(|| {
        DrmError::Modeset(format!(
            "No compatible CRTC found for connector {:?}. Connector has {} encoders.",
            connector_info.handle(),
            connector_info.encoders().len()
        ))
    })?;

    log::info!(
        "Selected CRTC {:?} for connector {:?}",
        crtc_handle,
        connector_info.handle()
    );
    Ok(crtc_handle)
}

/// The CRTC currently driving `connector_info`, if any.
fn current_crtc(card: &Card, connector_info: &connector::Info) -> Option<crtc::Handle> {
    let encoder = card.get_encoder(connector_info.current_encoder()?).ok()?;
    encoder.crtc()
}

/// The CRTCs that currently drive a connector other than `connector`.
fn crtcs_in_use(
    card: &Card,
    connector: connector::Handle,
    resources: &ResourceHandles,
) -> Vec<crtc::Handle> {
    resources
        .connectors()
        .iter()
        .filter(|&&handle| handle != connector)
        .filter_map(|&handle| {
            let info = card.get_connector(handle, false).ok()?;
            current_crtc(card, &info)
        })
        .collect()
}

/// Picks the first candidate that no other connector uses, or the first candidate when
/// all of them are in use.
fn pick_crtc(candidates: &[crtc::Handle], in_use: &[crtc::Handle]) -> Option<crtc::Handle> {
    candidates
        .iter()
        .copied()
        .find(|candidate| !in_use.contains(candidate))
        .or_else(|| {
            let first = candidates.first().copied();
            if first.is_some() {
                log::warn!("Every compatible CRTC already drives another connector");
            }
            first
        })
}

/// Selects the display mode: the one marked as preferred (the display's native mode), or
/// the first one when none is.
fn select_best_mode(connector_info: &connector::Info) -> Result<Mode, DrmError> {
    pick_preferred(connector_info.modes(), |mode| {
        mode.mode_type().contains(ModeTypeFlags::PREFERRED)
    })
    .ok_or_else(|| DrmError::Modeset("No display modes available".to_string()))
}

/// The first item for which `is_preferred` holds, else the first item.
fn pick_preferred<T: Copy>(items: &[T], is_preferred: impl Fn(&T) -> bool) -> Option<T> {
    items
        .iter()
        .find(|item| is_preferred(item))
        .or_else(|| items.first())
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;

    fn crtc(id: u32) -> crtc::Handle {
        crtc::Handle::from(NonZeroU32::new(id).unwrap())
    }

    #[test]
    fn pick_crtc_skips_crtcs_used_by_other_connectors() {
        let picked = pick_crtc(&[crtc(1), crtc(2), crtc(3)], &[crtc(1), crtc(2)]);

        assert_eq!(picked, Some(crtc(3)));
    }

    #[test]
    fn pick_crtc_takes_the_first_free_candidate() {
        let picked = pick_crtc(&[crtc(4), crtc(5)], &[]);

        assert_eq!(picked, Some(crtc(4)));
    }

    #[test]
    fn pick_crtc_falls_back_to_the_first_candidate_when_all_are_in_use() {
        let picked = pick_crtc(&[crtc(1), crtc(2)], &[crtc(1), crtc(2)]);

        assert_eq!(picked, Some(crtc(1)));
    }

    #[test]
    fn pick_crtc_without_candidates_finds_nothing() {
        assert_eq!(pick_crtc(&[], &[crtc(1)]), None);
    }

    #[test]
    fn pick_preferred_takes_the_preferred_item() {
        // Items are (size, preferred).
        let modes = [(800, false), (1024, true), (1280, false)];

        assert_eq!(pick_preferred(&modes, |m| m.1), Some((1024, true)));
    }

    #[test]
    fn pick_preferred_falls_back_to_the_first_item() {
        let modes = [(800, false), (1024, false)];

        assert_eq!(pick_preferred(&modes, |m| m.1), Some((800, false)));
    }

    #[test]
    fn pick_preferred_without_items_finds_nothing() {
        assert_eq!(pick_preferred::<(u32, bool)>(&[], |m| m.1), None);
    }
}
