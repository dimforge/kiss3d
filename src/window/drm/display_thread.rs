//! Display thread: shows frames on the display without blocking the render thread.
//!
//! The render thread hands over RGB frames; this thread converts them to the
//! display's XRGB8888 layout, copies them into a dumb buffer and sets the CRTC.

use super::card::Card;
use super::error::DrmError;
use drm::buffer::Buffer;
use drm::control::dumbbuffer::DumbBuffer;
use drm::control::{connector, crtc, framebuffer, Device as ControlDevice, Mode};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// A frame to display.
pub struct DisplayCommand {
    /// Tightly packed RGB pixels (3 bytes each), rows from top to bottom.
    pub rgb: Vec<u8>,
    /// Width of the frame in pixels.
    pub width: u32,
    /// Height of the frame in pixels.
    pub height: u32,
}

/// Where and how the display thread shows frames.
pub struct DisplayThreadConfig {
    pub connector: connector::Handle,
    pub crtc: crtc::Handle,
    pub mode: Mode,
}

/// Owns the worker thread that presents frames.
pub struct DisplayThread {
    /// Bounded channel (capacity 1): the render thread can queue one frame
    /// while the worker presents another, and blocks beyond that.
    sender: Option<SyncSender<DisplayCommand>>,
    thread_handle: Option<JoinHandle<()>>,
}

impl DisplayThread {
    /// Starts the display thread. `width` and `height` are the display size in pixels.
    ///
    /// The scan-out buffers are created here, so that a device that cannot provide
    /// them is reported to the caller instead of failing on the worker thread.
    pub fn new(
        card: Card,
        config: DisplayThreadConfig,
        width: u32,
        height: u32,
    ) -> Result<Self, DrmError> {
        let front = ScanoutBuffer::create(&card, width, height)?;
        let back = match ScanoutBuffer::create(&card, width, height) {
            Ok(back) => back,
            Err(e) => {
                front.destroy(&card);
                return Err(e);
            }
        };

        // What the display showed before, to put back when the thread ends.
        let saved_crtc = match card.get_crtc(config.crtc) {
            Ok(info) => Some(info),
            Err(e) => {
                log::warn!("Could not save the current CRTC state: {}", e);
                None
            }
        };

        let (sender, receiver) = sync_channel(1);

        let thread_handle = thread::spawn(move || {
            Self::display_worker(card, config, receiver, front, back, width, saved_crtc);
        });

        log::info!("Display thread started with bounded channel (capacity 1)");

        Ok(DisplayThread {
            sender: Some(sender),
            thread_handle: Some(thread_handle),
        })
    }

    /// Queues a frame for display.
    ///
    /// Blocks while the worker is still presenting the previous frame and one more
    /// is already queued, which gives natural backpressure. Fails once the worker
    /// has stopped (for example after a panic).
    pub fn send_frame(&self, command: DisplayCommand) -> Result<(), DrmError> {
        match &self.sender {
            Some(sender) => sender
                .send(command)
                .map_err(|_| DrmError::DisplayThreadStopped),
            None => Err(DrmError::DisplayThreadStopped),
        }
    }

    fn display_worker(
        card: Card,
        config: DisplayThreadConfig,
        receiver: Receiver<DisplayCommand>,
        mut front: ScanoutBuffer,
        mut back: ScanoutBuffer,
        width: u32,
        saved_crtc: Option<crtc::Info>,
    ) {
        log::debug!("Display worker thread started");

        // Double-buffered: the two buffers are alternated every frame.
        let mut use_front = true;

        // One row of XRGB pixels, converted in cached memory before the copy: the
        // mapped dumb buffer is typically uncached memory, slow for scattered writes.
        let mut row = vec![0u8; width as usize * 4];

        // Time spent per activity, logged on shutdown: a worker that is mostly idle is
        // not what limits the frame rate.
        let mut frames = 0u32;
        let (mut idle, mut copy, mut set_crtc) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
        // `set_crtc` fails on every frame once it fails at all (e.g. after losing DRM
        // master), so only the first failure is logged at error level.
        let mut set_crtc_failed = false;

        loop {
            let wait_start = Instant::now();
            let Ok(command) = receiver.recv() else {
                break;
            };
            idle += wait_start.elapsed();

            let target = if use_front { &mut front } else { &mut back };

            let copy_start = Instant::now();
            if let Err(e) =
                Self::copy_to_dumb_buffer(&card, &mut target.dumb_buffer, &command, &mut row)
            {
                log::error!("Display worker: failed to copy to dumb buffer: {}", e);
                continue;
            }
            copy += copy_start.elapsed();

            let set_crtc_start = Instant::now();
            let shown = match card.set_crtc(
                config.crtc,
                Some(target.framebuffer),
                (0, 0),
                &[config.connector],
                Some(config.mode),
            ) {
                Ok(()) => true,
                Err(e) => {
                    // A permission error here usually means another process took over
                    // the display (DRM master was lost).
                    let hint = if e.kind() == std::io::ErrorKind::PermissionDenied {
                        " (DRM master may have been taken by another process)"
                    } else {
                        ""
                    };
                    if set_crtc_failed {
                        log::debug!("Display worker: set_crtc failed: {}{}", e, hint);
                    } else {
                        log::error!("Display worker: set_crtc failed: {}{}", e, hint);
                        set_crtc_failed = true;
                    }
                    false
                }
            };
            set_crtc += set_crtc_start.elapsed();

            // Only a buffer that went on screen changes which one is free to write
            // next: after a failure the other buffer is still the one being shown.
            if shown {
                frames += 1;
                use_front = !use_front;
            }
        }
        log::debug!("Display worker: channel closed, exiting");
        if frames > 0 {
            let avg_ms = |d: Duration| d.as_secs_f64() * 1000.0 / frames as f64;
            log::info!(
                "Display worker: {} frames, avg per frame: {:.2} ms idle, {:.2} ms copy, {:.2} ms set_crtc",
                frames,
                avg_ms(idle),
                avg_ms(copy),
                avg_ms(set_crtc)
            );
        }

        Self::restore_crtc(&card, &config, saved_crtc.as_ref());
        front.destroy(&card);
        back.destroy(&card);

        log::info!("Display worker thread terminated");
    }

    /// Puts the display back as it was before the thread started, or switches it off
    /// when it showed nothing. The scan-out buffers must not be destroyed while the
    /// display still points at them.
    fn restore_crtc(card: &Card, config: &DisplayThreadConfig, saved: Option<&crtc::Info>) {
        let result = match saved {
            Some(info) if info.mode().is_some() && info.framebuffer().is_some() => card.set_crtc(
                config.crtc,
                info.framebuffer(),
                info.position(),
                &[config.connector],
                info.mode(),
            ),
            _ => card.set_crtc(config.crtc, None, (0, 0), &[], None),
        };

        if let Err(e) = result {
            log::warn!("Failed to restore the display state: {}", e);
        }
    }

    /// Converts `command` to XRGB8888 and writes it into `dumb_buffer`.
    ///
    /// `row` is a scratch buffer of `width * 4` bytes.
    fn copy_to_dumb_buffer(
        card: &Card,
        dumb_buffer: &mut DumbBuffer,
        command: &DisplayCommand,
        row: &mut [u8],
    ) -> Result<(), String> {
        // The hardware row pitch may be larger than `width * 4`.
        let pitch = dumb_buffer.pitch() as usize;
        let mut mapping = card
            .map_dumb_buffer(dumb_buffer)
            .map_err(|e| format!("Failed to map dumb buffer: {}", e))?;

        write_frame(
            &command.rgb,
            command.width as usize,
            command.height as usize,
            mapping.as_mut(),
            pitch,
            row,
        )
    }
}

/// Converts tightly packed RGB rows to XRGB8888 and writes them into `dst`, whose rows
/// start `pitch` bytes apart. Padding bytes at the end of a destination row are left
/// untouched. `row` is a scratch buffer of `width * 4` bytes.
fn write_frame(
    rgb: &[u8],
    width: usize,
    height: usize,
    dst: &mut [u8],
    pitch: usize,
    row: &mut [u8],
) -> Result<(), String> {
    let src_stride = width * 3;
    let dst_stride = width * 4;

    if rgb.len() != src_stride * height {
        return Err(format!(
            "Pixel data size mismatch: expected {}, got {}",
            src_stride * height,
            rgb.len()
        ));
    }
    if row.len() != dst_stride {
        return Err(format!(
            "Frame width {} does not match the display width",
            width
        ));
    }
    if pitch < dst_stride {
        return Err(format!(
            "Row pitch {} is smaller than a row of {} bytes",
            pitch, dst_stride
        ));
    }
    if dst.len() < pitch * height {
        return Err(format!(
            "Dumb buffer too small: {} bytes for {} rows of pitch {}",
            dst.len(),
            height,
            pitch
        ));
    }

    for (y, src_row) in rgb.chunks_exact(src_stride).enumerate() {
        rgb_to_xrgb_row(src_row, row);
        dst[y * pitch..y * pitch + dst_stride].copy_from_slice(row);
    }

    Ok(())
}

/// A dumb buffer together with the framebuffer that scans it out.
struct ScanoutBuffer {
    dumb_buffer: DumbBuffer,
    framebuffer: framebuffer::Handle,
}

impl ScanoutBuffer {
    fn create(card: &Card, width: u32, height: u32) -> Result<Self, DrmError> {
        use drm::buffer::DrmFourcc;

        let dumb_buffer = card
            .create_dumb_buffer((width, height), DrmFourcc::Xrgb8888, 32)
            .map_err(|e| DrmError::Init(format!("Failed to create dumb buffer: {}", e)))?;

        match card.add_framebuffer(&dumb_buffer, 24, 32) {
            Ok(framebuffer) => Ok(ScanoutBuffer {
                dumb_buffer,
                framebuffer,
            }),
            Err(e) => {
                if let Err(e) = card.destroy_dumb_buffer(dumb_buffer) {
                    log::warn!("Failed to destroy dumb buffer: {}", e);
                }
                Err(DrmError::Init(format!(
                    "Failed to create framebuffer: {}",
                    e
                )))
            }
        }
    }

    fn destroy(self, card: &Card) {
        if let Err(e) = card.destroy_framebuffer(self.framebuffer) {
            log::warn!("Failed to destroy framebuffer: {}", e);
        }
        if let Err(e) = card.destroy_dumb_buffer(self.dumb_buffer) {
            log::warn!("Failed to destroy dumb buffer: {}", e);
        }
    }
}

/// Expands a row of RGB pixels into XRGB8888 (little-endian memory order B, G, R, X).
fn rgb_to_xrgb_row(rgb: &[u8], xrgb: &mut [u8]) {
    let (rgb, _) = rgb.as_chunks::<3>();
    let (xrgb, _) = xrgb.as_chunks_mut::<4>();
    for (src, dst) in rgb.iter().zip(xrgb.iter_mut()) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        dst[3] = 0xFF;
    }
}

impl Drop for DisplayThread {
    fn drop(&mut self) {
        // Dropping the sender closes the channel, which makes the worker leave its
        // receive loop; it must happen before `join`.
        self.sender.take();

        if let Some(handle) = self.thread_handle.take() {
            if let Err(e) = handle.join() {
                log::error!("Failed to join display thread: {:?}", e);
            }
        }

        log::info!("Display thread shut down");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_to_xrgb_row_reorders_channels() {
        let rgb = [1, 2, 3, 10, 20, 30];
        let mut xrgb = [0u8; 8];

        rgb_to_xrgb_row(&rgb, &mut xrgb);

        // B, G, R, X for each pixel.
        assert_eq!(xrgb, [3, 2, 1, 0xFF, 30, 20, 10, 0xFF]);
    }

    #[test]
    fn rgb_to_xrgb_row_handles_empty_row() {
        let mut xrgb: [u8; 0] = [];

        rgb_to_xrgb_row(&[], &mut xrgb);
    }

    // Two pixels per row, two rows: (1,2,3) (4,5,6) / (7,8,9) (10,11,12).
    const RGB_2X2: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    #[test]
    fn write_frame_packs_rows_when_pitch_equals_row_size() {
        let mut dst = [0u8; 16];
        let mut row = [0u8; 8];

        write_frame(&RGB_2X2, 2, 2, &mut dst, 8, &mut row).unwrap();

        assert_eq!(
            dst,
            [3, 2, 1, 0xFF, 6, 5, 4, 0xFF, 9, 8, 7, 0xFF, 12, 11, 10, 0xFF]
        );
    }

    #[test]
    fn write_frame_honors_a_larger_pitch_and_keeps_padding() {
        // Rows start 12 bytes apart; the last 4 bytes of each row are padding.
        let mut dst = [0xAAu8; 24];
        let mut row = [0u8; 8];

        write_frame(&RGB_2X2, 2, 2, &mut dst, 12, &mut row).unwrap();

        assert_eq!(&dst[0..8], [3, 2, 1, 0xFF, 6, 5, 4, 0xFF]);
        assert_eq!(&dst[8..12], [0xAA; 4]);
        assert_eq!(&dst[12..20], [9, 8, 7, 0xFF, 12, 11, 10, 0xFF]);
        assert_eq!(&dst[20..24], [0xAA; 4]);
    }

    #[test]
    fn write_frame_rejects_a_pixel_data_size_mismatch() {
        let mut dst = [0u8; 16];
        let mut row = [0u8; 8];

        let err = write_frame(&RGB_2X2[..11], 2, 2, &mut dst, 8, &mut row).unwrap_err();

        assert!(err.contains("size mismatch"), "{}", err);
        assert_eq!(dst, [0u8; 16], "nothing may be written on error");
    }

    #[test]
    fn write_frame_rejects_a_scratch_row_of_the_wrong_width() {
        let mut dst = [0u8; 16];
        let mut row = [0u8; 4];

        let err = write_frame(&RGB_2X2, 2, 2, &mut dst, 8, &mut row).unwrap_err();

        assert!(err.contains("width"), "{}", err);
    }

    #[test]
    fn write_frame_rejects_a_pitch_smaller_than_a_row() {
        let mut dst = [0u8; 16];
        let mut row = [0u8; 8];

        let err = write_frame(&RGB_2X2, 2, 2, &mut dst, 4, &mut row).unwrap_err();

        assert!(err.contains("pitch"), "{}", err);
    }

    #[test]
    fn write_frame_rejects_a_destination_that_is_too_small() {
        let mut dst = [0u8; 15];
        let mut row = [0u8; 8];

        let err = write_frame(&RGB_2X2, 2, 2, &mut dst, 8, &mut row).unwrap_err();

        assert!(err.contains("too small"), "{}", err);
    }
}
