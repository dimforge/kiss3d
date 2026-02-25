//! Async display thread for non-blocking DRM operations.
//!
//! This module implements an asynchronous display thread that handles the blocking
//! operations (buffer copy + set_crtc) in a separate thread, allowing the main
//! rendering thread to continue without waiting for display operations to complete.
//!
//! Based on the drm-gfx approach with double buffering and channel-based communication.

use super::card::Card;
use drm::control::dumbbuffer::DumbBuffer;
use drm::control::{connector, crtc, framebuffer, Device as ControlDevice, Mode};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

/// Command sent to the display thread
pub struct DisplayCommand {
    /// Pixel data to be copied to DRM buffer and displayed
    pub pixel_data: Vec<u8>,
    /// Width of the frame in pixels
    pub width: u32,
    /// Height of the frame in pixels
    pub height: u32,
}

/// Configuration for the display thread
pub struct DisplayThreadConfig {
    pub connector: connector::Handle,
    pub crtc: crtc::Handle,
    pub mode: Mode,
}

/// Buffer pool for recycling pixel buffers between threads
pub struct BufferPool {
    /// Channel for receiving recycled buffers
    available: Receiver<Vec<u8>>,
    /// Channel for sending buffers back to the pool
    recycle: Sender<Vec<u8>>,
}

impl BufferPool {
    /// Create a new buffer pool with pre-allocated buffers
    ///
    /// # Arguments
    /// * `num_buffers` - Number of buffers to pre-allocate (typically 2-3 for double/triple buffering)
    /// * `buffer_size` - Size of each buffer in bytes (width * height * bytes_per_pixel)
    pub fn new(num_buffers: usize, buffer_size: usize) -> Self {
        let (recycle_tx, available_rx) = channel();

        // Pre-allocate buffers
        for i in 0..num_buffers {
            let buffer = vec![0u8; buffer_size];
            recycle_tx
                .send(buffer)
                .expect("Failed to initialize buffer pool");
            log::debug!(
                "Buffer pool: pre-allocated buffer {}/{}",
                i + 1,
                num_buffers
            );
        }

        log::info!(
            "Buffer pool created with {} buffers of {} bytes each",
            num_buffers,
            buffer_size
        );

        BufferPool {
            available: available_rx,
            recycle: recycle_tx,
        }
    }

    /// Try to get an available buffer without blocking
    pub(crate) fn try_get_buffer(&self) -> Option<Vec<u8>> {
        self.available.try_recv().ok()
    }

    /// Get a sender for recycling buffers (for use in other threads)
    pub fn recycler(&self) -> Sender<Vec<u8>> {
        self.recycle.clone()
    }
}

/// Manages the async display thread for non-blocking presentation
pub struct DisplayThread {
    /// Sender for sending display commands to the worker thread
    sender: Option<Sender<DisplayCommand>>,
    /// Handle to the worker thread (for cleanup)
    thread_handle: Option<JoinHandle<()>>,
    /// Synchronization mutex to ensure display operations complete before buffer reuse
    sync_mutex: Arc<Mutex<()>>,
}

impl DisplayThread {
    /// Create and start a new display thread
    ///
    /// # Arguments
    /// * `card` - DRM card device (moved to display thread)
    /// * `config` - Display configuration
    /// * `buffer_recycler` - Sender for returning buffers to the pool
    /// * `width` - Display width in pixels
    /// * `height` - Display height in pixels
    pub fn new(
        card: Card,
        config: DisplayThreadConfig,
        buffer_recycler: Sender<Vec<u8>>,
        width: u32,
        height: u32,
    ) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        let sync_mutex = Arc::new(Mutex::new(()));
        let sync_mutex_clone = Arc::clone(&sync_mutex);

        // Spawn the display worker thread
        // Note: buffer_recycler is already a clone from buffer_pool.recycler()
        let thread_handle = thread::spawn(move || {
            Self::display_worker(
                card,
                config,
                receiver,
                sync_mutex_clone,
                buffer_recycler,
                width,
                height,
            );
        });

        log::info!("Display thread started");

        DisplayThread {
            sender: Some(sender),
            thread_handle: Some(thread_handle),
            sync_mutex,
        }
    }

    /// Send a frame to be displayed
    ///
    /// This is non-blocking - the frame is queued for display and the function
    /// returns immediately, allowing the main thread to continue rendering.
    pub fn send_frame(&self, command: DisplayCommand) -> Result<(), String> {
        // Try to acquire the lock to ensure previous frame completed
        // This prevents sending frames faster than the display can handle
        let _guard = self.sync_mutex.lock().unwrap();

        if let Some(sender) = &self.sender {
            sender
                .send(command)
                .map_err(|e| format!("Failed to send display command: {}", e))?;
            Ok(())
        } else {
            Err("Display thread sender has been closed".to_string())
        }
    }

    /// Worker thread function that handles display operations
    fn display_worker(
        card: Card,
        config: DisplayThreadConfig,
        receiver: Receiver<DisplayCommand>,
        sync_mutex: Arc<Mutex<()>>,
        buffer_recycler: Sender<Vec<u8>>,
        width: u32,
        height: u32,
    ) {
        log::debug!("Display worker thread started");

        // Create double buffers for DRM
        let mut dumb_buffer_front = Self::create_dumb_buffer(&card, width, height);
        let mut dumb_buffer_back = Self::create_dumb_buffer(&card, width, height);

        // Create framebuffers for the dumb buffers
        let fb_front = Self::create_framebuffer(&card, &dumb_buffer_front);
        let fb_back = Self::create_framebuffer(&card, &dumb_buffer_back);

        let mut use_front = true;

        loop {
            match receiver.recv() {
                Ok(command) => {
                    // Acquire lock to signal we're processing
                    let _guard = sync_mutex.lock().unwrap();

                    log::trace!(
                        "Display worker: received frame {}x{}",
                        command.width,
                        command.height
                    );

                    // Select which buffer to use
                    let (dumb_buffer, fb) = if use_front {
                        (&mut dumb_buffer_front, fb_front)
                    } else {
                        (&mut dumb_buffer_back, fb_back)
                    };

                    // Copy pixel data to DRM buffer
                    if let Err(e) = Self::copy_to_dumb_buffer(
                        &card,
                        dumb_buffer,
                        &command.pixel_data,
                        command.width,
                        command.height,
                    ) {
                        log::error!("Display worker: failed to copy to dumb buffer: {}", e);
                        // Recycle buffer and continue
                        buffer_recycler.send(command.pixel_data).ok();
                        continue;
                    }

                    // Perform the blocking set_crtc operation
                    if let Err(e) = card.set_crtc(
                        config.crtc,
                        Some(fb),
                        (0, 0),
                        &[config.connector],
                        Some(config.mode),
                    ) {
                        log::error!("Display worker: set_crtc failed: {}", e);
                        log::error!("Lost DRM master (another process took control of display)");
                    } else {
                        log::trace!("Display worker: frame presented successfully");
                    }

                    // Toggle buffers for next frame
                    use_front = !use_front;

                    // Recycle the pixel buffer
                    buffer_recycler.send(command.pixel_data).ok();

                    // Lock is automatically released here, signaling completion
                }
                Err(_) => {
                    log::debug!("Display worker: channel closed, exiting");
                    break;
                }
            }
        }

        // Cleanup
        if let Err(e) = card.destroy_framebuffer(fb_front) {
            log::warn!("Failed to destroy front framebuffer: {}", e);
        }
        if let Err(e) = card.destroy_framebuffer(fb_back) {
            log::warn!("Failed to destroy back framebuffer: {}", e);
        }
        if let Err(e) = card.destroy_dumb_buffer(dumb_buffer_front) {
            log::warn!("Failed to destroy front dumb buffer: {}", e);
        }
        if let Err(e) = card.destroy_dumb_buffer(dumb_buffer_back) {
            log::warn!("Failed to destroy back dumb buffer: {}", e);
        }

        log::info!("Display worker thread terminated");
    }

    /// Create a dumb buffer for DRM
    fn create_dumb_buffer(card: &Card, width: u32, height: u32) -> DumbBuffer {
        use drm::buffer::DrmFourcc;

        let format = DrmFourcc::Xrgb8888;
        card.create_dumb_buffer((width, height), format, 32)
            .expect("Failed to create dumb buffer")
    }

    /// Create a framebuffer for a dumb buffer
    fn create_framebuffer(card: &Card, buffer: &DumbBuffer) -> framebuffer::Handle {
        card.add_framebuffer(buffer, 24, 32)
            .expect("Failed to create framebuffer")
    }

    /// Copy pixel data to a DRM dumb buffer
    fn copy_to_dumb_buffer(
        card: &Card,
        dumb_buffer: &mut DumbBuffer,
        pixel_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        // Map the dumb buffer for writing
        let mut mapping = card
            .map_dumb_buffer(dumb_buffer)
            .map_err(|e| format!("Failed to map dumb buffer: {}", e))?;

        let buffer = mapping.as_mut();
        let stride = (width * 4) as usize; // 4 bytes per pixel
        let expected_size = stride * height as usize;

        if pixel_data.len() != expected_size {
            return Err(format!(
                "Pixel data size mismatch: expected {}, got {}",
                expected_size,
                pixel_data.len()
            ));
        }

        // Copy all pixels at once
        buffer[..expected_size].copy_from_slice(&pixel_data[..expected_size]);

        Ok(())
    }

    /// Wait for all pending display operations to complete
    ///
    /// Note: This is generally not needed as the Drop implementation
    /// handles proper shutdown by closing the channel and joining the thread
    #[allow(dead_code)]
    pub fn sync(&self) {
        // Simply acquiring and releasing the lock ensures the worker
        // has completed its current operation
        let _guard = self.sync_mutex.lock().unwrap();
    }
}

impl Drop for DisplayThread {
    fn drop(&mut self) {
        log::debug!("Shutting down display thread");

        // Drop the sender to close the channel, which signals the worker thread to exit
        // This must happen before join() so the worker thread can exit its recv() loop
        self.sender.take();
        log::debug!("Display thread channel closed, worker should exit");

        // Wait for the thread to finish gracefully
        // The join() call will block until the thread exits after processing
        // any remaining frames and completing cleanup
        if let Some(handle) = self.thread_handle.take() {
            log::debug!("Waiting for display worker thread to exit...");
            if let Err(e) = handle.join() {
                log::error!("Failed to join display thread: {:?}", e);
            } else {
                log::debug!("Display worker thread exited cleanly");
            }
        }

        log::info!("Display thread shut down");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_command_creation() {
        // Verify DisplayCommand can be created
        let pixel_data = vec![0u8; 1920 * 1080 * 4];
        let cmd = DisplayCommand {
            pixel_data,
            width: 1920,
            height: 1080,
        };

        assert_eq!(cmd.width, 1920);
        assert_eq!(cmd.height, 1080);
        assert_eq!(cmd.pixel_data.len(), 1920 * 1080 * 4);
    }

    #[test]
    fn test_buffer_pool_creation() {
        let pool = BufferPool::new(3, 1024);

        // Should be able to get 3 buffers
        let buf1 = pool.try_get_buffer();
        let buf2 = pool.try_get_buffer();
        let buf3 = pool.try_get_buffer();

        assert!(buf1.is_some());
        assert!(buf2.is_some());
        assert!(buf3.is_some());

        // Should be empty now
        let buf4 = pool.try_get_buffer();
        assert!(buf4.is_none());

        // Recycle one buffer
        pool.recycle_buffer(buf1.unwrap());

        // Should be able to get it back
        let buf5 = pool.try_get_buffer();
        assert!(buf5.is_some());
    }

    #[test]
    fn test_buffer_pool_recycling() {
        let pool = BufferPool::new(2, 512);

        let mut buffer = pool.get_buffer();
        assert_eq!(buffer.len(), 512);

        // Modify buffer
        buffer[0] = 42;
        buffer[100] = 99;

        // Recycle it
        pool.recycle_buffer(buffer);

        // Get it back
        let recycled = pool.get_buffer();
        assert_eq!(recycled.len(), 512);
        // Note: We get a buffer back, but it may be newly allocated or recycled
        // The important thing is that the pool works and provides buffers
    }
}
