//! DRM Canvas for headless rendering without a window manager.

use crate::context::Context;
use crate::resource::OffscreenBuffers;
use std::error::Error;
use std::fmt;

#[cfg(feature = "drm")]
use super::card::Card;
#[cfg(feature = "drm")]
use drm::buffer::DrmFourcc;
#[cfg(feature = "drm")]
use drm::control::{connector, crtc, framebuffer, Device as ControlDevice, Mode, ResourceHandles};
#[cfg(feature = "drm")]
use gbm;
#[cfg(feature = "drm")]
use std::collections::HashMap;

/// Error type for DRM canvas operations.
#[derive(Debug)]
pub enum DrmCanvasError {
    /// Failed to create wgpu adapter.
    NoAdapter,
    /// Failed to request wgpu device.
    DeviceRequest(String),
    /// General initialization error.
    Init(String),
    /// DRM-specific errors.
    DrmError(String),
    /// GBM-specific errors.
    GbmError(String),
    /// Display configuration errors.
    ModesetError(String),
    /// Page flip failures.
    PageFlipError(String),
    /// File I/O errors.
    IoError(std::io::Error),
    /// No connected display found.
    NoConnectedDisplay,
}

impl fmt::Display for DrmCanvasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrmCanvasError::NoAdapter => write!(f, "Failed to find suitable wgpu adapter"),
            DrmCanvasError::DeviceRequest(msg) => write!(f, "Failed to request device: {}", msg),
            DrmCanvasError::Init(msg) => write!(f, "Initialization error: {}", msg),
            DrmCanvasError::DrmError(msg) => write!(f, "DRM error: {}", msg),
            DrmCanvasError::GbmError(msg) => write!(f, "GBM error: {}", msg),
            DrmCanvasError::ModesetError(msg) => write!(f, "Display configuration error: {}", msg),
            DrmCanvasError::PageFlipError(msg) => write!(f, "Page flip error: {}", msg),
            DrmCanvasError::IoError(e) => write!(f, "I/O error: {}", e),
            DrmCanvasError::NoConnectedDisplay => write!(f, "No connected display found"),
        }
    }
}

impl Error for DrmCanvasError {}

impl From<std::io::Error> for DrmCanvasError {
    fn from(err: std::io::Error) -> Self {
        DrmCanvasError::IoError(err)
    }
}

/// Rendering mode for DrmCanvas
enum RenderMode {
    /// Offscreen rendering only (screenshots/recording)
    Offscreen,
    /// Display output via DRM/KMS
    #[cfg(feature = "drm")]
    Display(Box<DrmDisplayState>),
}

/// A canvas for headless rendering using offscreen buffers.
///
/// This canvas initializes wgpu without winit, allowing rendering
/// on systems without a window manager (e.g., console-only Raspberry Pi).
pub struct DrmCanvas {
    /// Offscreen render target
    offscreen_buffers: OffscreenBuffers,
    /// Surface configuration (dimensions and format)
    surface_config: DrmSurfaceConfig,
    /// Depth texture for 3D rendering
    depth_texture: wgpu::Texture,
    /// Depth texture view
    depth_view: wgpu::TextureView,
    /// Rendering mode (offscreen or display)
    mode: RenderMode,
}

/// Configuration for the DRM surface (mimics wgpu::SurfaceConfiguration).
struct DrmSurfaceConfig {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

/// Display configuration discovered from hardware
#[cfg(feature = "drm")]
struct DisplayConfig {
    connector: connector::Handle,
    crtc: crtc::Handle,
    mode: Mode,
    width: u32,
    height: u32,
}

/// Format compatibility information
#[cfg(feature = "drm")]
struct FormatInfo {
    wgpu_format: wgpu::TextureFormat,
    gbm_format: gbm::Format,
    drm_format: DrmFourcc,
    bytes_per_pixel: u32,
}

/// DRM display state for actual screen output
#[cfg(feature = "drm")]
struct DrmDisplayState {
    /// DRM device handle
    card: Card,
    /// Display connector
    connector: connector::Handle,
    /// Display controller (CRTC)
    crtc: crtc::Handle,
    /// Display mode (resolution, refresh rate)
    mode: Mode,
    /// GBM device for buffer allocation
    gbm_device: gbm::Device<Card>,
    /// GBM surface (buffer pool)
    gbm_surface: gbm::Surface<()>,
    /// Framebuffer cache (maps GBM buffer pointers to DRM framebuffer handles)
    framebuffer_cache: HashMap<usize, framebuffer::Handle>,
    /// Currently displayed framebuffer
    current_fb: Option<framebuffer::Handle>,
    /// Buffer being displayed (front buffer)
    front_buffer: Option<gbm::BufferObject<()>>,
    /// Pixel format for DRM
    drm_format: DrmFourcc,
    /// Pixel format for GBM
    gbm_format: gbm::Format,
}

impl DrmCanvas {
    /// Creates a new DRM canvas for headless rendering.
    ///
    /// # Arguments
    /// * `_device_path` - Path to DRM device (e.g., "/dev/dri/card0") - currently unused
    /// * `width` - Width of the render target
    /// * `height` - Height of the render target
    ///
    /// # Returns
    /// A new DrmCanvas ready for offscreen rendering
    ///
    /// # Errors
    /// Returns an error if wgpu initialization fails
    pub async fn new(_device_path: &str, width: u32, height: u32) -> Result<Self, DrmCanvasError> {
        // Ensure minimum dimensions
        let width = width.max(1);
        let height = height.max(1);

        // Initialize wgpu without winit
        Self::init_wgpu_headless().await?;

        let format = wgpu::TextureFormat::Bgra8Unorm;

        let surface_config = DrmSurfaceConfig {
            width,
            height,
            format,
        };

        // Create offscreen render target
        let offscreen_buffers = OffscreenBuffers::new(width, height, format, true);

        // Create depth texture
        let ctxt = Context::get();
        let depth_texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("drm_depth_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self {
            offscreen_buffers,
            surface_config,
            depth_texture,
            depth_view,
            mode: RenderMode::Offscreen,
        })
    }

    /// Creates a new DRM canvas with display output.
    ///
    /// This constructor initializes GBM and sets up the display pipeline for
    /// actual screen output via KMS/DRM.
    ///
    /// # Arguments
    /// * `device_path` - Path to DRM device (e.g., "/dev/dri/card0")
    ///
    /// # Returns
    /// A new DrmCanvas ready for display rendering
    ///
    /// # Errors
    /// Returns an error if:
    /// - DRM device cannot be opened
    /// - No connected display is found
    /// - GBM initialization fails
    /// - wgpu initialization fails
    #[cfg(feature = "drm")]
    pub async fn new_with_display(device_path: &str) -> Result<Self, DrmCanvasError> {
        log::info!("Creating DRM canvas with display output: {}", device_path);

        // Step 1: Open DRM device (for querying)
        let card_query = Card::open(device_path)?;
        log::info!("Opened DRM device: {}", device_path);

        // Step 2: Query display resources
        let display_config = Self::query_display_resources(&card_query)?;
        log::info!(
            "Display configuration: {}x{} @ {}Hz",
            display_config.width,
            display_config.height,
            display_config.mode.vrefresh()
        );

        // Step 3: Choose compatible formats
        let format_info = Self::choose_formats();
        log::info!(
            "Format selection - wgpu: {:?}, gbm: {:?}, drm: {:?}",
            format_info.wgpu_format,
            format_info.gbm_format,
            format_info.drm_format
        );

        // Step 4: Open a separate Card handle for GBM (GBM takes ownership)
        let card_for_gbm = Card::open(device_path)?;

        // Step 5: Create GBM device (takes ownership of card_for_gbm)
        let gbm_device = gbm::Device::new(card_for_gbm)
            .map_err(|e| DrmCanvasError::GbmError(format!("Failed to create GBM device: {}", e)))?;
        log::info!("Created GBM device");

        // Step 6: Create GBM surface
        let gbm_surface = gbm_device
            .create_surface::<()>(
                display_config.width,
                display_config.height,
                format_info.gbm_format,
                gbm::BufferObjectFlags::SCANOUT | gbm::BufferObjectFlags::RENDERING,
            )
            .map_err(|e| {
                DrmCanvasError::GbmError(format!("Failed to create GBM surface: {}", e))
            })?;
        log::info!(
            "Created GBM surface: {}x{}",
            display_config.width,
            display_config.height
        );

        // Step 7: Initialize wgpu (headless for now, will integrate with GBM later)
        Self::init_wgpu_headless().await?;

        // Step 8: Create offscreen buffers for rendering
        let offscreen_buffers = OffscreenBuffers::new(
            display_config.width,
            display_config.height,
            format_info.wgpu_format,
            true,
        );

        // Step 9: Create depth texture
        let ctxt = Context::get();
        let depth_texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("drm_display_depth_texture"),
            size: wgpu::Extent3d {
                width: display_config.width,
                height: display_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Step 10: Create display state (use the query card for DRM operations)
        let display_state = DrmDisplayState {
            card: card_query, // Use the card we opened for querying
            connector: display_config.connector,
            crtc: display_config.crtc,
            mode: display_config.mode,
            gbm_device,
            gbm_surface,
            framebuffer_cache: HashMap::new(),
            current_fb: None,
            front_buffer: None,
            drm_format: format_info.drm_format,
            gbm_format: format_info.gbm_format,
        };

        log::info!("DRM canvas with display created successfully");

        Ok(Self {
            offscreen_buffers,
            surface_config: DrmSurfaceConfig {
                width: display_config.width,
                height: display_config.height,
                format: format_info.wgpu_format,
            },
            depth_texture,
            depth_view,
            mode: RenderMode::Display(Box::new(display_state)),
        })
    }

    /// Initialize wgpu without a window (headless mode).
    async fn init_wgpu_headless() -> Result<(), DrmCanvasError> {
        // Skip initialization if already done (multi-window case)
        if Context::is_initialized() {
            log::info!("wgpu context already initialized, reusing");
            return Ok(());
        }

        log::info!("Initializing wgpu for headless DRM rendering");

        // Create wgpu instance with primary backends (Vulkan on Linux)
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        // Request adapter without a surface (headless)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None, // No surface for headless
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| {
                DrmCanvasError::DeviceRequest(format!("Failed to request adapter: {:?}", e))
            })?;

        log::info!("Adapter info: {:?}", adapter.get_info());

        // Use the adapter's supported limits to ensure compatibility with lower-end hardware
        // like Raspberry Pi (V3D GPU supports max_color_attachments=4, not 8)
        let adapter_limits = adapter.limits();
        log::debug!("Adapter limits: {:?}", adapter_limits);

        // Request device and queue
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("drm_device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .map_err(|e: wgpu::RequestDeviceError| DrmCanvasError::DeviceRequest(e.to_string()))?;

        // Choose surface format (standard for offscreen rendering)
        let surface_format = wgpu::TextureFormat::Bgra8Unorm;

        // Initialize global context
        Context::init(instance, device, queue, adapter, surface_format);
        Context::increment_window_count();

        log::info!("wgpu initialized successfully for DRM");

        Ok(())
    }

    /// Gets the current texture for rendering.
    ///
    /// For DRM canvas, this returns a wrapper around the offscreen color texture.
    pub fn get_current_texture(&self) -> Result<DrmSurfaceTexture<'_>, DrmCanvasError> {
        Ok(DrmSurfaceTexture {
            texture: &self.offscreen_buffers.color_texture,
        })
    }

    /// Returns the depth texture view.
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// Presents the rendered frame.
    ///
    /// For offscreen rendering, this is a no-op. For display mode,
    /// this triggers a page flip to show the rendered frame.
    pub fn present(&mut self) -> Result<(), DrmCanvasError> {
        match &mut self.mode {
            RenderMode::Offscreen => {
                // No-op for offscreen rendering
                Ok(())
            }
            #[cfg(feature = "drm")]
            RenderMode::Display(_display) => {
                // TODO: Implement page flip in Phase 3
                Ok(())
            }
        }
    }

    /// Returns the dimensions of the render target.
    pub fn size(&self) -> (u32, u32) {
        (self.surface_config.width, self.surface_config.height)
    }

    /// Returns the surface format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }

    /// Returns the sample count (always 1 for now).
    pub fn sample_count(&self) -> u32 {
        1
    }

    /// Access to the offscreen buffers (for screenshot capability)
    pub fn offscreen_buffers(&self) -> &OffscreenBuffers {
        &self.offscreen_buffers
    }

    /// Reads pixels from the offscreen framebuffer into a buffer.
    ///
    /// This captures the current rendered frame as RGB pixel data.
    /// Pixels are stored in RGB format (3 bytes per pixel), row by row from bottom to top.
    ///
    /// # Arguments
    /// * `out` - The output buffer. It will be resized to width × height × 3 bytes.
    /// * `x` - The x-coordinate of the region to read (always 0 for now)
    /// * `y` - The y-coordinate of the region to read (always 0 for now)
    /// * `width` - The width of the region to read
    /// * `height` - The height of the region to read
    pub fn read_pixels(&self, out: &mut Vec<u8>, x: usize, y: usize, width: usize, height: usize) {
        let ctxt = Context::get();

        // Calculate buffer size with alignment
        // wgpu requires rows to be aligned to 256 bytes
        let bytes_per_pixel = 4; // RGBA
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = padded_bytes_per_row * height;

        // Create staging buffer
        let staging_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("drm_screenshot_staging_buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy from offscreen texture to staging buffer
        let mut encoder = ctxt.create_command_encoder(Some("drm_screenshot_copy_encoder"));

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.offscreen_buffers.color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: x as u32,
                    y: y as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row as u32),
                    rows_per_image: Some(height as u32),
                },
            },
            wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
        );

        ctxt.submit(std::iter::once(encoder.finish()));

        // Map the buffer and read the data
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        // Wait for the GPU to finish
        let _ = ctxt.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        // Read the data
        let data = buffer_slice.get_mapped_range();

        // Convert from RGBA to RGB and handle row padding
        let rgb_size = width * height * 3;
        out.clear();
        out.reserve(rgb_size);

        // Read rows in reverse order for bottom-left origin compatibility
        for row in (0..height).rev() {
            let row_start = row * padded_bytes_per_row;
            for col in 0..width {
                let pixel_start = row_start + col * bytes_per_pixel;
                // RGBA -> RGB
                out.push(data[pixel_start]); // R
                out.push(data[pixel_start + 1]); // G
                out.push(data[pixel_start + 2]); // B
            }
        }

        drop(data);
        staging_buffer.unmap();
    }
}

/// Wrapper for the surface texture to match wgpu::SurfaceTexture API.
pub struct DrmSurfaceTexture<'a> {
    pub texture: &'a wgpu::Texture,
}

impl Drop for DrmCanvas {
    fn drop(&mut self) {
        // Decrement window count and reset context if this is the last window
        if Context::decrement_window_count() {
            log::info!("Last DRM canvas dropped, resetting wgpu context");
            Context::reset();
        }
    }
}

#[cfg(feature = "drm")]
impl Drop for DrmDisplayState {
    fn drop(&mut self) {
        log::info!("Dropping DRM display state");

        // Destroy all cached framebuffers
        for &fb in self.framebuffer_cache.values() {
            if let Err(e) = self.card.destroy_framebuffer(fb) {
                log::warn!("Failed to destroy framebuffer: {}", e);
            }
        }

        // Release front buffer if held
        if let Some(buffer) = self.front_buffer.take() {
            drop(buffer);
        }

        log::info!("DRM display state cleaned up");
    }
}

// ============================================================================
// Phase 1: Display Resource Query Functions
// ============================================================================

#[cfg(feature = "drm")]
impl DrmCanvas {
    /// Query display resources and find a suitable display configuration.
    fn query_display_resources(card: &Card) -> Result<DisplayConfig, DrmCanvasError> {
        log::info!("Querying DRM display resources");

        // Get resource handles
        let resources = card.resource_handles().map_err(|e| {
            DrmCanvasError::DrmError(format!("Failed to get resource handles: {}", e))
        })?;

        // Find connected connector
        let connector_info = Self::find_connected_connector(card, &resources)?;
        log::info!(
            "Found connected connector: {:?} (id: {:?})",
            connector_info.interface(),
            connector_info.handle()
        );

        // Find available CRTC
        let crtc = Self::find_available_crtc(card, &connector_info, &resources)?;
        log::info!("Selected CRTC: {:?}", crtc);

        // Select best mode
        let mode = Self::select_best_mode(&connector_info)?;
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

    /// Find the first connected connector.
    fn find_connected_connector(
        card: &Card,
        resources: &ResourceHandles,
    ) -> Result<connector::Info, DrmCanvasError> {
        for &conn_handle in resources.connectors() {
            let conn_info = card.get_connector(conn_handle, false).map_err(|e| {
                DrmCanvasError::DrmError(format!("Failed to get connector info: {}", e))
            })?;

            if conn_info.state() == connector::State::Connected {
                return Ok(conn_info);
            }
        }

        Err(DrmCanvasError::NoConnectedDisplay)
    }

    /// Find an available CRTC for the given connector.
    fn find_available_crtc(
        _card: &Card,
        _connector_info: &connector::Info,
        resources: &ResourceHandles,
    ) -> Result<crtc::Handle, DrmCanvasError> {
        // For simplicity, just use the first available CRTC
        // A more sophisticated implementation would check encoder compatibility
        resources
            .crtcs()
            .first()
            .copied()
            .ok_or_else(|| DrmCanvasError::ModesetError("No CRTCs available".to_string()))
    }

    /// Select the best display mode (usually the preferred/native mode).
    fn select_best_mode(connector_info: &connector::Info) -> Result<Mode, DrmCanvasError> {
        let modes = connector_info.modes();

        if modes.is_empty() {
            return Err(DrmCanvasError::ModesetError(
                "No display modes available".to_string(),
            ));
        }

        // The first mode is typically the preferred/native resolution
        Ok(*modes.first().unwrap())
    }

    /// Choose compatible pixel formats for wgpu, GBM, and DRM.
    fn choose_formats() -> FormatInfo {
        // Use XRGB8888 for maximum compatibility with displays
        // Note: wgpu uses BGRA8Unorm which we'll need to convert
        FormatInfo {
            wgpu_format: wgpu::TextureFormat::Bgra8Unorm,
            gbm_format: gbm::Format::Xrgb8888,
            drm_format: DrmFourcc::Xrgb8888,
            bytes_per_pixel: 4,
        }
    }
}
