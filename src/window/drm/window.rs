//! A headless window whose frames are shown on a display through DRM/KMS.

use super::card::Card;
use super::device;
use super::display_thread::{DisplayCommand, DisplayThread, DisplayThreadConfig};
use super::error::DrmError;
use super::kms::DisplayConfig;
use crate::camera::{Camera2d, Camera3d};
use crate::context::Context;
use crate::post_processing::PostProcessingEffect;
use crate::renderer::Renderer3d;
use crate::scene::{SceneNode2d, SceneNode3d};
use crate::window::{CanvasSetup, NumSamples, Window};
use std::ops::{Deref, DerefMut};

/// A [`Window`] rendered off-screen and shown on a display through DRM/KMS.
///
/// It needs no window system (e.g. a console-only Raspberry Pi) and runs the same
/// render pipeline as a regular window: the scene is rendered into a headless
/// [`Window`] and each finished frame is read back without stalling the GPU, then
/// handed to a display thread. The frame shown is one frame behind the one just
/// rendered.
///
/// The window has the size of the display's selected mode and takes no input.
///
/// The `render*` methods return `false`, like a closed window, when the display thread
/// has stopped and frames can no longer be shown.
///
/// `DrmWindow` dereferences to [`Window`] for everything but rendering. The
/// `render*` methods defined here render and then [`present`](Self::present). The
/// other `render*` methods of [`Window`] (such as `render_3d_with_chain`) do not
/// present: call [`present`](Self::present) after them.
///
/// # Performance on a Raspberry Pi 4
///
/// The GPU is the bottleneck, not the readback or the display. On a Pi 4 (V3D 4.2) at
/// 1024x600, shadow mapping alone costs about 55 ms per frame for 125 cubes and one point
/// light (about 11 FPS); with `set_shadows_enabled(false)` the same scene runs at about
/// 30 FPS. Turn shadows off there.
///
/// # Example
/// ```no_run
/// use kiss3d::prelude::*;
/// use kiss3d::window::DrmWindow;
///
/// #[kiss3d::main]
/// async fn main() {
///     let mut window = DrmWindow::new("/dev/dri/card0").await.unwrap();
///     let mut camera = OrbitCamera3d::default();
///     let mut scene = SceneNode3d::empty();
///     scene.add_cube(1.0, 1.0, 1.0).set_color(RED);
///
///     while window.render_3d(&mut scene, &mut camera).await {}
/// }
/// ```
pub struct DrmWindow {
    // Declared first so the display thread stops before the window releases the GPU.
    display: DisplayThread,
    window: Window,
}

impl DrmWindow {
    /// Opens the DRM device (e.g. `/dev/dri/card0`) and shows frames on its first
    /// connected display, at that display's selected mode.
    pub async fn new(device_path: &str) -> Result<Self, DrmError> {
        let (card, display_config) = device::open_device(device_path)?;
        Self::from_device(card, display_config).await
    }

    /// Like [`new`](Self::new), but picks the device itself: it tries `/dev/dri/card0`,
    /// `card1`, ... in order and uses the first one that can drive a connected display.
    ///
    /// Use it when the display is not always on `card0`, for example on a Raspberry Pi,
    /// where the display controller and the 3D unit are separate nodes. If none works,
    /// the error ([`DrmError::NoUsableDevice`]) says why each one was rejected.
    pub async fn new_auto() -> Result<Self, DrmError> {
        let (card, display_config) = device::open_first_usable()?;
        Self::from_device(card, display_config).await
    }

    async fn from_device(card: Card, display_config: DisplayConfig) -> Result<Self, DrmError> {
        check_graphics_adapter().await?;

        // Rendering is off-screen and single-sampled, so MSAA would only waste memory.
        let setup = CanvasSetup {
            samples: NumSamples::One,
            ..Default::default()
        };
        let window =
            Window::new_headless_with_setup(display_config.width, display_config.height, setup)
                .await;

        // With a software Vulkan device (llvmpipe) next to the GPU, a wrong pick
        // would silently render on the CPU.
        let info = Context::get().adapter.get_info();
        log::info!(
            "Rendering on {:?} ({:?}, {:?})",
            info.name,
            info.device_type,
            info.backend
        );
        if info.device_type == wgpu::DeviceType::Cpu {
            log::warn!("The selected graphics adapter is a software rasterizer");
        }

        let display = DisplayThread::new(
            card,
            DisplayThreadConfig {
                connector: display_config.connector,
                crtc: display_config.crtc,
                mode: display_config.mode,
            },
            display_config.width,
            display_config.height,
        )?;

        Ok(DrmWindow { display, window })
    }

    /// Sends the previously rendered frame to the display and starts the readback
    /// of the one just rendered.
    ///
    /// Called by the `render*` methods of this type; call it yourself after the
    /// other `render*` methods of [`Window`].
    ///
    /// Fails with [`DrmError::DisplayThreadStopped`] once the display thread has
    /// stopped.
    pub fn present(&mut self) -> Result<(), DrmError> {
        // The readback started after the previous frame has had a frame of GPU work
        // to complete, so finishing it does not stall.
        if let Some(frame) = self.window.snap_finish() {
            self.display.send_frame(DisplayCommand {
                width: frame.width(),
                height: frame.height(),
                rgb: frame.into_raw(),
            })?;
        }
        self.window.snap_begin();
        Ok(())
    }

    /// Presents the frame and combines the result with the window's own `running` state.
    fn present_and_report(&mut self, running: bool) -> bool {
        match self.present() {
            Ok(()) => running,
            Err(e) => {
                log::error!("Display output stopped: {}", e);
                false
            }
        }
    }

    /// Renders a 3D scene, then presents it. See [`Window::render_3d`].
    pub async fn render_3d(&mut self, scene: &mut SceneNode3d, camera: &mut impl Camera3d) -> bool {
        let running = self.window.render_3d(scene, camera).await;
        self.present_and_report(running)
    }

    /// Renders a 2D scene, then presents it. See [`Window::render_2d`].
    pub async fn render_2d(&mut self, scene: &mut SceneNode2d, camera: &mut impl Camera2d) -> bool {
        let running = self.window.render_2d(scene, camera).await;
        self.present_and_report(running)
    }

    /// Renders the given scenes, then presents them. See [`Window::render`].
    pub async fn render(
        &mut self,
        scene: Option<&mut SceneNode3d>,
        scene_2d: Option<&mut SceneNode2d>,
        camera: Option<&mut dyn Camera3d>,
        camera_2d: Option<&mut dyn Camera2d>,
        renderer: Option<&mut dyn Renderer3d>,
        post_processing: Option<&mut dyn PostProcessingEffect>,
    ) -> bool {
        let running = self
            .window
            .render(
                scene,
                scene_2d,
                camera,
                camera_2d,
                renderer,
                post_processing,
            )
            .await;
        self.present_and_report(running)
    }
}

/// Checks that a graphics adapter exists. Creating the headless window panics without
/// one, and a missing GPU is better reported as an error.
async fn check_graphics_adapter() -> Result<(), DrmError> {
    if Context::is_initialized() {
        return Ok(());
    }

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .map(|_| ())
        .map_err(|e| DrmError::Init(format!("No suitable graphics adapter found: {}", e)))
}

impl Deref for DrmWindow {
    type Target = Window;

    fn deref(&self) -> &Window {
        &self.window
    }
}

impl DerefMut for DrmWindow {
    fn deref_mut(&mut self) -> &mut Window {
        &mut self.window
    }
}
