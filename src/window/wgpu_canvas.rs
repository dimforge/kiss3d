//! Unified wgpu-based canvas for both native and web platforms.

use std::cell::RefCell;
use std::sync::mpsc::Sender;
use std::sync::Arc;

use crate::context::Context;
use crate::event::{Action, Key, Modifiers, MouseButton, TouchAction, WindowEvent};
use crate::window::canvas::{CanvasSetup, NumSamples};
use image::{GenericImage, Pixel};
#[cfg(not(target_arch = "wasm32"))]
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
#[cfg(not(target_arch = "wasm32"))]
use winit::event::{MouseScrollDelta, TouchPhase, WindowEvent as WinitWindowEvent};
#[cfg(not(target_arch = "wasm32"))]
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::EventLoop;
use winit::keyboard::ModifiersState;
#[cfg(not(target_arch = "wasm32"))]
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Icon, Window, WindowAttributes};

#[cfg(target_arch = "wasm32")]
use std::rc::Rc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

// Thread-local EventLoop singleton for native platforms.
// winit only allows one EventLoop per program, so we store it in thread-local
// storage and reuse it across window recreations. EventLoop is not Send/Sync,
// so we use thread_local! instead of a static Mutex.
#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    static EVENT_LOOP: RefCell<Option<EventLoop<()>>> = const { RefCell::new(None) };
    // Shared event storage for multi-window support. Events are stored per window_id
    // so each window can retrieve only its own events after pump_app_events runs.
    static PENDING_WINDOW_EVENTS: RefCell<std::collections::HashMap<winit::window::WindowId, Vec<PendingEvent>>> = RefCell::new(std::collections::HashMap::new());
}

/// Internal event type that stores both the event data and state updates needed.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
enum PendingEvent {
    WindowEvent(WindowEvent),
    ButtonState(MouseButton, Action),
    KeyState(Key, Action),
    CursorPos(f64, f64),
    #[allow(dead_code)]
    Modifiers(ModifiersState),
    Resize {
        width: u32,
        height: u32,
    },
}

/// A unified canvas based on wgpu that works on both native and web platforms.
#[allow(dead_code)]
pub struct WgpuCanvas {
    window: Arc<Window>,
    #[cfg(not(target_arch = "wasm32"))]
    window_id: winit::window::WindowId,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    cursor_pos: Option<(f64, f64)>,
    key_states: [Action; Key::Unknown as usize + 1],
    button_states: [Action; MouseButton::Button8 as usize + 1],
    out_events: Sender<WindowEvent>,
    modifiers_state: ModifiersState,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// Multisampling texture for MSAA (if enabled)
    msaa_texture: Option<wgpu::Texture>,
    msaa_view: Option<wgpu::TextureView>,
    /// Number of samples for MSAA
    sample_count: u32,
    /// Texture for reading back pixels (for screenshots)
    readback_texture: wgpu::Texture,
    /// Pending events from web callbacks (WASM only)
    #[cfg(target_arch = "wasm32")]
    pending_events: Rc<RefCell<Vec<WindowEvent>>>,
    /// Keep closures alive (WASM only)
    #[cfg(target_arch = "wasm32")]
    _event_closures: Vec<wasm_bindgen::JsValue>,
}

impl WgpuCanvas {
    /// Opens a new window and initializes the wgpu context.
    pub async fn open(
        title: &str,
        hide: bool,
        width: u32,
        height: u32,
        canvas_setup: Option<CanvasSetup>,
        out_events: Sender<WindowEvent>,
    ) -> Self {
        let window_attrs = WindowAttributes::default()
            .with_title(title)
            .with_inner_size(LogicalSize::new(width as f64, height as f64))
            .with_visible(!hide);

        let canvas_setup = canvas_setup.unwrap_or(CanvasSetup {
            vsync: true,
            samples: NumSamples::Zero,
        });

        // Create the window
        #[cfg(not(target_arch = "wasm32"))]
        let window = {
            // Get or create the thread-local EventLoop (winit only allows one per program)
            EVENT_LOOP.with(|event_loop_cell| {
                let mut event_loop_opt = event_loop_cell.borrow_mut();
                if event_loop_opt.is_none() {
                    *event_loop_opt = Some(EventLoop::new().expect("Failed to create event loop"));
                }
                let event_loop = event_loop_opt.as_ref().unwrap();
                #[allow(deprecated)]
                event_loop
                    .create_window(window_attrs)
                    .expect("Failed to create window")
            })
        };

        #[cfg(target_arch = "wasm32")]
        let window = {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            // For WASM, we create a local EventLoop (single-threaded environment)
            let events = EventLoop::new().expect("Failed to create event loop");

            let web_window = web_sys::window().expect("Failed to get web_sys window");
            let document = web_window.document().expect("Failed to get document");

            // Try to find an existing canvas with id "canvas", or create one
            let canvas = document
                .get_element_by_id("canvas")
                .and_then(|elem| elem.dyn_into::<web_sys::HtmlCanvasElement>().ok())
                .unwrap_or_else(|| {
                    // Create a new canvas element
                    let canvas = document
                        .create_element("canvas")
                        .expect("Failed to create canvas element")
                        .dyn_into::<web_sys::HtmlCanvasElement>()
                        .expect("Failed to cast to HtmlCanvasElement");
                    canvas.set_id("canvas");

                    // Append to body
                    if let Some(body) = document.body() {
                        body.append_child(&canvas)
                            .expect("Failed to append canvas to body");
                    }

                    canvas
                });

            // Style html and body to fill 100%
            if let Some(html) = document.document_element() {
                if let Some(html) = html.dyn_ref::<web_sys::HtmlElement>() {
                    let style = html.style();
                    let _ = style.set_property("margin", "0");
                    let _ = style.set_property("padding", "0");
                    let _ = style.set_property("width", "100%");
                    let _ = style.set_property("height", "100%");
                }
            }
            if let Some(body) = document.body() {
                let style = body.style();
                let _ = style.set_property("margin", "0");
                let _ = style.set_property("padding", "0");
                let _ = style.set_property("width", "100%");
                let _ = style.set_property("height", "100%");
                let _ = style.set_property("overflow", "hidden");
            }

            let window_attrs = window_attrs.with_canvas(Some(canvas));

            #[allow(deprecated)]
            let window = events
                .create_window(window_attrs)
                .expect("Failed to create window");

            // Style the canvas AFTER winit creates the window (winit may overwrite styles)
            use winit::platform::web::WindowExtWebSys;
            if let Some(canvas) = window.canvas() {
                let style = canvas.style();
                let _ = style.set_property("display", "block");
                let _ = style.set_property("width", "100%");
                let _ = style.set_property("height", "100%");
            }

            window
        };

        let window = Arc::new(window);

        // Check if we already have a context initialized (multi-window case)
        let (surface, surface_format) = if Context::is_initialized() {
            // Reuse the existing context - create a new surface using the shared instance
            let ctxt = Context::get();

            let surface = ctxt
                .instance
                .create_surface(window.clone())
                .expect("Failed to create surface");

            // Configure surface with existing device
            let surface_caps = surface.get_capabilities(&ctxt.adapter);
            let surface_format = surface_caps
                .formats
                .iter()
                .find(|f| !f.is_srgb())
                .copied()
                .unwrap_or(surface_caps.formats[0]);

            (surface, surface_format)
        } else {
            // First window - create the full wgpu context
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            // Create surface
            let surface = instance
                .create_surface(window.clone())
                .expect("Failed to create surface");

            // Request adapter (async on all platforms)
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::default(),
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .expect("Failed to find an appropriate adapter");

            // Request device (async on all platforms)
            // Use downlevel defaults for WebGL2 compatibility
            #[cfg(target_arch = "wasm32")]
            let limits = wgpu::Limits::downlevel_webgl2_defaults();
            #[cfg(not(target_arch = "wasm32"))]
            let limits = wgpu::Limits::default();

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("kiss3d device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                })
                .await
                .expect("Failed to create device");

            // Get surface capabilities
            // We explicitly prefer non-sRGB formats for consistent behavior across platforms.
            // WebGL2 often doesn't support sRGB framebuffers, so we do manual gamma correction
            // in shaders instead. This ensures colors look the same on native and web.
            let surface_caps = surface.get_capabilities(&adapter);
            let surface_format = surface_caps
                .formats
                .iter()
                .find(|f| !f.is_srgb())
                .copied()
                .unwrap_or(surface_caps.formats[0]);

            // Initialize the global context (only for first window)
            Context::init(instance, device, queue, adapter, surface_format);

            (surface, surface_format)
        };

        let ctxt = Context::get();

        // Get surface capabilities for alpha mode
        let surface_caps = surface.get_capabilities(&ctxt.adapter);

        // Get the actual window size
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        // Configure surface
        let present_mode = if canvas_setup.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: surface_format,
            width,
            height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&ctxt.device, &surface_config);

        // Create depth texture
        let (depth_texture, depth_view) =
            Self::create_depth_texture(&ctxt.device, width, height, canvas_setup.samples as u32);

        // Create MSAA texture if needed
        let sample_count = canvas_setup.samples as u32;
        let (msaa_texture, msaa_view) = if sample_count > 1 {
            let (tex, view) = Self::create_msaa_texture(
                &ctxt.device,
                width,
                height,
                surface_format,
                sample_count,
            );
            (Some(tex), Some(view))
        } else {
            (None, None)
        };

        // Create readback texture for screenshots
        let readback_texture =
            Self::create_readback_texture(&ctxt.device, width, height, surface_format);

        // Set up WASM event listeners
        #[cfg(target_arch = "wasm32")]
        let (pending_events, _event_closures) = {
            use winit::platform::web::WindowExtWebSys;

            let pending_events = Rc::new(RefCell::new(Vec::new()));
            let mut closures: Vec<wasm_bindgen::JsValue> = Vec::new();

            if let Some(canvas) = window.canvas() {
                // Pointer move (using pointer events for consistency)
                {
                    let pending = pending_events.clone();
                    let canvas_clone = canvas.clone();
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::PointerEvent| {
                            // Get coordinates relative to canvas, accounting for CSS scaling
                            let rect = canvas_clone.get_bounding_client_rect();
                            let css_x = event.client_x() as f64 - rect.left();
                            let css_y = event.client_y() as f64 - rect.top();

                            // Scale from CSS pixels to canvas pixels
                            let scale_x = canvas_clone.width() as f64 / rect.width();
                            let scale_y = canvas_clone.height() as f64 / rect.height();
                            let x = css_x * scale_x;
                            let y = css_y * scale_y;

                            pending.borrow_mut().push(WindowEvent::CursorPos(
                                x,
                                y,
                                Modifiers::empty(),
                            ));
                        });
                    let _ = canvas.add_event_listener_with_callback(
                        "pointermove",
                        closure.as_ref().unchecked_ref(),
                    );
                    closures.push(closure.into_js_value());
                }

                // Pointer down
                {
                    let pending = pending_events.clone();
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::PointerEvent| {
                            // Only handle mouse pointer type (not touch - that's handled separately)
                            if event.pointer_type() == "mouse" {
                                let button = translate_web_mouse_button(event.button());
                                pending.borrow_mut().push(WindowEvent::MouseButton(
                                    button,
                                    Action::Press,
                                    Modifiers::empty(),
                                ));
                            }
                        });
                    let _ = canvas.add_event_listener_with_callback(
                        "pointerdown",
                        closure.as_ref().unchecked_ref(),
                    );
                    closures.push(closure.into_js_value());
                }

                // Pointer up
                {
                    let pending = pending_events.clone();
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::PointerEvent| {
                            // Only handle mouse pointer type (not touch - that's handled separately)
                            if event.pointer_type() == "mouse" {
                                let button = translate_web_mouse_button(event.button());
                                pending.borrow_mut().push(WindowEvent::MouseButton(
                                    button,
                                    Action::Release,
                                    Modifiers::empty(),
                                ));
                            }
                        });
                    let _ = canvas.add_event_listener_with_callback(
                        "pointerup",
                        closure.as_ref().unchecked_ref(),
                    );
                    closures.push(closure.into_js_value());
                }

                // Wheel
                {
                    let pending = pending_events.clone();
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::WheelEvent| {
                            // Prevent default scrolling behavior
                            event.prevent_default();
                            // Scale based on delta mode to match native behavior:
                            // Browsers report much larger pixel deltas than native platforms,
                            // so we normalize them to produce similar scroll behavior.
                            // 0 = DOM_DELTA_PIXEL, 1 = DOM_DELTA_LINE, 2 = DOM_DELTA_PAGE
                            let scale = match event.delta_mode() {
                                0 => 0.1,  // Pixel mode - scale down (browsers report ~100px per tick)
                                1 => 1.0,  // Line mode - use as-is (browsers report ~1-3 lines)
                                _ => 10.0, // Page mode - scale up slightly
                            };
                            let dx = event.delta_x() * scale;
                            let dy = -event.delta_y() * scale; // Invert for natural scrolling
                            pending.borrow_mut().push(WindowEvent::Scroll(
                                dx,
                                dy,
                                Modifiers::empty(),
                            ));
                        });
                    let _ = canvas.add_event_listener_with_callback(
                        "wheel",
                        closure.as_ref().unchecked_ref(),
                    );
                    closures.push(closure.into_js_value());
                }

                // Context menu (prevent right-click menu)
                {
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
                            event.prevent_default();
                        });
                    let _ = canvas.add_event_listener_with_callback(
                        "contextmenu",
                        closure.as_ref().unchecked_ref(),
                    );
                    closures.push(closure.into_js_value());
                }

                // Touch events
                {
                    let pending = pending_events.clone();
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::TouchEvent| {
                            event.prevent_default();
                            let touches = event.changed_touches();
                            for i in 0..touches.length() {
                                if let Some(touch) = touches.get(i) {
                                    pending.borrow_mut().push(WindowEvent::Touch(
                                        touch.identifier() as u64,
                                        touch.client_x() as f64,
                                        touch.client_y() as f64,
                                        TouchAction::Start,
                                        Modifiers::empty(),
                                    ));
                                }
                            }
                        });
                    let _ = canvas.add_event_listener_with_callback(
                        "touchstart",
                        closure.as_ref().unchecked_ref(),
                    );
                    closures.push(closure.into_js_value());
                }

                {
                    let pending = pending_events.clone();
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::TouchEvent| {
                            event.prevent_default();
                            let touches = event.changed_touches();
                            for i in 0..touches.length() {
                                if let Some(touch) = touches.get(i) {
                                    pending.borrow_mut().push(WindowEvent::Touch(
                                        touch.identifier() as u64,
                                        touch.client_x() as f64,
                                        touch.client_y() as f64,
                                        TouchAction::Move,
                                        Modifiers::empty(),
                                    ));
                                }
                            }
                        });
                    let _ = canvas.add_event_listener_with_callback(
                        "touchmove",
                        closure.as_ref().unchecked_ref(),
                    );
                    closures.push(closure.into_js_value());
                }

                {
                    let pending = pending_events.clone();
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::TouchEvent| {
                            event.prevent_default();
                            let touches = event.changed_touches();
                            for i in 0..touches.length() {
                                if let Some(touch) = touches.get(i) {
                                    pending.borrow_mut().push(WindowEvent::Touch(
                                        touch.identifier() as u64,
                                        touch.client_x() as f64,
                                        touch.client_y() as f64,
                                        TouchAction::End,
                                        Modifiers::empty(),
                                    ));
                                }
                            }
                        });
                    let _ = canvas.add_event_listener_with_callback(
                        "touchend",
                        closure.as_ref().unchecked_ref(),
                    );
                    closures.push(closure.into_js_value());
                }

                {
                    let pending = pending_events.clone();
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::TouchEvent| {
                            event.prevent_default();
                            let touches = event.changed_touches();
                            for i in 0..touches.length() {
                                if let Some(touch) = touches.get(i) {
                                    pending.borrow_mut().push(WindowEvent::Touch(
                                        touch.identifier() as u64,
                                        touch.client_x() as f64,
                                        touch.client_y() as f64,
                                        TouchAction::Cancel,
                                        Modifiers::empty(),
                                    ));
                                }
                            }
                        });
                    let _ = canvas.add_event_listener_with_callback(
                        "touchcancel",
                        closure.as_ref().unchecked_ref(),
                    );
                    closures.push(closure.into_js_value());
                }
            }

            // Keyboard events on window (document level)
            let web_window = web_sys::window().expect("Failed to get web_sys window");
            {
                let pending = pending_events.clone();
                let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::KeyboardEvent| {
                    let key = translate_web_key(&event.code());
                    pending.borrow_mut().push(WindowEvent::Key(
                        key,
                        Action::Press,
                        Modifiers::empty(),
                    ));
                });
                let _ = web_window
                    .add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref());
                closures.push(closure.into_js_value());
            }

            {
                let pending = pending_events.clone();
                let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::KeyboardEvent| {
                    let key = translate_web_key(&event.code());
                    pending.borrow_mut().push(WindowEvent::Key(
                        key,
                        Action::Release,
                        Modifiers::empty(),
                    ));
                });
                let _ = web_window
                    .add_event_listener_with_callback("keyup", closure.as_ref().unchecked_ref());
                closures.push(closure.into_js_value());
            }

            (pending_events, closures)
        };

        #[cfg(not(target_arch = "wasm32"))]
        let window_id = window.id();

        WgpuCanvas {
            window,
            #[cfg(not(target_arch = "wasm32"))]
            window_id,
            surface,
            surface_config,
            cursor_pos: None,
            key_states: [Action::Release; Key::Unknown as usize + 1],
            button_states: [Action::Release; MouseButton::Button8 as usize + 1],
            out_events,
            modifiers_state: ModifiersState::default(),
            depth_texture,
            depth_view,
            msaa_texture,
            msaa_view,
            sample_count,
            readback_texture,
            #[cfg(target_arch = "wasm32")]
            pending_events,
            #[cfg(target_arch = "wasm32")]
            _event_closures,
        }
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let sample_count = sample_count.max(1);
        // Ensure minimum dimensions of 1x1 to avoid wgpu validation errors
        let width = width.max(1);
        let height = height.max(1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: Context::depth_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_msaa_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        // Ensure minimum dimensions of 1x1 to avoid wgpu validation errors
        let width = width.max(1);
        let height = height.max(1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("msaa_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_readback_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::Texture {
        // Ensure minimum dimensions of 1x1 to avoid wgpu validation errors
        let width = width.max(1);
        let height = height.max(1);
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("readback_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    /// Polls events from the window system.
    pub fn poll_events(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use winit::platform::pump_events::EventLoopExtPumpEvents;

            // First, pump all events into the shared storage
            struct EventCollector;

            impl ApplicationHandler for EventCollector {
                fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

                fn window_event(
                    &mut self,
                    _event_loop: &ActiveEventLoop,
                    window_id: winit::window::WindowId,
                    event: WinitWindowEvent,
                ) {
                    let pending_events: Vec<PendingEvent> = match event {
                        WinitWindowEvent::CloseRequested => {
                            vec![PendingEvent::WindowEvent(WindowEvent::Close)]
                        }
                        WinitWindowEvent::Resized(physical_size) => {
                            if physical_size.width > 0 && physical_size.height > 0 {
                                vec![
                                    PendingEvent::Resize {
                                        width: physical_size.width,
                                        height: physical_size.height,
                                    },
                                    PendingEvent::WindowEvent(WindowEvent::FramebufferSize(
                                        physical_size.width,
                                        physical_size.height,
                                    )),
                                ]
                            } else {
                                vec![]
                            }
                        }
                        WinitWindowEvent::CursorMoved { position, .. } => {
                            vec![
                                PendingEvent::CursorPos(position.x, position.y),
                                PendingEvent::WindowEvent(WindowEvent::CursorPos(
                                    position.x,
                                    position.y,
                                    Modifiers::empty(), // Will be filled in when processing
                                )),
                            ]
                        }
                        WinitWindowEvent::MouseInput { state, button, .. } => {
                            let action = translate_action(state);
                            let button = translate_mouse_button(button);
                            vec![
                                PendingEvent::ButtonState(button, action),
                                PendingEvent::WindowEvent(WindowEvent::MouseButton(
                                    button,
                                    action,
                                    Modifiers::empty(),
                                )),
                            ]
                        }
                        WinitWindowEvent::Touch(touch) => {
                            let action = match touch.phase {
                                TouchPhase::Started => TouchAction::Start,
                                TouchPhase::Ended => TouchAction::End,
                                TouchPhase::Moved => TouchAction::Move,
                                TouchPhase::Cancelled => TouchAction::Cancel,
                            };
                            vec![PendingEvent::WindowEvent(WindowEvent::Touch(
                                touch.id,
                                touch.location.x,
                                touch.location.y,
                                action,
                                Modifiers::empty(),
                            ))]
                        }
                        WinitWindowEvent::MouseWheel { delta, .. } => {
                            let (x, y) = match delta {
                                MouseScrollDelta::LineDelta(dx, dy) => {
                                    (dx as f64 * 10.0, dy as f64 * 10.0)
                                }
                                MouseScrollDelta::PixelDelta(delta) => (delta.x, delta.y),
                            };
                            vec![PendingEvent::WindowEvent(WindowEvent::Scroll(
                                x,
                                y,
                                Modifiers::empty(),
                            ))]
                        }
                        WinitWindowEvent::KeyboardInput { event, .. } => {
                            let action = translate_action(event.state);
                            let key = translate_key(event.physical_key);
                            let mut events = vec![
                                PendingEvent::KeyState(key, action),
                                PendingEvent::WindowEvent(WindowEvent::Key(
                                    key,
                                    action,
                                    Modifiers::empty(),
                                )),
                            ];
                            if let winit::keyboard::Key::Character(ref c) = event.logical_key {
                                for ch in c.chars() {
                                    events.push(PendingEvent::WindowEvent(WindowEvent::Char(ch)));
                                }
                            }
                            events
                        }
                        WinitWindowEvent::ModifiersChanged(new_modifiers) => {
                            vec![PendingEvent::Modifiers(new_modifiers.state())]
                        }
                        _ => vec![],
                    };

                    if !pending_events.is_empty() {
                        PENDING_WINDOW_EVENTS.with(|storage| {
                            storage
                                .borrow_mut()
                                .entry(window_id)
                                .or_default()
                                .extend(pending_events);
                        });
                    }
                }
            }

            let timeout = Some(std::time::Duration::ZERO);
            EVENT_LOOP.with(|event_loop_cell| {
                if let Some(ref mut event_loop) = *event_loop_cell.borrow_mut() {
                    let mut collector = EventCollector;
                    let _ = event_loop.pump_app_events(timeout, &mut collector);
                }
            });

            // Now process only this window's events
            let events = PENDING_WINDOW_EVENTS.with(|storage| {
                storage
                    .borrow_mut()
                    .remove(&self.window_id)
                    .unwrap_or_default()
            });

            for event in events {
                match event {
                    PendingEvent::WindowEvent(we) => {
                        let _ = self.out_events.send(we);
                    }
                    PendingEvent::ButtonState(button, action) => {
                        self.button_states[button as usize] = action;
                    }
                    PendingEvent::KeyState(key, action) => {
                        self.key_states[key as usize] = action;
                    }
                    PendingEvent::CursorPos(x, y) => {
                        self.cursor_pos = Some((x, y));
                    }
                    PendingEvent::Modifiers(m) => {
                        self.modifiers_state = m;
                    }
                    PendingEvent::Resize { width, height } => {
                        let ctxt = Context::get();

                        // Resize surface
                        self.surface_config.width = width;
                        self.surface_config.height = height;
                        self.surface.configure(&ctxt.device, &self.surface_config);

                        // Recreate depth texture
                        let (new_depth, new_depth_view) = Self::create_depth_texture(
                            &ctxt.device,
                            width,
                            height,
                            self.sample_count,
                        );
                        self.depth_texture = new_depth;
                        self.depth_view = new_depth_view;

                        // Recreate MSAA texture if needed
                        if self.sample_count > 1 {
                            let (new_msaa, new_msaa_view) = Self::create_msaa_texture(
                                &ctxt.device,
                                width,
                                height,
                                self.surface_config.format,
                                self.sample_count,
                            );
                            self.msaa_texture = Some(new_msaa);
                            self.msaa_view = Some(new_msaa_view);
                        }

                        // Recreate readback texture
                        self.readback_texture = Self::create_readback_texture(
                            &ctxt.device,
                            width,
                            height,
                            self.surface_config.format,
                        );
                    }
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Check for resize - compare current window size to surface config
            let current_size = self.window.inner_size();
            if current_size.width > 0
                && current_size.height > 0
                && (current_size.width != self.surface_config.width
                    || current_size.height != self.surface_config.height)
            {
                let ctxt = Context::get();

                // Resize surface
                self.surface_config.width = current_size.width;
                self.surface_config.height = current_size.height;
                self.surface.configure(&ctxt.device, &self.surface_config);

                // Recreate depth texture
                let (new_depth, new_depth_view) = Self::create_depth_texture(
                    &ctxt.device,
                    current_size.width,
                    current_size.height,
                    self.sample_count,
                );
                self.depth_texture = new_depth;
                self.depth_view = new_depth_view;

                // Recreate MSAA texture if needed
                if self.sample_count > 1 {
                    let (new_msaa, new_msaa_view) = Self::create_msaa_texture(
                        &ctxt.device,
                        current_size.width,
                        current_size.height,
                        self.surface_config.format,
                        self.sample_count,
                    );
                    self.msaa_texture = Some(new_msaa);
                    self.msaa_view = Some(new_msaa_view);
                }

                // Recreate readback texture
                self.readback_texture = Self::create_readback_texture(
                    &ctxt.device,
                    current_size.width,
                    current_size.height,
                    self.surface_config.format,
                );

                let _ = self.out_events.send(WindowEvent::FramebufferSize(
                    current_size.width,
                    current_size.height,
                ));
            }

            // Process pending events from web callbacks
            let events: Vec<WindowEvent> = self.pending_events.borrow_mut().drain(..).collect();
            for event in events {
                match &event {
                    WindowEvent::CursorPos(x, y, _) => {
                        self.cursor_pos = Some((*x, *y));
                    }
                    WindowEvent::MouseButton(button, action, _) => {
                        self.button_states[*button as usize] = *action;
                    }
                    WindowEvent::Key(key, action, _) => {
                        self.key_states[*key as usize] = *action;
                    }
                    _ => {}
                }
                let _ = self.out_events.send(event);
            }
        }
    }

    /// Gets the current surface texture for rendering.
    pub fn get_current_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }

    /// Copies the frame texture to the readback texture for later reading.
    pub fn copy_frame_to_readback(&self, frame: &wgpu::SurfaceTexture) {
        let ctxt = Context::get();
        let mut encoder = ctxt.create_command_encoder(Some("readback_copy_encoder"));

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &frame.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.readback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
        );

        ctxt.submit(std::iter::once(encoder.finish()));
    }

    /// Presents the current frame.
    pub fn present(&self, frame: wgpu::SurfaceTexture) {
        frame.present();
    }

    /// Reads pixels from the readback texture into the provided buffer.
    /// Returns RGB data (3 bytes per pixel).
    pub fn read_pixels(&self, out: &mut Vec<u8>, x: usize, y: usize, width: usize, height: usize) {
        let ctxt = Context::get();

        // Calculate buffer size with alignment
        // wgpu requires rows to be aligned to 256 bytes
        let bytes_per_pixel = 4; // RGBA or BGRA
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = padded_bytes_per_row * height;

        // Create staging buffer
        let staging_buffer = ctxt.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot_staging_buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy from readback texture to staging buffer
        let mut encoder = ctxt.create_command_encoder(Some("screenshot_copy_encoder"));

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.readback_texture,
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
        let _ = ctxt.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();

        // Read the data
        let data = buffer_slice.get_mapped_range();

        // Convert from BGRA/RGBA to RGB and handle row padding
        let rgb_size = width * height * 3;
        out.clear();
        out.reserve(rgb_size);

        let is_bgra = matches!(
            self.surface_config.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );

        // wgpu has origin at top-left, but we want bottom-left origin for OpenGL compatibility
        // So we read rows in reverse order
        for row in (0..height).rev() {
            let row_start = row * padded_bytes_per_row;
            for col in 0..width {
                let pixel_start = row_start + col * bytes_per_pixel;
                if is_bgra {
                    // BGRA -> RGB
                    out.push(data[pixel_start + 2]); // R
                    out.push(data[pixel_start + 1]); // G
                    out.push(data[pixel_start]); // B
                } else {
                    // RGBA -> RGB
                    out.push(data[pixel_start]); // R
                    out.push(data[pixel_start + 1]); // G
                    out.push(data[pixel_start + 2]); // B
                }
            }
        }

        drop(data);
        staging_buffer.unmap();
    }

    /// Gets the depth texture view for rendering.
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// Gets the MSAA texture view if MSAA is enabled.
    pub fn msaa_view(&self) -> Option<&wgpu::TextureView> {
        self.msaa_view.as_ref()
    }

    /// Gets the sample count for MSAA.
    pub fn sample_count(&self) -> u32 {
        self.sample_count.max(1)
    }

    /// Gets the surface format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }

    /// The size of the render surface.
    ///
    /// This returns the configured surface size, which matches the depth texture
    /// and is guaranteed to be consistent with the render targets.
    pub fn size(&self) -> (u32, u32) {
        (self.surface_config.width, self.surface_config.height)
    }

    /// The current position of the cursor, if known.
    pub fn cursor_pos(&self) -> Option<(f64, f64)> {
        self.cursor_pos
    }

    /// The scale factor.
    pub fn scale_factor(&self) -> f64 {
        self.window.scale_factor()
    }

    /// Set the window title.
    pub fn set_title(&mut self, title: &str) {
        self.window.set_title(title)
    }

    /// Set the window icon.
    pub fn set_icon(&mut self, icon: impl GenericImage<Pixel = impl Pixel<Subpixel = u8>>) {
        let (width, height) = icon.dimensions();
        let mut rgba = Vec::with_capacity((width * height) as usize * 4);
        for (_, _, pixel) in icon.pixels() {
            rgba.extend_from_slice(&pixel.to_rgba().0);
        }
        let icon = Icon::from_rgba(rgba, width, height).unwrap();
        self.window.set_window_icon(Some(icon))
    }

    /// Set the cursor grabbing behaviour.
    pub fn set_cursor_grab(&self, grab: bool) {
        use winit::window::CursorGrabMode;
        let mode = if grab {
            CursorGrabMode::Confined
        } else {
            CursorGrabMode::None
        };
        let _ = self.window.set_cursor_grab(mode);
    }

    /// Set the cursor position.
    pub fn set_cursor_position(&self, x: f64, y: f64) {
        let _ = self
            .window
            .set_cursor_position(winit::dpi::PhysicalPosition::new(x, y));
    }

    /// Toggle the cursor visibility.
    pub fn hide_cursor(&self, hide: bool) {
        self.window.set_cursor_visible(!hide)
    }

    /// Hide the window.
    pub fn hide(&mut self) {
        self.window.set_visible(false)
    }

    /// Show the window.
    pub fn show(&mut self) {
        self.window.set_visible(true)
    }

    /// The state of a mouse button.
    pub fn get_mouse_button(&self, button: MouseButton) -> Action {
        self.button_states[button as usize]
    }

    /// The state of a key.
    pub fn get_key(&self, key: Key) -> Action {
        self.key_states[key as usize]
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn translate_action(action: winit::event::ElementState) -> Action {
    use winit::event::ElementState;
    match action {
        ElementState::Pressed => Action::Press,
        ElementState::Released => Action::Release,
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn translate_modifiers(modifiers: ModifiersState) -> Modifiers {
    let mut res = Modifiers::empty();
    if modifiers.shift_key() {
        res.insert(Modifiers::Shift)
    }
    if modifiers.control_key() {
        res.insert(Modifiers::Control)
    }
    if modifiers.alt_key() {
        res.insert(Modifiers::Alt)
    }
    if modifiers.super_key() {
        res.insert(Modifiers::Super)
    }
    res
}

#[cfg(not(target_arch = "wasm32"))]
fn translate_mouse_button(button: winit::event::MouseButton) -> MouseButton {
    match button {
        winit::event::MouseButton::Left => MouseButton::Button1,
        winit::event::MouseButton::Right => MouseButton::Button2,
        winit::event::MouseButton::Middle => MouseButton::Button3,
        _ => MouseButton::Button4,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn translate_key(physical_key: PhysicalKey) -> Key {
    if let PhysicalKey::Code(key_code) = physical_key {
        match key_code {
            KeyCode::Digit1 => Key::Key1,
            KeyCode::Digit2 => Key::Key2,
            KeyCode::Digit3 => Key::Key3,
            KeyCode::Digit4 => Key::Key4,
            KeyCode::Digit5 => Key::Key5,
            KeyCode::Digit6 => Key::Key6,
            KeyCode::Digit7 => Key::Key7,
            KeyCode::Digit8 => Key::Key8,
            KeyCode::Digit9 => Key::Key9,
            KeyCode::Digit0 => Key::Key0,
            KeyCode::KeyA => Key::A,
            KeyCode::KeyB => Key::B,
            KeyCode::KeyC => Key::C,
            KeyCode::KeyD => Key::D,
            KeyCode::KeyE => Key::E,
            KeyCode::KeyF => Key::F,
            KeyCode::KeyG => Key::G,
            KeyCode::KeyH => Key::H,
            KeyCode::KeyI => Key::I,
            KeyCode::KeyJ => Key::J,
            KeyCode::KeyK => Key::K,
            KeyCode::KeyL => Key::L,
            KeyCode::KeyM => Key::M,
            KeyCode::KeyN => Key::N,
            KeyCode::KeyO => Key::O,
            KeyCode::KeyP => Key::P,
            KeyCode::KeyQ => Key::Q,
            KeyCode::KeyR => Key::R,
            KeyCode::KeyS => Key::S,
            KeyCode::KeyT => Key::T,
            KeyCode::KeyU => Key::U,
            KeyCode::KeyV => Key::V,
            KeyCode::KeyW => Key::W,
            KeyCode::KeyX => Key::X,
            KeyCode::KeyY => Key::Y,
            KeyCode::KeyZ => Key::Z,
            KeyCode::Escape => Key::Escape,
            KeyCode::F1 => Key::F1,
            KeyCode::F2 => Key::F2,
            KeyCode::F3 => Key::F3,
            KeyCode::F4 => Key::F4,
            KeyCode::F5 => Key::F5,
            KeyCode::F6 => Key::F6,
            KeyCode::F7 => Key::F7,
            KeyCode::F8 => Key::F8,
            KeyCode::F9 => Key::F9,
            KeyCode::F10 => Key::F10,
            KeyCode::F11 => Key::F11,
            KeyCode::F12 => Key::F12,
            KeyCode::F13 => Key::F13,
            KeyCode::F14 => Key::F14,
            KeyCode::F15 => Key::F15,
            KeyCode::F16 => Key::F16,
            KeyCode::F17 => Key::F17,
            KeyCode::F18 => Key::F18,
            KeyCode::F19 => Key::F19,
            KeyCode::F20 => Key::F20,
            KeyCode::F21 => Key::F21,
            KeyCode::F22 => Key::F22,
            KeyCode::F23 => Key::F23,
            KeyCode::F24 => Key::F24,
            KeyCode::PrintScreen => Key::Snapshot,
            KeyCode::ScrollLock => Key::Scroll,
            KeyCode::Pause => Key::Pause,
            KeyCode::Insert => Key::Insert,
            KeyCode::Home => Key::Home,
            KeyCode::Delete => Key::Delete,
            KeyCode::End => Key::End,
            KeyCode::PageDown => Key::PageDown,
            KeyCode::PageUp => Key::PageUp,
            KeyCode::ArrowLeft => Key::Left,
            KeyCode::ArrowUp => Key::Up,
            KeyCode::ArrowRight => Key::Right,
            KeyCode::ArrowDown => Key::Down,
            KeyCode::Backspace => Key::Back,
            KeyCode::Enter => Key::Return,
            KeyCode::Space => Key::Space,
            KeyCode::NumLock => Key::Numlock,
            KeyCode::Numpad0 => Key::Numpad0,
            KeyCode::Numpad1 => Key::Numpad1,
            KeyCode::Numpad2 => Key::Numpad2,
            KeyCode::Numpad3 => Key::Numpad3,
            KeyCode::Numpad4 => Key::Numpad4,
            KeyCode::Numpad5 => Key::Numpad5,
            KeyCode::Numpad6 => Key::Numpad6,
            KeyCode::Numpad7 => Key::Numpad7,
            KeyCode::Numpad8 => Key::Numpad8,
            KeyCode::Numpad9 => Key::Numpad9,
            KeyCode::NumpadAdd => Key::Add,
            KeyCode::Quote => Key::Apostrophe,
            KeyCode::Backslash => Key::Backslash,
            KeyCode::NumpadClear => Key::NumpadEquals,
            KeyCode::Comma => Key::Comma,
            KeyCode::Convert => Key::Convert,
            KeyCode::NumpadDecimal => Key::Decimal,
            KeyCode::NumpadDivide => Key::Divide,
            KeyCode::NumpadMultiply => Key::Multiply,
            KeyCode::Equal => Key::Equals,
            KeyCode::Backquote => Key::Grave,
            KeyCode::KanaMode => Key::Kana,
            KeyCode::AltLeft => Key::LAlt,
            KeyCode::BracketLeft => Key::LBracket,
            KeyCode::ControlLeft => Key::LControl,
            KeyCode::ShiftLeft => Key::LShift,
            KeyCode::SuperLeft => Key::LWin,
            KeyCode::LaunchMail => Key::Mail,
            KeyCode::MediaSelect => Key::MediaSelect,
            KeyCode::MediaStop => Key::MediaStop,
            KeyCode::Minus => Key::Minus,
            KeyCode::AudioVolumeMute => Key::Mute,
            KeyCode::BrowserForward => Key::NavigateForward,
            KeyCode::BrowserBack => Key::NavigateBackward,
            KeyCode::MediaTrackNext => Key::NextTrack,
            KeyCode::NonConvert => Key::NoConvert,
            KeyCode::NumpadComma => Key::NumpadComma,
            KeyCode::NumpadEnter => Key::NumpadEnter,
            KeyCode::IntlBackslash => Key::OEM102,
            KeyCode::Period => Key::Period,
            KeyCode::MediaPlayPause => Key::PlayPause,
            KeyCode::Power => Key::Power,
            KeyCode::MediaTrackPrevious => Key::PrevTrack,
            KeyCode::AltRight => Key::RAlt,
            KeyCode::BracketRight => Key::RBracket,
            KeyCode::ControlRight => Key::RControl,
            KeyCode::ShiftRight => Key::RShift,
            KeyCode::SuperRight => Key::RWin,
            KeyCode::Semicolon => Key::Semicolon,
            KeyCode::Slash => Key::Slash,
            KeyCode::Sleep => Key::Sleep,
            KeyCode::NumpadSubtract => Key::Subtract,
            KeyCode::Tab => Key::Tab,
            KeyCode::AudioVolumeDown => Key::VolumeDown,
            KeyCode::AudioVolumeUp => Key::VolumeUp,
            KeyCode::WakeUp => Key::Wake,
            KeyCode::BrowserHome => Key::WebHome,
            KeyCode::BrowserRefresh => Key::WebRefresh,
            KeyCode::BrowserSearch => Key::WebSearch,
            KeyCode::IntlYen => Key::Yen,
            KeyCode::Copy => Key::Copy,
            KeyCode::Paste => Key::Paste,
            KeyCode::Cut => Key::Cut,
            _ => Key::Unknown,
        }
    } else {
        Key::Unknown
    }
}

#[cfg(target_arch = "wasm32")]
fn translate_web_mouse_button(button: i16) -> MouseButton {
    match button {
        0 => MouseButton::Button1, // Left
        1 => MouseButton::Button3, // Middle
        2 => MouseButton::Button2, // Right
        3 => MouseButton::Button4,
        4 => MouseButton::Button5,
        _ => MouseButton::Button1,
    }
}

#[cfg(target_arch = "wasm32")]
fn translate_web_key(code: &str) -> Key {
    match code {
        "Digit1" => Key::Key1,
        "Digit2" => Key::Key2,
        "Digit3" => Key::Key3,
        "Digit4" => Key::Key4,
        "Digit5" => Key::Key5,
        "Digit6" => Key::Key6,
        "Digit7" => Key::Key7,
        "Digit8" => Key::Key8,
        "Digit9" => Key::Key9,
        "Digit0" => Key::Key0,
        "KeyA" => Key::A,
        "KeyB" => Key::B,
        "KeyC" => Key::C,
        "KeyD" => Key::D,
        "KeyE" => Key::E,
        "KeyF" => Key::F,
        "KeyG" => Key::G,
        "KeyH" => Key::H,
        "KeyI" => Key::I,
        "KeyJ" => Key::J,
        "KeyK" => Key::K,
        "KeyL" => Key::L,
        "KeyM" => Key::M,
        "KeyN" => Key::N,
        "KeyO" => Key::O,
        "KeyP" => Key::P,
        "KeyQ" => Key::Q,
        "KeyR" => Key::R,
        "KeyS" => Key::S,
        "KeyT" => Key::T,
        "KeyU" => Key::U,
        "KeyV" => Key::V,
        "KeyW" => Key::W,
        "KeyX" => Key::X,
        "KeyY" => Key::Y,
        "KeyZ" => Key::Z,
        "Escape" => Key::Escape,
        "F1" => Key::F1,
        "F2" => Key::F2,
        "F3" => Key::F3,
        "F4" => Key::F4,
        "F5" => Key::F5,
        "F6" => Key::F6,
        "F7" => Key::F7,
        "F8" => Key::F8,
        "F9" => Key::F9,
        "F10" => Key::F10,
        "F11" => Key::F11,
        "F12" => Key::F12,
        "Insert" => Key::Insert,
        "Home" => Key::Home,
        "Delete" => Key::Delete,
        "End" => Key::End,
        "PageDown" => Key::PageDown,
        "PageUp" => Key::PageUp,
        "ArrowLeft" => Key::Left,
        "ArrowUp" => Key::Up,
        "ArrowRight" => Key::Right,
        "ArrowDown" => Key::Down,
        "Backspace" => Key::Back,
        "Enter" => Key::Return,
        "Space" => Key::Space,
        "NumLock" => Key::Numlock,
        "Numpad0" => Key::Numpad0,
        "Numpad1" => Key::Numpad1,
        "Numpad2" => Key::Numpad2,
        "Numpad3" => Key::Numpad3,
        "Numpad4" => Key::Numpad4,
        "Numpad5" => Key::Numpad5,
        "Numpad6" => Key::Numpad6,
        "Numpad7" => Key::Numpad7,
        "Numpad8" => Key::Numpad8,
        "Numpad9" => Key::Numpad9,
        "NumpadAdd" => Key::Add,
        "NumpadSubtract" => Key::Subtract,
        "NumpadMultiply" => Key::Multiply,
        "NumpadDivide" => Key::Divide,
        "NumpadDecimal" => Key::Decimal,
        "NumpadEnter" => Key::NumpadEnter,
        "Quote" => Key::Apostrophe,
        "Backslash" => Key::Backslash,
        "Comma" => Key::Comma,
        "Equal" => Key::Equals,
        "Backquote" => Key::Grave,
        "AltLeft" => Key::LAlt,
        "BracketLeft" => Key::LBracket,
        "ControlLeft" => Key::LControl,
        "ShiftLeft" => Key::LShift,
        "MetaLeft" => Key::LWin,
        "Minus" => Key::Minus,
        "Period" => Key::Period,
        "AltRight" => Key::RAlt,
        "BracketRight" => Key::RBracket,
        "ControlRight" => Key::RControl,
        "ShiftRight" => Key::RShift,
        "MetaRight" => Key::RWin,
        "Semicolon" => Key::Semicolon,
        "Slash" => Key::Slash,
        "Tab" => Key::Tab,
        _ => Key::Unknown,
    }
}
