//! Egui integration for immediate mode UI.

use egui::RawInput;

use crate::event::{Action, Key, Modifiers, WindowEvent};
use crate::renderer::EguiRenderer;

use super::Window;

pub(crate) struct EguiContext {
    pub(crate) renderer: EguiRenderer,
    pub(crate) raw_input: RawInput,
    /// Whether an egui pass is currently open. A pass is opened lazily by the
    /// first `draw_ui` of the frame and closed by `finish_egui_pass` (called at
    /// render time). This lets several `draw_ui` / `draw_inspector` calls share a
    /// single pass instead of each starting its own (which would overwrite the
    /// previous one's shapes).
    pub(crate) pass_active: bool,
    /// The touch currently driving the egui pointer. Touch screens have no
    /// cursor, so the first finger down plays that role until it lifts;
    /// other fingers are ignored rather than fighting over the pointer.
    pub(crate) pointer_touch_id: Option<u64>,
    /// What a rerun of the open pass begins with: the pass's input less its
    /// events, as `Context::run` hands the later passes of a frame.
    pub(crate) rerun_input: RawInput,
    /// The shape the pointer was last set to. Setting one is a hop to the
    /// main thread, so only a change is worth making.
    pub(crate) cursor: egui::CursorIcon,
    /// The UI zoom the host asked for, in points per design pixel. Held here
    /// rather than read back from the context because `set_zoom_factor` only
    /// lands at the next `begin_pass`, so the context answers with the old
    /// one until then and every size computed from it would be a frame late.
    pub(crate) zoom: f32,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) start_time: std::time::Instant,
}

impl EguiContext {
    pub(crate) fn new() -> Self {
        Self {
            renderer: EguiRenderer::new(),
            raw_input: RawInput::default(),
            pass_active: false,
            pointer_touch_id: None,
            rerun_input: RawInput::default(),
            cursor: egui::CursorIcon::Default,
            zoom: 1.0,
            #[cfg(not(target_arch = "wasm32"))]
            start_time: std::time::Instant::now(),
        }
    }
}

/// Whether ⌘ (Super) is the platform's command modifier: on macOS, and on
/// any Apple platform reached through a browser. Everywhere else it is Ctrl.
fn command_is_super() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        super::wgpu_canvas::apple_platform()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        cfg!(target_os = "macos")
    }
}

/// egui's view of the modifiers an event arrived with. Read from the event
/// rather than from the key states: those are polled once a frame, so a chord
/// tapped within one frame has already released its modifier by then.
fn egui_modifiers(modifiers: Modifiers) -> egui::Modifiers {
    let ctrl = modifiers.contains(Modifiers::Control);
    // Only where the platform's command key is ⌘: egui's mac-only paths read
    // this, and a browser on an Apple platform counts.
    let mac_cmd = command_is_super() && modifiers.contains(Modifiers::Super);
    egui::Modifiers {
        alt: modifiers.contains(Modifiers::Alt),
        ctrl,
        shift: modifiers.contains(Modifiers::Shift),
        mac_cmd,
        // egui's own rule: a `COMMAND` chord is ⌘ where the platform has one
        // and Ctrl everywhere else, so a shortcut reads the way the rest of
        // the platform's apps read it.
        command: if command_is_super() { mac_cmd } else { ctrl },
    }
}

/// egui's key for one of ours. Keys arrive as physical codes on both
/// backends, so a shifted `\` is still `Backslash` and one arm covers every
/// layout. Anything egui has no key for is dropped.
fn translate_key_to_egui(key: Key) -> Option<egui::Key> {
    Some(match key {
        Key::A => egui::Key::A,
        Key::B => egui::Key::B,
        Key::C => egui::Key::C,
        Key::D => egui::Key::D,
        Key::E => egui::Key::E,
        Key::F => egui::Key::F,
        Key::G => egui::Key::G,
        Key::H => egui::Key::H,
        Key::I => egui::Key::I,
        Key::J => egui::Key::J,
        Key::K => egui::Key::K,
        Key::L => egui::Key::L,
        Key::M => egui::Key::M,
        Key::N => egui::Key::N,
        Key::O => egui::Key::O,
        Key::P => egui::Key::P,
        Key::Q => egui::Key::Q,
        Key::R => egui::Key::R,
        Key::S => egui::Key::S,
        Key::T => egui::Key::T,
        Key::U => egui::Key::U,
        Key::V => egui::Key::V,
        Key::W => egui::Key::W,
        Key::X => egui::Key::X,
        Key::Y => egui::Key::Y,
        Key::Z => egui::Key::Z,
        Key::Key0 | Key::Numpad0 => egui::Key::Num0,
        Key::Key1 | Key::Numpad1 => egui::Key::Num1,
        Key::Key2 | Key::Numpad2 => egui::Key::Num2,
        Key::Key3 | Key::Numpad3 => egui::Key::Num3,
        Key::Key4 | Key::Numpad4 => egui::Key::Num4,
        Key::Key5 | Key::Numpad5 => egui::Key::Num5,
        Key::Key6 | Key::Numpad6 => egui::Key::Num6,
        Key::Key7 | Key::Numpad7 => egui::Key::Num7,
        Key::Key8 | Key::Numpad8 => egui::Key::Num8,
        Key::Key9 | Key::Numpad9 => egui::Key::Num9,
        Key::F1 => egui::Key::F1,
        Key::F2 => egui::Key::F2,
        Key::F3 => egui::Key::F3,
        Key::F4 => egui::Key::F4,
        Key::F5 => egui::Key::F5,
        Key::F6 => egui::Key::F6,
        Key::F7 => egui::Key::F7,
        Key::F8 => egui::Key::F8,
        Key::F9 => egui::Key::F9,
        Key::F10 => egui::Key::F10,
        Key::F11 => egui::Key::F11,
        Key::F12 => egui::Key::F12,
        Key::F13 => egui::Key::F13,
        Key::F14 => egui::Key::F14,
        Key::F15 => egui::Key::F15,
        Key::F16 => egui::Key::F16,
        Key::F17 => egui::Key::F17,
        Key::F18 => egui::Key::F18,
        Key::F19 => egui::Key::F19,
        Key::F20 => egui::Key::F20,
        Key::F21 => egui::Key::F21,
        Key::F22 => egui::Key::F22,
        Key::F23 => egui::Key::F23,
        Key::F24 => egui::Key::F24,
        Key::Backslash => egui::Key::Backslash,
        Key::Slash | Key::Divide => egui::Key::Slash,
        Key::Comma | Key::NumpadComma => egui::Key::Comma,
        Key::Period | Key::Decimal => egui::Key::Period,
        Key::Minus | Key::Subtract => egui::Key::Minus,
        Key::Add => egui::Key::Plus,
        Key::Equals | Key::NumpadEquals => egui::Key::Equals,
        Key::Semicolon => egui::Key::Semicolon,
        Key::Colon => egui::Key::Colon,
        Key::Apostrophe => egui::Key::Quote,
        Key::Grave => egui::Key::Backtick,
        Key::LBracket => egui::Key::OpenBracket,
        Key::RBracket => egui::Key::CloseBracket,
        Key::Escape => egui::Key::Escape,
        Key::Tab => egui::Key::Tab,
        Key::Back => egui::Key::Backspace,
        Key::Return | Key::NumpadEnter => egui::Key::Enter,
        Key::Space => egui::Key::Space,
        Key::Insert => egui::Key::Insert,
        Key::Delete => egui::Key::Delete,
        Key::Home => egui::Key::Home,
        Key::End => egui::Key::End,
        Key::PageUp => egui::Key::PageUp,
        Key::PageDown => egui::Key::PageDown,
        Key::Left => egui::Key::ArrowLeft,
        Key::Up => egui::Key::ArrowUp,
        Key::Right => egui::Key::ArrowRight,
        Key::Down => egui::Key::ArrowDown,
        Key::Copy => egui::Key::Copy,
        Key::Cut => egui::Key::Cut,
        Key::Paste => egui::Key::Paste,
        Key::NavigateBackward => egui::Key::BrowserBack,
        _ => return None,
    })
}

impl Window {
    /// Retrieves a mutable reference to the egui context.
    ///
    /// Use this to access egui's full API for creating custom UI elements.
    ///
    /// # Returns
    /// A mutable reference to the egui Context
    ///
    /// # Note
    /// Only available when the `egui` feature is enabled.
    pub fn egui_context_mut(&mut self) -> &mut egui::Context {
        self.egui_context.renderer.context_mut()
    }

    /// Retrieves a reference to the egui context.
    ///
    /// Use this to access egui's API for reading UI state.
    ///
    /// # Returns
    /// A reference to the egui Context
    ///
    /// # Note
    /// Only available when the `egui` feature is enabled.
    pub fn egui_context(&self) -> &egui::Context {
        self.egui_context.renderer.context()
    }

    /// Physical pixels per egui point: the display's own scale factor times
    /// the UI zoom held by [`egui::Context::set_zoom_factor`]. Every position
    /// egui is fed and the screen it is told about are divided by this, which
    /// is what `egui-winit` does.
    ///
    /// A host that wants the UI larger sets the zoom factor and reads this to
    /// learn what a point became; it must not set `pixels_per_point` itself,
    /// since egui derives that from the zoom and the native scale below.
    pub fn egui_pixels_per_point(&self) -> f32 {
        self.egui_context.zoom * self.scale_factor() as f32
    }

    /// Draw the UI this many points per design pixel, on top of whatever the
    /// display's own scale is. Applied to egui's zoom factor at the start of
    /// the next pass, which is what makes every widget and every font grow
    /// together, egui's own controls included.
    pub fn set_ui_zoom(&mut self, zoom: f32) {
        self.egui_context.zoom = zoom.max(f32::EPSILON);
    }

    /// What [`Window::set_ui_zoom`] was last given.
    pub fn ui_zoom(&self) -> f32 {
        self.egui_context.zoom
    }

    /// Registers a native wgpu texture view with this window's egui renderer,
    /// returning a [`egui::TextureId`] that can be drawn with
    /// `ui.image((id, size))` inside [`Window::draw_ui`] — entirely on the
    /// GPU, no read-back, so it works on the web too.
    ///
    /// Typical use: display the live output of an
    /// [`OffscreenSurface`](crate::window::OffscreenSurface) (its
    /// [`output_view`](crate::window::OffscreenSurface::output_view)) as a
    /// picture-in-picture panel.
    ///
    /// The id stays valid until [`Window::unregister_egui_texture`]. If the
    /// underlying texture is reallocated (e.g. the surface is resized),
    /// re-register the new view.
    pub fn register_egui_texture(
        &mut self,
        view: &wgpu::TextureView,
        filter: wgpu::FilterMode,
    ) -> egui::TextureId {
        self.egui_context
            .renderer
            .register_native_texture(view, filter)
    }

    /// Frees a texture id previously returned by
    /// [`Window::register_egui_texture`].
    pub fn unregister_egui_texture(&mut self, id: egui::TextureId) {
        self.egui_context.renderer.unregister_native_texture(id)
    }

    /// Checks if egui is currently capturing mouse input.
    ///
    /// Returns `true` if the mouse is hovering over or interacting with an egui widget.
    /// This is useful for preventing 3D camera controls from interfering with UI interaction.
    ///
    /// # Returns
    /// `true` if egui wants mouse input, `false` otherwise
    ///
    /// # Note
    /// Only available when the `egui` feature is enabled.
    pub fn is_egui_capturing_mouse(&self) -> bool {
        self.egui_context.renderer.wants_pointer_input()
    }

    /// Checks if egui is currently capturing keyboard input.
    ///
    /// Returns `true` if an egui text field or other widget has keyboard focus.
    /// This is useful for preventing keyboard shortcuts from triggering while typing in UI.
    ///
    /// # Returns
    /// `true` if egui wants keyboard input, `false` otherwise
    ///
    /// # Note
    /// Only available when the `egui` feature is enabled.
    pub fn is_egui_capturing_keyboard(&self) -> bool {
        self.egui_context.renderer.wants_keyboard_input()
    }

    /// Feed a window event to egui for processing.
    pub(crate) fn feed_egui_event(&mut self, event: &WindowEvent) {
        let scale_factor = self.egui_pixels_per_point();

        match *event {
            WindowEvent::CursorPos(x, y, _) => {
                // Convert physical pixels to logical coordinates
                let pos = egui::Pos2::new((x as f32) / scale_factor, (y as f32) / scale_factor);
                self.egui_context
                    .raw_input
                    .events
                    .push(egui::Event::PointerMoved(pos));
            }
            WindowEvent::MouseButton(button, action, modifiers) => {
                let button = match button {
                    crate::event::MouseButton::Button1 => egui::PointerButton::Primary,
                    crate::event::MouseButton::Button2 => egui::PointerButton::Secondary,
                    crate::event::MouseButton::Button3 => egui::PointerButton::Middle,
                    _ => return,
                };

                if let Some(pos) = self.cursor_pos() {
                    // Convert physical pixels to logical coordinates
                    let pos = egui::Pos2::new(
                        (pos.0 as f32) / scale_factor,
                        (pos.1 as f32) / scale_factor,
                    );
                    let pressed = action == Action::Press;

                    self.egui_context
                        .raw_input
                        .events
                        .push(egui::Event::PointerButton {
                            pos,
                            button,
                            pressed,
                            modifiers: egui_modifiers(modifiers),
                        });
                }
            }
            WindowEvent::Scroll(x, y, modifiers) => {
                // Use Point unit since kiss3d's scroll values are already scaled
                // (native multiplies LineDelta by 10, WASM applies various scales)
                self.egui_context
                    .raw_input
                    .events
                    .push(egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        delta: egui::Vec2::new(x as f32, y as f32),
                        phase: egui::TouchPhase::Move,
                        modifiers: egui_modifiers(modifiers),
                    });
            }
            WindowEvent::Touch(id, x, y, action, _) => {
                use crate::event::TouchAction;

                let pos = egui::Pos2::new((x as f32) / scale_factor, (y as f32) / scale_factor);
                let events = &mut self.egui_context.raw_input.events;

                // Every touch is reported as a raw touch first: egui derives
                // pinch-zoom and rotate from the full set (MultiTouchInfo), the
                // same way egui-winit feeds it.
                events.push(egui::Event::Touch {
                    device_id: egui::TouchDeviceId(0),
                    id: egui::TouchId(id),
                    phase: match action {
                        TouchAction::Start => egui::TouchPhase::Start,
                        TouchAction::Move => egui::TouchPhase::Move,
                        TouchAction::End => egui::TouchPhase::End,
                        TouchAction::Cancel => egui::TouchPhase::Cancel,
                    },
                    pos,
                    force: None,
                });

                // The first finger down additionally becomes the egui pointer:
                // moved-then-pressed on Start, released-then-gone on End, so
                // widgets see the same sequence a mouse would produce.
                match action {
                    TouchAction::Start => {
                        if self.egui_context.pointer_touch_id.is_none() {
                            self.egui_context.pointer_touch_id = Some(id);
                            events.push(egui::Event::PointerMoved(pos));
                            events.push(egui::Event::PointerButton {
                                pos,
                                button: egui::PointerButton::Primary,
                                pressed: true,
                                modifiers: egui::Modifiers::default(),
                            });
                        }
                    }
                    TouchAction::Move => {
                        if self.egui_context.pointer_touch_id == Some(id) {
                            events.push(egui::Event::PointerMoved(pos));
                        }
                    }
                    TouchAction::End => {
                        if self.egui_context.pointer_touch_id == Some(id) {
                            self.egui_context.pointer_touch_id = None;
                            events.push(egui::Event::PointerButton {
                                pos,
                                button: egui::PointerButton::Primary,
                                pressed: false,
                                modifiers: egui::Modifiers::default(),
                            });
                            events.push(egui::Event::PointerGone);
                        }
                    }
                    TouchAction::Cancel => {
                        if self.egui_context.pointer_touch_id == Some(id) {
                            self.egui_context.pointer_touch_id = None;
                            events.push(egui::Event::PointerGone);
                        }
                    }
                }
            }
            WindowEvent::Char(ch) if !ch.is_control() => {
                self.egui_context
                    .raw_input
                    .events
                    .push(egui::Event::Text(ch.to_string()));
            }
            WindowEvent::Key(key, action, modifiers) => {
                if let Some(egui_key) = translate_key_to_egui(key) {
                    self.egui_context.raw_input.events.push(egui::Event::Key {
                        key: egui_key,
                        physical_key: None,
                        pressed: action == Action::Press,
                        repeat: false,
                        modifiers: egui_modifiers(modifiers),
                    });
                }
            }
            _ => {}
        }
    }

    /// Whether the UI built by one frame's [`Self::draw_ui`] calls is drawn
    /// again on the frames that make none. Off by default.
    ///
    /// Turning it on lets an app run its widgets only when something changed
    /// and keep the UI on screen in between. An app that shows its UI
    /// conditionally (`if show_hud { window.draw_ui(..) }`) wants it off, or
    /// has to call [`Self::clear_ui`] when it hides the UI.
    pub fn set_ui_retained(&mut self, retained: bool) {
        self.egui_context.renderer.set_retain_shapes(retained);
    }

    /// Whether the last frame's UI is redrawn on a frame that builds none.
    pub fn ui_retained(&self) -> bool {
        self.egui_context.renderer.retains_shapes()
    }

    /// Drops the UI built by the last [`Self::draw_ui`], so a retaining app
    /// stops drawing it. A no-op unless [`Self::set_ui_retained`] is on.
    pub fn clear_ui(&mut self) {
        self.egui_context.renderer.clear_shapes();
    }

    /// Draws an immediate mode UI using egui.
    ///
    /// Call this method from your render loop to create and display UI elements.
    /// The UI is drawn on top of the 3D scene.
    ///
    /// # Arguments
    /// * `ui_fn` - A closure that receives the egui Context and can create UI elements
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # use kiss3d::camera::OrbitCamera3d;
    /// # use kiss3d::scene::SceneNode3d;
    /// # #[cfg(feature = "egui")]
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// # let mut camera = OrbitCamera3d::default();
    /// # let mut scene = SceneNode3d::empty();
    /// while window.render_3d(&mut scene, &mut camera).await {
    ///     window.draw_ui(|ctx| {
    ///         egui::Window::new("My Window").show(ctx, |ui| {
    ///             ui.label("Hello, world!");
    ///             if ui.button("Click me").clicked() {
    ///                 println!("Button clicked!");
    ///             }
    ///         });
    ///     });
    /// }
    /// # }
    /// # #[cfg(not(feature = "egui"))]
    /// # fn main() {}
    /// ```
    ///
    /// # Note
    /// Only available when the `egui` feature is enabled.
    /// # Passes
    ///
    /// `ui_fn` may run more than once per frame. A pass that only measured
    /// something (a new `Area`, a `Grid`, a `Resize`) asks egui to discard it
    /// and run it again before the frame is shown, so a closure that mutates
    /// state has to tolerate being replayed. Only the closure that opened the
    /// pass is replayed: when a second `draw_ui` of the same frame is the one
    /// that asks, its measurement lands on the next frame instead.
    pub fn draw_ui<F>(&mut self, mut ui_fn: F)
    where
        F: FnMut(&egui::Context),
    {
        // Open the egui pass lazily so that several `draw_ui` (and
        // `draw_inspector`) calls in the same frame all run their widgets into
        // the *same* pass. The pass is closed at render time by
        // `finish_egui_pass`. Beginning a fresh pass per call would have the
        // second call's `end_frame` overwrite the first call's shapes (and the
        // `std::mem::take` below would starve it of input).
        let opened = !self.egui_context.pass_active;
        if opened {
            self.begin_egui_pass();
        }

        ui_fn(self.egui_context.renderer.context());

        // A pass that only learned a size (a new `Area`, which draws none of
        // itself while it is measured, a `Grid`, a `Resize`) asks to be
        // discarded and run again before the frame is shown. `Context::run`
        // does that for its callers; with the pass open here, this does, or
        // the frame shows the gap. Only the call that opened the pass reruns:
        // a rerun replays one closure, and an earlier call's shapes are gone.
        let max_passes = self
            .egui_context
            .renderer
            .context()
            .options(|options| options.max_passes.get());
        let mut passes = 1;
        while opened && passes < max_passes && self.egui_context.renderer.context().will_discard() {
            self.egui_context
                .renderer
                .rerun_frame(self.egui_context.rerun_input.clone());
            ui_fn(self.egui_context.renderer.context());
            passes += 1;
        }
    }

    /// Begins a new egui pass, feeding it the events accumulated since the last
    /// pass. Idempotent callers should guard on `pass_active`.
    fn begin_egui_pass(&mut self) {
        // Get time for animations - use egui context's own start time
        #[cfg(not(target_arch = "wasm32"))]
        let time = Some(self.egui_context.start_time.elapsed().as_secs_f64());
        #[cfg(target_arch = "wasm32")]
        let time = {
            use web_time::Instant;
            static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
            let start = START.get_or_init(Instant::now);
            Some(start.elapsed().as_secs_f64())
        };

        let native = self.canvas.scale_factor() as f32;
        let ppp = self.egui_pixels_per_point();
        // Before the pass opens: egui holds a new zoom until its next
        // `begin_pass`, so setting it here is what makes this pass use it.
        self.egui_context
            .renderer
            .context()
            .set_zoom_factor(self.egui_context.zoom);

        // Build raw input with accumulated events
        let mut raw_input = std::mem::take(&mut self.egui_context.raw_input);
        // The display's own scale, which egui multiplies by the zoom factor to
        // reach `pixels_per_point`. Setting that directly instead would pin the
        // zoom at 1, since `set_pixels_per_point` is a zoom setter in disguise.
        let viewport = raw_input.viewport_id;
        raw_input
            .viewports
            .entry(viewport)
            .or_default()
            .native_pixels_per_point = Some(native);
        raw_input.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(self.width() as f32 / ppp, self.height() as f32 / ppp),
        ));
        raw_input.time = time;
        raw_input.predicted_dt = 1.0 / 60.0;

        // `take` leaves what `Context::run` hands a later pass of the same
        // frame: the time, and none of the events.
        let first = raw_input.take();
        self.egui_context.rerun_input = raw_input;
        self.egui_context.renderer.begin_frame(first);
        self.egui_context.pass_active = true;
    }

    /// Closes the egui pass opened by `draw_ui`/`draw_inspector`, if any, so the
    /// accumulated shapes are ready to be painted by the egui renderer. Called
    /// once per frame from the render path. No-op when no UI was drawn.
    pub(crate) fn finish_egui_pass(&mut self) {
        if self.egui_context.pass_active {
            self.egui_context.renderer.end_frame();
            self.egui_context.pass_active = false;
            // What the pass asked for, applied where the window is: egui says
            // which shape a widget wants and only the host can set one.
            let want = self.egui_context.renderer.cursor();
            if want != self.egui_context.cursor {
                self.egui_context.cursor = want;
                self.canvas.set_cursor_icon(winit_cursor(want));
            }
        }
        // Note: `raw_input` is *not* reset here. It is drained by
        // `begin_egui_pass` (via `std::mem::take`) when the next pass opens, and
        // events fed by `handle_events` between this point and that next pass
        // must be preserved — resetting here would discard them and the UI would
        // stop responding to input.
    }
}

/// egui's cursor names as winit's. The two lists are the same set under
/// different names, and neither crate knows about the other.
fn winit_cursor(icon: egui::CursorIcon) -> winit::window::CursorIcon {
    use egui::CursorIcon as E;
    use winit::window::CursorIcon as W;
    match icon {
        E::Default => W::Default,
        // Hiding the pointer is `hide_cursor`, not a shape; nothing in a
        // widget's own paint should take it off the screen.
        E::None => W::Default,
        E::ContextMenu => W::ContextMenu,
        E::Help => W::Help,
        E::PointingHand => W::Pointer,
        E::Progress => W::Progress,
        E::Wait => W::Wait,
        E::Cell => W::Cell,
        E::Crosshair => W::Crosshair,
        E::Text => W::Text,
        E::VerticalText => W::VerticalText,
        E::Alias => W::Alias,
        E::Copy => W::Copy,
        E::Move => W::Move,
        E::NoDrop => W::NoDrop,
        E::NotAllowed => W::NotAllowed,
        E::Grab => W::Grab,
        E::Grabbing => W::Grabbing,
        E::AllScroll => W::AllScroll,
        E::ResizeHorizontal => W::EwResize,
        E::ResizeNeSw => W::NeswResize,
        E::ResizeNwSe => W::NwseResize,
        E::ResizeVertical => W::NsResize,
        E::ResizeEast => W::EResize,
        E::ResizeSouthEast => W::SeResize,
        E::ResizeSouth => W::SResize,
        E::ResizeSouthWest => W::SwResize,
        E::ResizeWest => W::WResize,
        E::ResizeNorthWest => W::NwResize,
        E::ResizeNorth => W::NResize,
        E::ResizeNorthEast => W::NeResize,
        E::ResizeColumn => W::ColResize,
        E::ResizeRow => W::RowResize,
        E::ZoomIn => W::ZoomIn,
        E::ZoomOut => W::ZoomOut,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chords the editor binds are punctuation, digits and function keys,
    /// none of which egui ever saw while the map held letters only.
    #[test]
    fn punctuation_digits_and_function_keys_reach_egui() {
        for (key, want) in [
            (Key::Backslash, egui::Key::Backslash),
            (Key::Slash, egui::Key::Slash),
            (Key::Comma, egui::Key::Comma),
            (Key::Equals, egui::Key::Equals),
            (Key::Minus, egui::Key::Minus),
            (Key::Key1, egui::Key::Num1),
            (Key::Numpad1, egui::Key::Num1),
            (Key::F5, egui::Key::F5),
            (Key::F11, egui::Key::F11),
            (Key::A, egui::Key::A),
        ] {
            assert_eq!(translate_key_to_egui(key), Some(want), "{:?}", key);
        }
        assert_eq!(translate_key_to_egui(Key::Unknown), None);
    }

    /// A `COMMAND` chord is the key the platform calls command, as egui has it:
    /// ⌘ where there is one, Ctrl everywhere else. The other key is not it.
    #[test]
    fn command_is_the_key_the_platform_calls_command() {
        let (command, other) = if command_is_super() {
            (Modifiers::Super, Modifiers::Control)
        } else {
            (Modifiers::Control, Modifiers::Super)
        };

        let held = egui_modifiers(command | Modifiers::Shift);
        assert!(held.command, "{:?} did not read as command", command);
        assert!(held.shift);
        assert!(
            held.matches_logically(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT),
            "{:?} did not match ⇧⌘",
            command
        );

        assert!(
            !egui_modifiers(other).command,
            "{:?} read as command on a platform that calls {:?} that",
            other,
            command
        );
    }

    /// ⌘ is a Mac's alone: egui's mac-only paths read `mac_cmd`, and a chord
    /// asking for no modifier still has to see none.
    #[test]
    fn mac_cmd_follows_the_platform_and_a_bare_key_stays_bare() {
        let sup = egui_modifiers(Modifiers::Super);
        assert_eq!(sup.mac_cmd, cfg!(target_os = "macos"));
        let none = egui_modifiers(Modifiers::empty());
        assert!(!none.command && !none.ctrl);
        assert!(none.matches_logically(egui::Modifiers::NONE));
        assert!(!egui_modifiers(Modifiers::Control).matches_logically(egui::Modifiers::NONE));
    }
}
