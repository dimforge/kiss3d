//! Egui integration for immediate mode UI.

use egui::RawInput;

use crate::event::{Action, Key, WindowEvent};
use crate::renderer::EguiRenderer;

use super::Window;

pub(crate) struct EguiContext {
    pub(crate) renderer: EguiRenderer,
    pub(crate) raw_input: RawInput,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) start_time: std::time::Instant,
}

impl EguiContext {
    pub(crate) fn new() -> Self {
        Self {
            renderer: EguiRenderer::new(),
            raw_input: RawInput::default(),
            #[cfg(not(target_arch = "wasm32"))]
            start_time: std::time::Instant::now(),
        }
    }
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
        let scale_factor = self.scale_factor() as f32;

        match *event {
            WindowEvent::CursorPos(x, y, _) => {
                // Convert physical pixels to logical coordinates
                let pos = egui::Pos2::new((x as f32) / scale_factor, (y as f32) / scale_factor);
                self.egui_context
                    .raw_input
                    .events
                    .push(egui::Event::PointerMoved(pos));
            }
            WindowEvent::MouseButton(button, action, _) => {
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
                            modifiers: self.get_egui_modifiers(),
                        });
                }
            }
            WindowEvent::Scroll(x, y, _) => {
                // Use Point unit since kiss3d's scroll values are already scaled
                // (native multiplies LineDelta by 10, WASM applies various scales)
                self.egui_context
                    .raw_input
                    .events
                    .push(egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        delta: egui::Vec2::new(x as f32, y as f32),
                        modifiers: self.get_egui_modifiers(),
                    });
            }
            WindowEvent::Char(ch) => {
                if !ch.is_control() {
                    self.egui_context
                        .raw_input
                        .events
                        .push(egui::Event::Text(ch.to_string()));
                }
            }
            WindowEvent::Key(key, action, _modifiers) => {
                if let Some(egui_key) = self.translate_key_to_egui(key) {
                    self.egui_context.raw_input.events.push(egui::Event::Key {
                        key: egui_key,
                        physical_key: None,
                        pressed: action == Action::Press,
                        repeat: false,
                        modifiers: self.get_egui_modifiers(),
                    });
                }
            }
            _ => {}
        }
    }

    pub(crate) fn get_egui_modifiers(&self) -> egui::Modifiers {
        egui::Modifiers {
            alt: self.get_key(Key::LAlt) == Action::Press
                || self.get_key(Key::RAlt) == Action::Press,
            ctrl: self.get_key(Key::LControl) == Action::Press
                || self.get_key(Key::RControl) == Action::Press,
            shift: self.get_key(Key::LShift) == Action::Press
                || self.get_key(Key::RShift) == Action::Press,
            mac_cmd: false,
            command: self.get_key(Key::LControl) == Action::Press
                || self.get_key(Key::RControl) == Action::Press,
        }
    }

    pub(crate) fn translate_key_to_egui(&self, key: Key) -> Option<egui::Key> {
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
            Key::Escape => egui::Key::Escape,
            Key::Tab => egui::Key::Tab,
            Key::Back => egui::Key::Backspace,
            Key::Return => egui::Key::Enter,
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
            _ => return None,
        })
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
    pub fn draw_ui<F>(&mut self, ui_fn: F)
    where
        F: FnOnce(&egui::Context),
    {
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

        let scale_factor = self.canvas.scale_factor() as f32;

        // Set pixels_per_point on the context to match our DPI scale
        self.egui_context
            .renderer
            .context()
            .set_pixels_per_point(scale_factor);

        // Build raw input with accumulated events
        let mut raw_input = std::mem::take(&mut self.egui_context.raw_input);
        raw_input.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(
                self.width() as f32 / scale_factor,
                self.height() as f32 / scale_factor,
            ),
        ));
        raw_input.time = time;
        raw_input.predicted_dt = 1.0 / 60.0;

        self.egui_context.renderer.begin_frame(raw_input);
        ui_fn(self.egui_context.renderer.context());
        self.egui_context.renderer.end_frame();

        // Reset raw_input for next frame (but keep it properly initialized)
        self.egui_context.raw_input = RawInput::default();
    }
}
