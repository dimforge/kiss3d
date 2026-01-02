//! Event handling functionality.

use crate::camera::Camera3d;
use crate::camera::Camera2d;
use crate::event::{Action, EventManager, Key, MouseButton, WindowEvent};

use super::Window;

impl Window {
    /// Returns an event manager for accessing window events.
    ///
    /// The event manager provides an iterator over events that occurred since the last frame,
    /// such as keyboard input, mouse movement, and window resizing.
    ///
    /// # Returns
    /// An `EventManager` that can be iterated to process events
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::prelude::*;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// # let mut camera = OrbitCamera3d::default();
    /// # let mut scene = SceneNode3d::empty();
    /// # while window.render_3d(&mut scene, &mut camera).await {
    /// for event in window.events().iter() {
    ///     match event.value {
    ///         WindowEvent::Key(Key::Escape, Action::Release, _) => {
    ///             println!("Escape pressed!");
    ///         }
    ///         _ => {}
    ///     }
    /// }
    /// # }
    /// # }
    /// ```
    pub fn events(&self) -> EventManager {
        EventManager::new(self.events.clone(), self.unhandled_events.clone())
    }

    /// Gets the current state of a keyboard key.
    ///
    /// # Arguments
    /// * `key` - The key to check
    ///
    /// # Returns
    /// The current `Action` state (e.g., `Action::Press`, `Action::Release`)
    pub fn get_key(&self, key: Key) -> Action {
        self.canvas.get_key(key)
    }

    /// Gets the current state of a mouse button.
    ///
    /// # Arguments
    /// * `button` - The mouse button to check
    ///
    /// # Returns
    /// The current `Action` state (e.g., `Action::Press`, `Action::Release`)
    pub fn get_mouse_button(&self, button: MouseButton) -> Action {
        self.canvas.get_mouse_button(button)
    }

    /// Gets the last known position of the mouse cursor.
    ///
    /// The position is automatically updated when the mouse moves over the window.
    /// Coordinates are in pixels, with (0, 0) at the top-left corner.
    ///
    /// # Returns
    /// `Some((x, y))` with the cursor position, or `None` if the cursor position is unknown
    pub fn cursor_pos(&self) -> Option<(f64, f64)> {
        self.canvas.cursor_pos()
    }

    #[inline]
    pub(crate) fn handle_events(&mut self, camera: &mut dyn Camera3d, camera_2d: &mut dyn Camera2d) {
        let unhandled_events = self.unhandled_events.clone(); // TODO: could we avoid the clone?
        let events = self.events.clone(); // TODO: could we avoid the clone?

        for event in unhandled_events.borrow().iter() {
            self.handle_event(camera, camera_2d, event)
        }

        for event in events.try_iter() {
            self.handle_event(camera, camera_2d, &event)
        }

        unhandled_events.borrow_mut().clear();
        self.canvas.poll_events();
    }

    pub(crate) fn handle_event(
        &mut self,
        camera: &mut dyn Camera3d,
        camera_2d: &mut dyn Camera2d,
        event: &WindowEvent,
    ) {
        match *event {
            WindowEvent::Key(Key::Escape, Action::Release, _) | WindowEvent::Close => {
                self.close();
            }
            _ => {}
        }

        // Feed events to egui and check if it wants to capture input
        #[cfg(feature = "egui")]
        {
            self.feed_egui_event(event);

            if event.is_keyboard_event() && self.is_egui_capturing_keyboard() {
                return;
            }

            if event.is_mouse_event() && self.is_egui_capturing_mouse() {
                return;
            }
        }

        camera.handle_event(&self.canvas, event);
        camera_2d.handle_event(&self.canvas, event);
    }
}
