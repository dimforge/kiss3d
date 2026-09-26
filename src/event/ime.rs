//! Composed text from the platform's input method, delivered beside the
//! `Copy` event stream as dropped files are.

/// One report from the input method; nothing arrives until
/// [`Window::set_ime_allowed`](crate::window::Window::set_ime_allowed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImeEvent {
    Enabled,
    /// The text composed so far, replacing the last preedit; `cursor` is a
    /// byte range of it.
    Preedit {
        text: String,
        cursor: Option<(usize, usize)>,
    },
    /// Settled text, to be appended; the preedit is gone.
    Commit(String),
    Disabled,
}

impl ImeEvent {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn from_winit(ime: winit::event::Ime) -> Self {
        match ime {
            winit::event::Ime::Enabled => Self::Enabled,
            winit::event::Ime::Preedit(text, cursor) => Self::Preedit { text, cursor },
            winit::event::Ime::Commit(text) => Self::Commit(text),
            winit::event::Ime::Disabled => Self::Disabled,
        }
    }
}
