# DRM Feature Refactoring Plan

## Goal

Eliminate `unsafe` transmutation, remove `DrmCanvasWrapper`, fix broken 2D camera
support under DRM, and minimize `#[cfg(feature = "drm")]` scatter — all without
using traits, by replacing `&Canvas` in camera interfaces with a lightweight
`CanvasInputState` value type.

---

## Root Cause

`Camera2d::handle_event` / `Camera3d::handle_event` (and `update`, `start_pass`,
`render_complete`) accept `&Canvas` — a concrete type backed by `WgpuCanvas`
(winit). In DRM mode the window holds a `DrmCanvas` instead. Because neither type
shares a common concrete base, `DrmCanvasWrapper` was introduced to impersonate
`Canvas` via:

```rust
let canvas_ref: &Canvas = unsafe { std::mem::transmute(&wrapper) };
```

This is instant UB if either struct gains or reorders any field. It is also the
reason 2D cameras are skipped in DRM mode (the `#[cfg(not(feature = "drm"))]`
guard in `rendering.rs` around `camera_2d.handle_event` / `camera_2d.update`).

Cameras only read **four things** from `Canvas`:

| Method | Used by |
|---|---|
| `scale_factor() -> f64` | `FixedView2d`, `PanZoomCamera2d` |
| `get_mouse_button(button) -> Action` | `OrbitCamera3d`, `FirstPersonCamera3d`, `PanZoomCamera2d` |
| `get_key(key) -> Action` | `FirstPersonCamera3dStereo` |
| `size() -> (u32, u32)` | logically related, no current camera uses it directly |

No rendering state is needed at all.

---

## Transmute / UB Locations to Fix

| File | Line(s) | Description |
|---|---|---|
| `src/window/drm/drm_window.rs` | ~283 | `transmute` in `handle_event` |
| `src/window/rendering.rs` | ~265–270 | `transmute` in `render_single_frame` |

---

## Step-by-Step Plan

### Step 1 — Add `CanvasInputState` in `src/window/canvas.rs`

Add a new `Copy`/`Clone` struct alongside `NumSamples` and `CanvasSetup`.

```rust
use crate::event::{Action, Key, MouseButton};

const NUM_KEYS:    usize = 512;
const NUM_BUTTONS: usize = 8;

/// Lightweight snapshot of canvas input state passed to cameras.
///
/// Cameras receive this instead of a `&Canvas` so that headless back-ends
/// (DRM, offscreen) can satisfy the same interface without wrapping or
/// transmuting unrelated types.
#[derive(Clone, Debug)]
pub struct CanvasInputState {
    pub scale_factor: f64,
    pub size: (u32, u32),
    key_states:    Vec<Action>,   // len == NUM_KEYS
    button_states: Vec<Action>,   // len == NUM_BUTTONS
}

impl CanvasInputState {
    /// Build from explicit values; used by both `Canvas` and `DrmCanvas`.
    pub fn new(
        scale_factor: f64,
        size: (u32, u32),
        key_states: Vec<Action>,
        button_states: Vec<Action>,
    ) -> Self {
        Self { scale_factor, size, key_states, button_states }
    }

    /// Headless / no-input variant (DRM default, offscreen, tests).
    pub fn headless(size: (u32, u32)) -> Self {
        Self {
            scale_factor: 1.0,
            size,
            key_states:    vec![Action::Release; NUM_KEYS],
            button_states: vec![Action::Release; NUM_BUTTONS],
        }
    }

    #[inline]
    pub fn get_key(&self, key: Key) -> Action {
        self.key_states.get(key as usize).copied().unwrap_or(Action::Release)
    }

    #[inline]
    pub fn get_mouse_button(&self, button: MouseButton) -> Action {
        self.button_states.get(button as usize).copied().unwrap_or(Action::Release)
    }
}
```

Re-export `CanvasInputState` from `src/window/mod.rs` alongside `Canvas`.

---

### Step 2 — Add `input_state()` on `Canvas` and `DrmCanvas`

#### `src/window/canvas.rs` — inside `impl Canvas`

```rust
pub fn input_state(&self) -> CanvasInputState {
    CanvasInputState::new(
        self.canvas.scale_factor(),
        self.canvas.size(),
        self.canvas.all_key_states(),
        self.canvas.all_button_states(),
    )
}
```

#### `src/window/wgpu_canvas.rs` — inside `impl WgpuCanvas`

Add two new getters that copy the internal state arrays:

```rust
pub fn all_key_states(&self) -> Vec<Action> {
    // self.key_states is already a Vec/array of Action indexed by Key as usize
    self.key_states.iter().copied().collect()
}

pub fn all_button_states(&self) -> Vec<Action> {
    self.button_states.iter().copied().collect()
}
```

#### `src/window/drm/drm_canvas.rs` — inside `impl DrmCanvas`

```rust
pub fn input_state(&self) -> CanvasInputState {
    // DRM is headless — no keyboard/mouse hardware input by default
    CanvasInputState::headless(self.size())
}
```

---

### Step 3 — Update the camera trait signatures

#### `src/camera/camera2d.rs`

```rust
// Before:
fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent);
fn update(&mut self, canvas: &Canvas);

// After:
fn handle_event(&mut self, input: &CanvasInputState, event: &WindowEvent);
fn update(&mut self, input: &CanvasInputState);
```

#### `src/camera/camera3d.rs`

```rust
// Before:
fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent);
fn update(&mut self, canvas: &Canvas);
fn start_pass(&self, _pass: usize, _canvas: &Canvas) {}
fn render_complete(&self, _canvas: &Canvas) {}

// After:
fn handle_event(&mut self, input: &CanvasInputState, event: &WindowEvent);
fn update(&mut self, input: &CanvasInputState);
fn start_pass(&self, _pass: usize, _input: &CanvasInputState) {}
fn render_complete(&self, _input: &CanvasInputState) {}
```

#### All camera implementations (mechanical rename)

For every file in `src/camera/`:

| Old call | New call |
|---|---|
| `canvas.get_key(k)` | `input.get_key(k)` |
| `canvas.get_mouse_button(b)` | `input.get_mouse_button(b)` |
| `canvas.scale_factor()` | `input.scale_factor` |
| `canvas.size()` | `input.size` |
| parameter `canvas: &Canvas` | parameter `input: &CanvasInputState` |

Files to update:
- `src/camera/fixed_view2d.rs`
- `src/camera/sidescroll2d.rs`
- `src/camera/orbit3d.rs`
- `src/camera/first_person3d.rs`
- `src/camera/first_person_stereo3d.rs`
- `src/camera/fixed_view3d.rs`

---

### Step 4 — Fix `src/window/rendering.rs`

Replace the transmute / conditional block:

```rust
// BEFORE — unsafe and skips 2D cameras under DRM:
#[cfg(feature = "drm")]
let canvas_wrapper = crate::window::drm::DrmCanvasWrapper::new(&self.canvas);
#[cfg(feature = "drm")]
let canvas_ref: &Canvas = unsafe { std::mem::transmute(&canvas_wrapper) };
#[cfg(not(feature = "drm"))]
let canvas_ref = &self.canvas;

#[cfg(not(feature = "drm"))]
{
    // todo make it work for drm
    camera_2d.handle_event(canvas_ref, &WindowEvent::FramebufferSize(w, h));
    camera_2d.update(canvas_ref);
}
camera.handle_event(canvas_ref, &WindowEvent::FramebufferSize(w, h));
camera.update(canvas_ref);
```

```rust
// AFTER — safe, no cfg guards, 2D cameras work under DRM:
let input = self.canvas.input_state();

camera_2d.handle_event(&input, &WindowEvent::FramebufferSize(w, h));
camera_2d.update(&input);
camera.handle_event(&input, &WindowEvent::FramebufferSize(w, h));
camera.update(&input);
```

Also fix the `render_frame_3d` signature — it currently takes `canvas: &Canvas` to
forward to `camera.start_pass` / `camera.render_complete`. Replace with
`input: &CanvasInputState`.

Unify the `present()` divergence at the bottom of `render_single_frame`:

```rust
// BEFORE:
#[cfg(not(feature = "drm"))]
self.canvas.present(frame);
#[cfg(feature = "drm")]
let _ = self.canvas.present(frame);

// AFTER (make DrmCanvas::present return () internally, logging errors):
self.canvas.present(frame);
```

---

### Step 5 — Fix `src/window/drm/drm_window.rs` `handle_event`

```rust
// BEFORE — transmute:
let wrapper = super::DrmCanvasWrapper::new(&self.canvas);
let canvas_ref: &crate::window::Canvas = unsafe { std::mem::transmute(&wrapper) };
camera.handle_event(canvas_ref, event);
camera_2d.handle_event(canvas_ref, event);

// AFTER — safe:
let input = self.canvas.input_state();
camera.handle_event(&input, event);
camera_2d.handle_event(&input, event);
```

---

### Step 6 — Delete `src/window/drm/drm_canvas_wrapper.rs`

With cameras taking `CanvasInputState`, `DrmCanvasWrapper` has no remaining
purpose. Delete the file and remove references from `drm/mod.rs`:

```rust
// Remove from src/window/drm/mod.rs:
mod drm_canvas_wrapper;
pub use drm_canvas_wrapper::DrmCanvasWrapper;
```

---

### Step 7 — (Optional) Introduce `WindowRenderState` to collapse dual-import boilerplate

Several files (`drawing.rs`, `screenshot.rs`, `rendering.rs`) begin with:

```rust
#[cfg(feature = "drm")]
use super::drm::Window;
#[cfg(not(feature = "drm"))]
use super::Window;
```

This pattern can be eliminated by extracting all shared rendering fields into a
single inner struct and writing the `impl` blocks against that struct instead:

```rust
// src/window/shared.rs  (new file)
pub(super) struct WindowRenderState {
    pub ambient_intensity: f32,
    pub background: Color,
    pub polyline_renderer_2d: PolylineRenderer2d,
    pub point_renderer_2d:    PointRenderer2d,
    pub point_renderer:       PointRenderer3d,
    pub polyline_renderer:    PolylineRenderer3d,
    pub text_renderer:        TextRenderer,
    pub framebuffer_manager:  FramebufferManager,
    pub post_process_render_target: RenderTarget,
    pub should_close: bool,
    #[cfg(feature = "egui")]
    pub egui_context: EguiContext,
    #[cfg(feature = "recording")]
    pub recording: Option<RecordingState>,
}
```

Both `Window` structs embed it as `pub(super) render: WindowRenderState`. The
`drawing.rs`, `screenshot.rs` and shared rendering helpers then accept
`&mut WindowRenderState` as a parameter, eliminating the dual `use` imports
entirely. The dual `use` lines in `mod.rs` remain, but nothing else needs them.

This step is lower priority than Steps 1–6; the earlier steps already fix all
unsafety and the 2D camera bug.

---

## File Change Summary

| File | Action | Reason |
|---|---|---|
| `src/window/canvas.rs` | Add `CanvasInputState` + `Canvas::input_state()` | Core new type |
| `src/window/wgpu_canvas.rs` | Add `all_key_states()` / `all_button_states()` | Feed `CanvasInputState` |
| `src/window/drm/drm_canvas.rs` | Add `DrmCanvas::input_state()` | Feed `CanvasInputState` |
| `src/window/drm/drm_canvas_wrapper.rs` | **Delete** | Replaced by `CanvasInputState` |
| `src/window/drm/mod.rs` | Remove `drm_canvas_wrapper` module and re-export | Cleanup |
| `src/camera/camera2d.rs` | `&Canvas` → `&CanvasInputState` in trait | Interface change |
| `src/camera/camera3d.rs` | `&Canvas` → `&CanvasInputState` in trait | Interface change |
| `src/camera/fixed_view2d.rs` | Mechanical rename | Implement new interface |
| `src/camera/sidescroll2d.rs` | Mechanical rename | Implement new interface |
| `src/camera/orbit3d.rs` | Mechanical rename | Implement new interface |
| `src/camera/first_person3d.rs` | Mechanical rename | Implement new interface |
| `src/camera/first_person_stereo3d.rs` | Mechanical rename | Implement new interface |
| `src/camera/fixed_view3d.rs` | Mechanical rename | Implement new interface |
| `src/window/rendering.rs` | Replace transmute block; remove 2D camera cfg guard; unify `present()` | Core fix |
| `src/window/drm/drm_window.rs` | Replace transmute in `handle_event`; use `input_state()` | Core fix |
| `src/window/mod.rs` | Re-export `CanvasInputState` | Public API |
| `src/window/shared.rs` | *(optional)* New `WindowRenderState` inner struct | Boilerplate reduction |

---

## Invariants Preserved

- Zero `unsafe` blocks in the final state (Steps 1–6 fully eliminate both transmutes).
- No traits added to `Canvas` or `DrmCanvas`.
- Conditional compilation (`#[cfg(feature = "drm")]`) remains only in:
  - `src/window/mod.rs` — module declaration and `Window` re-export.
  - `src/window/drm/drm_canvas.rs` — DRM-specific rendering logic.
  - `src/window/drm/drm_window.rs` — DRM-specific window creation.
  - Struct fields guarded by `egui` / `recording` features (unchanged).
- 2D cameras (`FixedView2d`, `PanZoomCamera2d`) receive correct framebuffer-size
  events and `update()` calls in DRM mode, fixing the previously broken behaviour.
- Public API for end users (`Window`, `Camera2d`, `Camera3d`) is source-compatible
  except for users who implemented the camera traits themselves — they need the
  mechanical parameter rename.