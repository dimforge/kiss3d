# Kiss3d

<p align="center">
  <img src="assets/kiss3d-logo.png" alt="crates.io" width="300px">
</p>

[![Crates.io](https://img.shields.io/crates/v/kiss3d.svg)](https://crates.io/crates/kiss3d)
[![Docs](https://docs.rs/kiss3d/badge.svg)](https://docs.rs/kiss3d)
[![License](https://img.shields.io/crates/l/kiss3d)](https://github.com/sebcrozet/kiss3d)

Keep It Simple, Stupid 3d graphics engine.

This library is born from the frustration that today’s 3D
graphics library are either:

* **Too low level**: you have to write your own shaders and opening a
  window takes 8 hours, 300 lines of code and 10L of coffee.
* High level, but **too hard to understand/use**: these libraries are made to
  create beautiful photoreal (or close to it) animations or games.
  They have many features; too many, in fact, if you just want to draw a few objects
  on the screen with as little friction as possible.

**kiss3d** is not designed to be feature-complete or fast.
It is designed to let you draw simple geometric figures and play with them
with as little friction as possible.

## Features

* WASM compatible.
* Out of the box, open a window with a default arc-ball camera and a point light.
* First-person camera available as well, and user-defined cameras are possible.
* Render boxes, spheres, cones, cylinders, quads and lines simply
* Change an object's color or texture.
* Change an object's transform (we use [nalgebra](http://nalgebra.org) to do that).
* Create basic post-processing effects.

As an example, creating a scene with a red, rotating cube with a light attached
to the camera is as simple as:

```rust
extern crate kiss3d;

use kiss3d::light::Light;
use kiss3d::window::Window;
use kiss3d::na::{UnitQuaternion, Vector3};

#[kiss3d::main]
async fn main() {
  let mut window = Window::new("Kiss3d: cube").await;
  let mut c = window.add_cube(1.0, 1.0, 1.0);

  c.set_color(1.0, 0.0, 0.0);

  window.set_light(Light::StickToCamera);

  let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);

  while window.render().await {
    c.prepend_to_local_rotation(&rot);
  }
}

```

This code works on **both native platforms and WASM** without any changes! The `#[kiss3d::main]`
macro and `async` rendering API handle the platform differences automatically:

* **On native**: The async runtime is managed with `pollster::block_on`
* **On WASM**: The async function integrates with the browser's event loop via `requestAnimationFrame`

This approach eliminates the need for platform-specific code or managing different entry points,
making it simple to write truly cross-platform 3D applications.

Some controls are handled by default by the engine (they can be overridden by the user):

* `scroll`: zoom in / zoom out.
* `left click + drag`: look around.
* `right click + drag`: translate the view point.
* `enter`: look at the origin (0.0, 0.0, 0.0).

## Compilation

You will need the last stable build of the [rust compiler](http://www.rust-lang.org)
and the official package manager: [cargo](https://github.com/rust-lang/cargo).

Simply add the following to your `Cargo.toml` file:

```toml
[dependencies]
kiss3d = "0.37"
```
Note: If your project already uses nalgebra, you'll need the same version used by `kiss3d`, or you may run into compatibility issues.

### Optional Features

#### Recording

Kiss3d supports recording your 3D scene to MP4 video files. This feature requires FFmpeg to be installed on your system.

**Install FFmpeg:**
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

**Enable the feature in your Cargo.toml:**
```toml
[dependencies]
kiss3d = { version = "0.37", features = ["recording"] }
```

**Example usage:**
```rust
use kiss3d::window::{Window, RecordingConfig};

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Recording Example").await;

    // Start recording (optionally skip frames to reduce file size)
    let config = RecordingConfig::new().with_frame_skip(2);
    window.begin_recording_with_config(config);

    // Render your scene
    for _ in 0..120 {
        window.render().await;
    }

    // Save to MP4 (30 fps)
    window.end_recording("output.mp4", 30).unwrap();
}
```

#### Egui Integration

For immediate mode GUI support, enable the `egui` feature:
```toml
[dependencies]
kiss3d = { version = "0.37", features = ["egui"] }
```


## Contributions
I’d love to see people improving this library for their own needs. However, keep in mind that
**kiss3d** is KISS. One-liner features (from the user point of view) are preferred.

## Acknowledgements

Thanks to all the Rustaceans for their help, and their OpenGL bindings.
