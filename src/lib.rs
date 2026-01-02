/*!
# Kiss3d

Keep It Simple, Stupid 3d graphics engine.

This library is born from the frustration in front of the fact that today's 3D
graphics library are:

* either too low level: you have to write your own shaders and opening a
  window steals you 8 hours, 300 lines of code and 10L of coffee.
* or high level but too hard to understand/use: those are libraries made to
  write beautiful animations or games. They have a lot of feature; too much
  feature if you only want to draw a few geometries on the screen.

**kiss3d** is not designed to be feature-complete or fast.
It is designed to be able to draw simple geometric figures and play with them
with one-liners.

An on-line version of this documentation is available [here](http://kiss3d.org).

## Features
Most features are one-liners.

* WASM compatibility.
* open a window with a default arc-ball camera and a point light.
* a first-person camera is available too and user-defined cameras are possible.
* display boxes, spheres, cones, cylinders, quads and lines.
* change an object color or texture.
* change an object transform (we use the [glam](https://docs.rs/glam/) library
  for math operations).
* create basic post-processing effects.

As an example, having a red, rotating cube with the light attached to the camera is as simple as:

```no_run
use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: cube").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();

    let mut c = scene.add_cube(1.0, 1.0, 1.0);
    c.set_color(RED);

    let rot = Quat::from_axis_angle(Vec3::Y, 0.014);

    while window.render_3d(&mut scene, &mut camera).await {
        c.rotate(rot);
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

```text
[dependencies]
kiss3d = "0.36"
```

## Contributions
I'd love to see people improving this library for their own needs. However, keep in mind that
**kiss3d** is KISS. One-liner features (from the user point of view) are preferred.

## Acknowledgements

Thanks to all the Rustaceans for their help, and their OpenGL bindings.
*/
#![allow(non_upper_case_globals)]
#![allow(unused_unsafe)] // TODO: should be denied
#![allow(missing_copy_implementations)]
#![doc(html_root_url = "http://kiss3d.org/doc")]
#![allow(clippy::module_inception)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

#[macro_use]
extern crate bitflags;
extern crate num_traits as num;
extern crate rusttype;
#[cfg(feature = "serde")]
extern crate serde;

#[cfg(feature = "egui")]
pub extern crate egui;

pub use glamx;
#[cfg(feature = "parry")]
pub use parry3d;

// Re-export the procedural macro and its runtime dependencies
pub use kiss3d_macro::main;

#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use pollster;

#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
pub use wasm_bindgen_futures;

#[deprecated(note = "Use the `renderer` module instead.")]
pub use crate::renderer::point_renderer3d;

pub mod builtin;
pub mod camera;
pub mod color;
pub mod context;
pub mod event;
pub mod light;
pub mod loader;
pub mod post_processing;
pub mod procedural;
pub mod renderer;
pub mod resource;
pub mod scene;
pub mod text;
pub mod window;

pub mod prelude {
    pub use crate::builtin::*;
    pub use crate::camera::*;
    pub use crate::color::*;
    pub use crate::context::*;
    pub use crate::event::*;
    pub use crate::light::*;
    pub use crate::loader::*;
    pub use crate::renderer::*;
    pub use crate::resource::*;
    pub use crate::scene::*;
    pub use crate::text::*;
    pub use crate::window::*;
    pub use glamx::{
        Mat2, Mat3, Pose2, Pose3, Quat, Rot2, Rot3, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles,
    };
    pub use std::cell::RefCell;
    pub use std::rc::Rc;
}
