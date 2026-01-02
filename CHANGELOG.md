# v0.39.0

Major API overhaul: scene separation from window, glam math library, simplified transform API, and 2D/3D naming conventions.

## Breaking Changes

### Scene Separation from Window
- **Scenes are no longer owned by the Window** - Create scenes independently and pass them to render
- **New render methods**: `window.render_3d(&mut scene, &mut camera)` and `window.render_2d(&mut scene, &mut camera)`
- **Removed**: `window.add_cube()`, `window.add_sphere()`, etc. - use `scene.add_cube()` instead
- **Removed**: `window.scene()` and `window.scene_mut()` - manage scenes directly
- Cameras must now be created and managed by user code

### 3D Type Renaming
- `SceneNode` → `SceneNode3d`
- `Object` → `Object3d`
- `GpuMesh` → `GpuMesh3d`
- `MeshManager` → `MeshManager3d`
- `MaterialManager` → `MaterialManager3d`
- `PointRenderer` → `PointRenderer3d`
- `PolylineRenderer` → `PolylineRenderer3d`
- `Camera` trait → `Camera3d` trait

### Math Library: nalgebra → glam
- **Switched** from `nalgebra` to `glamx` (glam wrapper) for all public APIs
- Key type changes:
  - `Point3<f32>` → `Vec3`
  - `Vector3<f32>` → `Vec3`
  - `UnitQuaternion<f32>` → `Quat`
  - `Translation3<f32>` → `Vec3`
  - `Isometry3<f32>` → `Pose3`
  - `Point2<f32>` / `Vector2<f32>` → `Vec2`
  - `UnitComplex<f32>` → `f32` (just use angle in radians directly)
- Common conversions:
  - `Vector3::y_axis()` → `Vec3::Y`
  - `Point3::origin()` → `Vec3::ZERO`
  - `UnitQuaternion::from_axis_angle(&Vector3::y_axis(), angle)` → `Quat::from_axis_angle(Vec3::Y, angle)`

### Simplified Transform API
- `prepend_to_local_rotation(&rot)` → `rotate(rot)`
- `prepend_to_local_translation(&t)` → `translate(t)`
- `set_local_rotation(r)` → `set_rotation(r)`
- `set_local_translation(t)` → `set_position(t)`
- `local_rotation()` → `rotation()`
- `local_translation()` → `position()`
- Rotation values passed by value, not reference

### 2D API Renaming ("planar" → "2d")
- `PlanarSceneNode` → `SceneNode2d`
- `PlanarCamera` → `Camera2d`
- `FixedView` (planar) → `FixedView2d`
- `Sidescroll` → `PanZoomCamera2d`
- `draw_planar_line()` → `draw_line_2d()`
- `draw_planar_point()` → `draw_point_2d()`
- `add_rectangle()` / `add_circle()` now on `SceneNode2d`
- Modules renamed: `planar_camera` → `camera2d`, `planar_polyline_renderer` → `polyline_renderer2d`

### Camera Renaming
- `ArcBall` → `OrbitCamera3d`
- `FirstPerson` → `FirstPersonCamera3d`
- `FirstPersonStereo` → `FirstPersonStereoCamera3d`
- `FixedView` → `FixedViewCamera3d`

### Color API
- Colors now use `[f32; 3]` arrays instead of `Point3<f32>`
- `set_color(r, g, b)` now takes a `[f32; 3]` and you can use color constants directly: `set_color(RED)`
- `set_lines_color(Some(Point3::new(1.0, 0.0, 0.0)))` → `set_lines_color(Some(RED))`
- New `color` module with CSS named color constants (re-exported in prelude)

### parry3d Now Optional
- `parry3d` moved behind `parry` feature flag
- `parry3d` now uses glam directly (no nalgebra conversion needed)
- `add_trimesh()` requires `parry` feature
- Examples `procedural` and `decomp` require `--features parry`

### Lighting API
- **Replaced** `Light::Absolute(Point3)` and `Light::StickToCamera` enum with new `Light` struct
- **Removed** `window.set_light()` - use `scene.add_light()` instead
- Lights are now scene nodes, not window-level configuration

### Other Breaking Changes
- `SceneNode::new_empty()` → `SceneNode3d::empty()`
- `SceneNode::unlink()` → `SceneNode3d::detach()`
- Index buffers always use `u32` (removed `vertex_index_u32` feature)
- `instant` crate replaced with `web_time`
- Camera modules merged: `camera2d` and `camera3d` now under single `camera` module
- **Removed** `Window::set_frame_limit()` method
- `set_background_color(r, g, b)` → `set_background_color(Color)`
- `draw_line(a, b, color, width)` → `draw_line(a, b, color, width, perspective)`

## Migration Guide

### Basic 3D scene (before):
```rust
use kiss3d::window::Window;
use nalgebra::{UnitQuaternion, Vector3};

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Example").await;
    let mut cube = window.add_cube(1.0, 1.0, 1.0);
    cube.set_color(1.0, 0.0, 0.0);

    let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.01);
    while window.render().await {
        cube.prepend_to_local_rotation(&rot);
    }
}
```

### Basic 3D scene (after):
```rust
use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Example").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();

    let mut cube = scene.add_cube(1.0, 1.0, 1.0);
    cube.set_color(RED);

    let rot = Quat::from_axis_angle(Vec3::Y, 0.01);
    while window.render_3d(&mut scene, &mut camera).await {
        cube.rotate(rot);
    }
}
```

### Basic 2D scene (before):
```rust
use kiss3d::window::Window;
use nalgebra::UnitComplex;

let mut window = Window::new("2D").await;
let mut rect = window.add_rectangle(50.0, 100.0);
let rot = UnitComplex::new(0.01);
while window.render().await {
    rect.prepend_to_local_rotation(&rot);
}
```

### Basic 2D scene (after):
```rust
use kiss3d::prelude::*;

let mut window = Window::new("2D").await;
let mut camera = PanZoomCamera2d::default();
let mut scene = SceneNode2d::empty();

let mut rect = scene.add_rectangle(50.0, 100.0);
while window.render_2d(&mut scene, &mut camera).await {
    rect.rotate(0.01);
}
```

## New Features

### Multiple Lights Support
- Scene now supports up to 8 lights (instead of just one)
- **New light types**: `Light::point()`, `Light::directional()`, `Light::spot()`
- Lights attach to scene nodes via `scene.add_light(light)`
- Convenience methods: `add_point_light()`, `add_directional_light()`, `add_spot_light()`
- Lights inherit transforms from parent scene nodes
- Light properties: `color`, `intensity`, `enabled`, `attenuation_radius`
- Builder pattern: `Light::point(100.0).with_color(RED).with_intensity(5.0)`

### Ambient Light Control
- `window.set_ambient(f32)` - set ambient light intensity
- `window.ambient()` - get current ambient light intensity

### Color Module
- New `kiss3d::color` module with all CSS named colors
- Example: `color::RED`, `color::LIME_GREEN`, `color::STEEL_BLUE`
- New `Color` struct (alias of `Rgba<f32>`) for colors with alpha channel

### Subdivision Control for Primitives
- `add_sphere_with_subdiv(r, ntheta, nphi)` - control sphere tessellation
- `add_cone_with_subdiv(r, h, nsubdiv)` - control cone segments
- `add_cylinder_with_subdiv(r, h, nsubdiv)` - control cylinder segments
- `add_capsule_with_subdiv(r, h, ntheta, nphi)` - control capsule tessellation

### Optional Serde Support
- New `serde` feature flag for serialization support
- Enable with: `kiss3d = { version = "0.39", features = ["serde"] }`

### Default Cameras
- `OrbitCamera3d::default()` - starts at (0, 0, -2) looking at origin
- `PanZoomCamera2d::default()` - centered at origin with 2x zoom

### Multi-Window Support Improvements
- Better handling of multiple windows in the same application

### Line Rendering
- `draw_line()` now takes a `perspective` parameter to control size scaling with distance
- 2D lines now render on top of 2D surfaces

---

# v0.38.0

Switch from OpenGL (glow/glutin) to wgpu for cross-platform GPU support.

## Breaking Changes

### Graphics Backend
- **Replaced** OpenGL backend (glow/glutin) with wgpu
- `Window::new()` is now async: `Window::new("Title").await`
- Shaders switched from GLSL to WGSL (`.vert`/`.frag` → `.wgsl`)

### Line Rendering API
- **Replaced** `LineRenderer` with `PolylineRenderer`
- **Replaced** `PlanarLineRenderer` with `PlanarPolylineRenderer`
- **Removed** `set_line_width()` - width is now per-line
- **Changed** `draw_line(a, b, color)` → `draw_line(a, b, color, width)`
- **Changed** `draw_planar_line(a, b, color)` → `draw_planar_line(a, b, color, width)`
- **Added** `Polyline` struct with builder pattern for multi-segment lines
- **Added** `PlanarPolyline` struct for 2D multi-segment lines
- **Added** `draw_polyline(&Polyline)` and `draw_planar_polyline(&PlanarPolyline)`

### Point Rendering API
- **Removed** `set_point_size()` - size is now per-point via `draw_point(pt, color, size)`

### Removed Types
- `Effect` (shader programs now use wgpu pipelines)
- `GLPrimitive`, `gl_context` module
- `line_renderer` module (use `polyline_renderer`)
- `planar_line_renderer` module (use `planar_polyline_renderer`)

### Dependencies
- **Removed**: `glow`, `glutin`, `glutin-winit`, `egui_glow`
- **Added**: `wgpu`, `bytemuck`, `log`
- **Changed**: `egui_glow` → `egui-wgpu`

## New Features

### Video Recording (`recording` feature)
- `begin_recording()` / `begin_recording_with_config(RecordingConfig)`
- `end_recording(path, fps)` - encodes to MP4
- `pause_recording()` / `resume_recording()`
- `is_recording()` / `is_recording_paused()`
- `RecordingConfig` with `frame_skip` option
- Requires ffmpeg libraries at runtime

### Polyline Rendering
- `Polyline::new(vertices)` with builder methods:
  - `.with_color(r, g, b)`
  - `.with_width(width)`
  - `.with_perspective(bool)`
  - `.with_depth_bias(bias)`
  - `.with_transform(Isometry3)`
- `PlanarPolyline::new(vertices)` with similar builder API

### Other
- `snap_image()` returns `ImageBuffer<Rgb<u8>, Vec<u8>>` directly

---

# v0.37.2

- Fix the egui UI not rendering if the display's hidpi factor is exactly 1.0.

# v0.37.1

- Improved documentations.
- Fix issue where lighting would not behave properly when an object is rotated through the instancing deformation matrix.
- Implement `Default` for `ArcBall`.

# v0.37.0

This release introduces async rendering support for better cross-platform compatibility (especially WASM), replaces the deprecated conrod UI library with egui, and updates several key dependencies.

## Breaking Changes

### Async Rendering API
- **Removed** `State` trait and `render_loop` methods
- **Introduced** `#[kiss3d::main]` procedural macro for platform-agnostic entry points
- **Changed** `window.render()` to async `window.render().await`
- The async API automatically handles platform differences:
  - **Native**: Uses `pollster::block_on` (re-exported by kiss3d)
  - **WASM**: Uses `wasm_bindgen_futures::spawn_local` and integrates with browser's `requestAnimationFrame`

**Migration example**:
```rust
// Old (v0.36.0)
fn main() {
    let mut window = Window::new("Title");
    while window.render() {
        // render loop
    }
}

// New (v0.37.0)
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Title");
    while window.render().await {
        // render loop
    }
}
```

### UI Library Changes
- **Replaced** conrod with egui for UI rendering
- egui is now an optional feature (enabled with `features = ["egui"]`)
- UI examples require the `egui` feature flag: `cargo run --example ui --features egui`

### Dependency Updates
- **glutin**: Updated to 0.32 (native only)
- **glow**: Updated to 0.16
- **image**: Updated to 0.25
- **egui**: 0.32 (optional feature)
- **bitflags**: Updated to 2.x
- **rusttype**, **env_logger**: Version bumps

## New Features

### Async Rendering Support (#339)
- Cross-platform async rendering with `#[kiss3d::main]` macro
- Better WASM integration with browser event loop
- Automatic platform-specific runtime management
- No need to manually add `pollster` or `wasm-bindgen-futures` dependencies

### egui Integration (#340)
- Modern immediate mode GUI library replaces deprecated conrod
- Optional feature flag for users who don't need UI
- Updated UI examples demonstrating egui integration
- Better rendering performance and maintenance

### WASM Improvements
- Auto-create canvas element if it doesn't exist (WASM targets)
- Improved instancing examples compatible with WASM
- Better async integration with browser APIs

### New Examples
- `instancing2d.rs`: Demonstrates 2D instancing with multiple shapes
- `instancing3d.rs`: Demonstrates 3D instancing with transformations and colors

## Bug Fixes
- Fixed obj.rs example file not found error (#327)
- Adjusted arcball camera near/far clipping planes for better depth precision
- Fixed various warnings and compatibility issues

## Migration Guide

### Update your main function:
1. Add `#[kiss3d::main]` attribute
2. Make the function `async`
3. Add `.await` to `window.render()`

### If using UI features:
1. Enable the `egui` feature in Cargo.toml
2. Update UI code to use egui instead of conrod (if you were using conrod)

### Dependencies:
No changes needed to your Cargo.toml if you're only using kiss3d's public API. The async runtime dependencies are re-exported by kiss3d.

---

# v0.36.0

This changelog documents the changes between the `master` branch and the `nalgebra-parry` branch.

## Overview

This branch updates kiss3d to use the latest versions of nalgebra and parry3d, replacing the deprecated ncollide3d library. Additionally, it incorporates the procedural mesh generation capabilities directly into kiss3d.

## Breaking Changes

### Dependency Updates

**nalgebra: 0.30 → 0.33**
- Updated from nalgebra 0.30 to 0.33 for both main and dev dependencies
- This is a major version update that may affect user code depending on nalgebra types

**ncollide3d → parry3d**
- `ncollide3d 0.33` has been replaced by `parry3d 0.17`
- `ncollide2d 0.33` has been replaced by `parry2d 0.17` (dev dependency)
- parry3d is the successor to ncollide3d with improved APIs and maintenance

### API Changes

#### Type Renames
- `Mesh` → `GpuMesh`: The internal mesh type has been renamed to better reflect its purpose
- Methods now use `GpuMesh` instead of `Mesh` in return types and parameters

#### Procedural Module
- The procedural mesh generation module has been copied from ncollide3d into kiss3d at `src/procedural/`
- New types introduced:
  - `RenderMesh`: High-level mesh descriptor for procedural generation
  - `RenderPolyline`: Descriptor for polyline generation
  - `IndexBuffer`: Enum for unified or split index buffers

#### MeshManager Changes
- `MeshManager::add_trimesh()` now accepts `parry3d::shape::TriMesh` instead of `ncollide3d::procedural::TriMesh`
- New method: `MeshManager::add_render_mesh()` for adding `RenderMesh` objects
- Default shapes (sphere, cube, cone, cylinder) now use `add_render_mesh()` instead of `add_trimesh()`

#### SceneNode Changes
- `add_render_mesh()`: New method to add procedurally generated meshes
- `add_trimesh()`: Updated to accept `parry3d::shape::TriMesh`
- All geometry addition methods internally use the new `RenderMesh` type

## New Features

### Procedural Mesh Generation Module

A complete procedural mesh generation module has been added at `src/procedural/` (copied from ncollide):

#### Basic Shapes
- **Cuboids**: `unit_cuboid()`, `cuboid()`, `unit_rectangle()`, `rectangle()`
- **Spheres**: `unit_sphere()`, `sphere()`, `unit_hemisphere()`, `unit_circle()`, `circle()`
- **Cones**: `unit_cone()`, `cone()`
- **Cylinders**: `unit_cylinder()`, `cylinder()`
- **Capsules**: `capsule()`
- **Quads**: `unit_quad()`, `quad()`, `quad_with_vertices()`

#### Path Generation
- Path extrusion system for creating shapes from 2D paths
- Path caps: `ArrowheadCap`, `NoCap`
- `PolylinePath` and `PolylinePattern` for complex path-based shapes

#### Utilities
- Bézier curve and surface generation
- Mesh manipulation utilities
- Normal and tangent computation

#### RenderMesh Type
The new `RenderMesh` type provides:
- Vertex coordinates, normals, UVs
- Flexible index buffers (unified or split per-primitive type)
- Conversion to/from `parry3d::shape::TriMesh`
- Direct addition to scenes via `SceneNode::add_render_mesh()`

## Migration Guide

### For Library Users

1. **Update Cargo.toml dependencies**:
```toml
[dependencies]
nalgebra = "0.33"
parry3d = "0.17"  # if using directly
```

2. **Update imports**:
```rust
// Replace ncollide3d with parry3d
use parry3d::shape::TriMesh;
use parry3d::transformation;

// Use kiss3d's procedural module
use kiss3d::procedural;
```

3. **Update mesh creation**:
```rust
// Old approach
use ncollide3d::procedural;
let mesh = procedural::unit_sphere(50, 50, true);

// New approach
use kiss3d::procedural;
let mesh = procedural::unit_sphere(50, 50, true);
window.add_render_mesh(mesh, scale);
```

4. **Update decomposition code** (if using VHACD):
```rust
// Old
use ncollide3d::transformation::HACD;

// New
use parry3d::transformation;
use parry3d::transformation::vhacd::VHACDParameters;
```

### Internal Changes

- Shader version pragma updated in vertex shaders
- Matrix and vector types now use nalgebra 0.33 conventions
- Material trait implementations updated for new type signatures
- OBJ loader updated to work with `GpuMesh` instead of `Mesh`

## File Changes Summary

- **41 files changed**: 2,423 insertions(+), 176 deletions(-)
- **New files**: Entire `src/procedural/` module (~2,000 lines)
- **Modified core files**:
  - `Cargo.toml`: Dependency updates
  - `src/lib.rs`: Re-export parry3d instead of ncollide3d
  - `src/resource/mesh_manager.rs`: Updated for `GpuMesh` and new procedural module
  - `src/scene/scene_node.rs`: New methods for `RenderMesh`
  - Multiple example files updated to demonstrate new API

## Examples Updated

- `custom_material.rs`: Updated imports and mesh handling
- `custom_mesh.rs`: Updated to use new mesh types
- `custom_mesh_shared.rs`: Updated to use new mesh types
- `decomp.rs`: Updated to use parry3d's VHACD implementation
- `procedural.rs`: Updated to demonstrate procedural module usage

## Compatibility Notes

- This is a breaking change that requires updating user code
- The API surface is similar but not identical to the ncollide-based version
- parry3d has better maintained and more feature-rich than ncollide3d
- The procedural module is now part of kiss3d, eliminating a dependency

## Benefits

1. **Up-to-date dependencies**: Latest nalgebra and parry3d versions with bug fixes and improvements
2. **Simplified dependency tree**: Procedural generation now built-in to kiss3d
3. **Better maintenance**: parry3d is actively maintained, unlike ncollide3d
4. **More control**: Having procedural generation in-tree allows for kiss3d-specific optimizations

## Testing

All existing tests pass with the new dependencies. Examples have been updated and verified to work correctly.
