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
