# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kiss3d is a "Keep It Simple, Stupid" 3D graphics engine for Rust. It's designed for simplicity and ease of use rather than feature completeness or performance. The library allows developers to draw simple geometric figures with minimal code and supports both native platforms and WASM without code changes.

**Key Principle**: One-liner features (from the user's point of view) are strongly preferred. Keep things simple.

## Dependencies

- **glam 0.29**: Used for all public math operations (vectors, matrices, quaternions)
- **nalgebra 0.34** (optional, with `parry` feature): For parry3d interop
- **parry3d 0.25** (optional, with `parry` feature): Physics/geometry library (successor to ncollide3d)
- **glow 0.16**: OpenGL bindings
- **glutin 0.32** (native only): Window creation and OpenGL context management
- **egui 0.32** (optional feature): Immediate mode GUI integration
- **wasm-bindgen** (WASM only): Browser integration

If users have glam in their project, they must use version 0.29 to match kiss3d's version.

## Feature Flags

- **egui**: Optional immediate mode GUI integration (enables `egui` and `egui_glow` dependencies)
- **parry**: Optional parry3d integration for physics/geometry (enables `nalgebra` and `parry3d` dependencies). Required for `add_trimesh()` methods and `From<TriMesh>` conversions.

Enable features in Cargo.toml:
```toml
[dependencies]
kiss3d = { version = "0.38", features = ["egui"] }
# Or with parry support:
kiss3d = { version = "0.38", features = ["parry"] }
```

## Build and Test Commands

### Building
```bash
# Check for errors (fast)
cargo check

# Build the project
cargo build

# Build with release optimizations
cargo build --release

# Check/build for WASM
cargo check --target wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --release
```

### Testing
```bash
# Run all tests
cargo test

# Run a specific test
cargo test test_name

# Run tests with output visible
cargo test -- --nocapture
```

### Running Examples
```bash
# Run a specific example (native)
cargo run --example cube
cargo run --example custom_material

# Run example with parry feature (for TriMesh support)
cargo run --example procedural --features parry
cargo run --example decomp --features parry

# Run example with egui feature
cargo run --example ui --features egui

# Run example with timeout (useful for quick testing)
timeout 3 cargo run --example cube

# Run with backtrace for debugging
RUST_BACKTRACE=1 cargo run --example cube
RUST_BACKTRACE=full cargo run --example cube

# Run example for WASM
cargo run --example cube --target wasm32-unknown-unknown
```

There are 36+ examples in the `examples/` directory demonstrating various features including:
- Basic shapes: `cube`, `primitives`
- Geometry with parry: `procedural`, `decomp` (require `parry` feature)
- Interaction: `event`, `mouse_events`, `camera`
- Advanced rendering: `custom_material`, `texturing`, `wireframe`, `post_processing`
- Scene management: `group`, `add_remove`, `instancing3d`
- UI integration: `ui` (requires `egui` feature)
- File loading: `obj`

### Linting
```bash
# Run clippy for linting
cargo clippy
```

### Documentation
```bash
# Build and open documentation
cargo doc --open
```

## Architecture

### Core Modules

**window** (`src/window/`)
- `Window`: Main entry point for the 3D engine. Handles rendering loop, event processing, and scene management.
- `Canvas`: Platform abstraction layer
  - `gl_canvas.rs`: Native OpenGL canvas implementation (glutin-based)
  - `webgl_canvas.rs`: WebGL canvas for WASM targets
- Window owns the scene graph, cameras, renderers, and manages the render loop.

**scene** (`src/scene/`)
- `SceneNode`: Hierarchical scene graph nodes for 3D objects. Each node can have:
  - Local and world transforms (Pose3) and scales
  - Children nodes (forming a hierarchy)
  - An optional `Object` to render
  - Parent references (using Weak pointers to avoid cycles)
- `PlanarSceneNode`: 2D scene graph nodes for overlay rendering
- `Object`: Wraps a `GpuMesh` and `Material` for rendering

**transform** (`src/transform.rs`)
- `Pose3` and `Pose2`: Rigid body transform types from glamx (rotation + translation)
- Helper functions for working with transforms

**resource** (`src/resource/`)
- `GpuMesh`: GPU-side mesh data with vertices, indices, normals, and UVs. All geometry data is stored in GPU buffers via `GPUVec`.
- `RenderMesh`: CPU-side mesh descriptor used for procedural generation (converted to `GpuMesh`)
- `MeshManager`: Singleton that manages shared mesh resources
- `Material`/`MaterialManager`: Shader programs and material properties
- `Texture`/`TextureManager`: Texture loading and management
- `FramebufferManager`: Manages render targets for post-processing

**procedural** (`src/procedural/`)
- Procedural mesh generation copied from ncollide3d
- Generators for basic shapes: cuboid, sphere, cone, cylinder, capsule, quad
- Path-based shape generation (extrusion, bezier curves)
- `RenderMesh`: High-level mesh descriptor with flexible index buffers
- All procedural functions return `RenderMesh` which can be added to scenes via `SceneNode::add_render_mesh()`

**camera** (`src/camera/`)
- `Camera` trait: Defines camera interface
- `OrbitCamera3d`: Default orbital camera (scroll to zoom, click+drag to rotate/pan)
- `FirstPersonCamera3d`: FPS-style camera
- Custom cameras can be implemented via the `Camera` trait

**renderer** (`src/renderer/`)
- `LineRenderer`: Renders 3D lines
- `PointRenderer`: Renders 3D points
- `EguiRenderer`: Integrates egui immediate mode GUI (when `egui` feature is enabled)

**builtin** (`src/builtin/`)
- Built-in shader programs (vertex and fragment shaders)
- Default materials and effects

**loader** (`src/loader/`)
- Mesh loading from files (OBJ, etc.)

**event** (`src/event/`)
- Event handling abstraction for keyboard, mouse, window events
- Platform-agnostic event types

**text** (`src/text/`)
- Text rendering using rusttype
- Font loading and glyph rasterization

**post_processing** (`src/post_processing/`)
- Post-processing effects framework
- Effects are applied after scene rendering

**context** (`src/context/`)
- OpenGL context management and state tracking

### Cross-Platform Support

The `#[kiss3d::main]` procedural macro (in `kiss3d-macro/`) enables writing platform-agnostic code.

**Implementation Details** (`kiss3d-macro/src/lib.rs`):
- The macro validates that it's applied to an async `fn main()` with no parameters
- Expands to two different entry points based on target architecture:
  - Native: Calls `::kiss3d::pollster::block_on(__kiss3d_async_main())`
  - WASM: Calls `::kiss3d::wasm_bindgen_futures::spawn_local(__kiss3d_async_main())`
- The original async function becomes `__kiss3d_async_main()`

**Usage**:

```rust
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Title");
    while window.render().await {
        // Render loop
    }
}
```

**Native**: `pollster::block_on` runs the async function synchronously
**WASM**: `wasm_bindgen_futures::spawn_local` integrates with browser's `requestAnimationFrame`

The `window.render().await` call:
- Native: Returns immediately each frame
- WASM: Yields to browser's event loop, returns when next frame is ready

Users should NOT add `pollster` or `wasm_bindgen_futures` to their dependencies - they are re-exported by kiss3d.

### Important Type Relationships

```
Window
├── SceneNode (3D root)
│   ├── Children: Vec<SceneNode>
│   └── Object (optional)
│       ├── GpuMesh (vertices, indices, normals, UVs on GPU)
│       └── Material (shader + properties)
├── PlanarSceneNode (2D overlay root)
├── Camera (Rc<RefCell<dyn Camera>>)
├── Renderers (LineRenderer, PointRenderer, TextRenderer)
└── Managers (MeshManager, MaterialManager, TextureManager, FramebufferManager)
```

### Scene Graph Pattern

The scene graph uses `Rc<RefCell<SceneNodeData>>` internally:
- `Rc` for shared ownership between parent and children
- `RefCell` for interior mutability
- Parent references use `Weak<RefCell<SceneNodeData>>` to prevent cycles
- Transforms are hierarchical (parent transforms affect children)

### Mesh Creation Flow

1. Generate `RenderMesh` using `procedural` module OR load from file
2. Add to scene: `window.add_cube()` or `scene_node.add_render_mesh(mesh)`
3. Internally: `RenderMesh` → `GpuMesh` (uploads to GPU)
4. `MeshManager` caches meshes to avoid duplication
5. `SceneNode` wraps `GpuMesh` + `Material` in an `Object`

## Common Patterns

### Adding Geometry
```rust
use glamx::Vec3;

// High-level API (recommended)
let mut cube = window.add_cube(1.0, 1.0, 1.0);
let mut sphere = window.add_sphere(0.5);

// Lower-level: add to specific scene node
let mesh = procedural::sphere(0.5, 32, 32, true);
let mut node = scene_node.add_render_mesh(mesh);

// Custom mesh from TriMesh
let trimesh = parry3d::shape::TriMesh::new(...);
let mut node = scene_node.add_trimesh(trimesh, Vec3::splat(1.0));
```

### Transforming Objects
```rust
use glamx::{Quat, Vec3};

// Set position
node.set_position(Vec3::new(1.0, 2.0, 3.0));

// Set rotation
let rot = Quat::from_axis_angle(Vec3::Y, 0.1);
node.set_rotation(rot);

// Incremental transforms
node.translate(Vec3::new(0.1, 0.0, 0.0));
node.prepend_rotation(&rot);

// Set scale
node.set_local_scale(2.0, 1.0, 2.0);
```

### Event Handling
```rust
for event in window.events().iter() {
    match event.value {
        WindowEvent::Key(button, Action::Press, _) => {
            // Handle key press
        }
        WindowEvent::MouseButton(button, Action::Press, _) => {
            // Handle mouse button
        }
        _ => {}
    }
}
```

## Version History Context

### v0.38.0 (Current)
- **Major**: Migrated public API from `nalgebra` to `glam` for math operations
- Uses `Pose3` and `Pose2` types from glamx (replacing nalgebra's Isometry types)
- nalgebra is now internal-only for parry3d interop
- All examples updated to use glam types

Key migration points from v0.37.0:
- Replace `nalgebra` imports with `glam` (`Vec3`, `Vec2`, `Mat4`, `Quat`, etc.)
- `Point3<f32>` → `Vec3` (glam uses vectors for both points and directions)
- `UnitQuaternion<f32>` → `Quat`
- `Translation3<f32>` → `Vec3`
- `Vector3::y_axis()` → `Vec3::Y`
- `UnitQuaternion::from_axis_angle(&Vector3::y_axis(), angle)` → `Quat::from_axis_angle(Vec3::Y, angle)`

### v0.37.0
- Latest release with all v0.36.0 features stabilized
- Egui integration improvements
- Bug fixes and stability improvements

### v0.36.0 (Major Breaking Changes)
- Replaced `ncollide3d` with `parry3d` (ncollide's successor)
- Updated `nalgebra` from 0.30 to 0.34
- Renamed `Mesh` → `GpuMesh` to clarify it's GPU-side data
- Copied procedural mesh generation from ncollide into kiss3d
- Introduced `RenderMesh` type for CPU-side mesh description
- Switched to async API with `#[kiss3d::main]` macro for cross-platform support
- Replaced conrod with egui for UI (optional feature)
- Updated glutin to 0.32 for window management

Key migration points for users:
- Use `parry3d::shape::TriMesh` instead of `ncollide3d::procedural::TriMesh`
- Use `kiss3d::procedural` instead of importing from ncollide
- Methods return `GpuMesh` instead of `Mesh`
- Must use `#[kiss3d::main]` and async render loop

## Code Style Notes

- Use `glam` types for all public APIs (`Vec3`, `Vec2`, `Mat4`, `Quat`, etc.)
- nalgebra is used internally only for parry3d interop
- Heavy use of `Rc<RefCell<T>>` for shared mutable state
- Platform-specific code uses `#[cfg(target_arch = "wasm32")]` and `#[cfg(not(target_arch = "wasm32"))]`
- Allowed clippy lints: `module_inception`, `too_many_arguments`, `type_complexity`
