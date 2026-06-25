# v0.44.0

## Breaking Changes

- Removed the `hw_raytracer` cargo feature. The hardware ray-query backend is now selected automatically at runtime when the GPU supports it, otherwise the compute backend is used.
- `Light` gained a public `layers: u32` field (lighting channels). Code that builds a `Light` with a struct literal must set it (defaults to `u32::MAX`); the `Light::point(..)` / `with_layers(..)` builders are unaffected.
- The `egui` feature now also enables the `rfd` dependency (native file dialogs used by the inspector).
- New dependencies: `gltf` 1.4 and `wesl` 0.4.

## New Features

### glTF, skinning & animation

- Load glTF/GLB models with `SceneNode3d::add_gltf` / `add_gltf_from_memory` (or `loader::gltf::load` / `load_from_slice`), returning a `GltfModel` (a scene node plus an `AnimationPlayer` holding every clip in the file).
- GPU vertex skinning (`Skin3d`) and morph targets (`Object3d::set_morph_weights`, up to 64 targets) — both run on the web backend too.
- Keyframe animation playback: `AnimationPlayer`, `AnimationClip`, `AnimationChannel`, `Interpolation`.
- Example: `gltf`.

### Materials

- Extended PBR StandardMaterial parameters on `Object3d`/`SceneNode3d`: `set_clearcoat`, `set_anisotropy`, `set_reflectance`, and volumetric `set_attenuation` / `set_thickness`.
- `AlphaMode` (`Opaque` / `Mask` / `Blend` / `Premultiplied`) via `set_alpha_mode`.
- Parallax-occlusion and relief mapping from a height map: `set_height_map` / `set_height_map_from_file`, `set_parallax_scale`, `set_parallax_layers`, `set_parallax_method` (`ParallaxMethod`).
- Examples: `material_pbr`, `parallax`, `fog`, `transmission`.

### Lighting & shadows

- Clustered (forward+) light culling, lifting the previous fixed light cap; enabled automatically when the GPU exposes compute/storage, with a fixed-light fallback otherwise.
- Lighting channels: per-light `Light::with_layers` / `Light.layers` and per-object `Object3d::set_light_layers`, to confine a light to a subset of objects.
- Per-object shadow opt-out via `set_casts_shadows(false)`.
- Configurable soft shadows: `Window::set_shadow_softness`.
- Colored ambient light (`Window::set_ambient_color`) and distance fog (`Window::set_fog`, `light::Fog` / `FogMode`).
- Example: `clustered_lights`.

### Reflections & screen-space effects

- Reflection probes (baked image or runtime cube capture), parallax-corrected: `Window::add_reflection_probe`, `capture_reflection_probe`, `set_reflection_probe_image`, `set_reflection_capture_layers`.
- Screen-space reflections: `Window::set_ssr_enabled` / `ssr_settings_mut`, with per-object `Object3d::set_ssr(SsrMaterial)`.
- Planar mirror reflectors integrated into the default PBR material: `SceneNode3d::add_reflector`, `Object3d::set_reflector` and `set_reflector_*`.
- Reflector captures render every phase, not just opaque surfaces: transparent (alpha < 1) objects are drawn into mirrors with the same weighted-blended OIT as the main pass, and refractive glass is drawn refracting the mirrored scene behind it (one snapshot layer).
- Screen-space ambient occlusion: `Window::set_ssao_enabled` / `ssao_settings_mut`.
- Screen-space refractive transmission (glass): `Window::set_transmission_enabled` / `transmission_settings_mut`.
- Examples: `reflections`, `mirror`, `mirror_sphere`.

### Camera, exposure & post-processing

- Orthographic 3D projection: `camera::Projection`, `OrbitCamera3d::set_projection` / `projection`.
- Physical camera exposure: `camera::Exposure` (`from_physical` / `from_exposure`), applied with `Window::set_exposure_value`.
- Equirectangular skybox + image-based lighting: `Window::set_skybox_from_file` / `set_skybox_from_memory` / `set_skybox_image` / `set_skybox_orientation` / `clear_skybox`.
- Color grading and auto-exposure (eye adaptation) on the HDR resolve, via `Window::hdr_settings_mut` (`ColorGrading`, `auto_exposure*` fields).
- Thin-lens depth of field: `Window::set_dof_enabled` / `dof_settings_mut` (`DofSettings`, `DepthOfFieldMode`).
- MSAA: `Window::set_samples(NumSamples)`.
- FXAA and CAS post-processing (`post_processing::Fxaa`, `Cas`) plus a pixel-inspection `Loupe` (`LoupeCorner`).
- Render layers (per-object and per-camera masks): `Object3d::set_render_layers`, `OrbitCamera3d::set_render_layers`.
- Runtime vsync toggle: `Window::set_vsync`.
- Examples: `camera_modes`, `skybox`, `color_grading`, `depth_of_field`, `antialiasing`.

### Tooling & internals

- Built-in egui scene inspector: `window::Inspector` + `Window::draw_inspector` (toggle effects, edit materials/lights/cameras, switch between the rasterizer and path tracer). Example: `inspector`.
- GPU render timings: `Window::render_timings` (`RenderTimings`).
- Optional `rt_switcher` feature: drive both backends through `render_3d` for side-by-side rasterizer/ray-tracer comparison.
- WGSL shaders refactored with WESL conditional compilation behind a pipeline cache, plus a shader-variant validity test.

# v0.43.0

## Breaking Changes

- The rasterizer now renders into a linear HDR film that is tonemapped on resolve, with **Khronos PBR Neutral as the default tonemap operator**. Existing scenes therefore look different (more filmic highlight roll-off, less hard clipping). Call `Window::set_tonemap(Tonemap::None)` to restore the previous look.
- **Shadows are enabled by default.** Lights now cast shadows (`Light::casts_shadows` defaults to `true`) whenever shadows are globally enabled, and the default shadow atlas is 4096² across 16 layers (~1 GB of GPU memory). Use `Window::set_shadows_enabled(false)` and/or `set_shadow_resolution(2048)` to reduce this.
- **Custom `Material3d` implementations** must target `Context::render_format()` (the `Rgba16Float` HDR film) instead of the surface format for their color attachment, or they fail with a render-pass-incompatibility validation error. A material is only invoked in the new transparent (order-independent-transparency) pass if it returns `true` from the new `Material3d::renders_in_transparent_phase()` (default `false`), so opaque custom materials need no other change.
- `RenderContext` gained `phase: RenderPhase` and `shadow_bind_group: Option<wgpu::BindGroup>` fields; `Light` gained `radius: f32` and `casts_shadows: bool` fields. Code that builds or exhaustively destructures these structs with a struct literal must be updated (builder usage such as `Light::point(..).with_intensity(..)` is unaffected).

## New Features

### GPU path tracer

- Added a progressive Monte-Carlo **path tracer** (`RayTracer`) that renders the existing scene graph as an alternative to the rasterizer, via `Window::raytrace_3d` and `OffscreenSurface::render_image_raytraced`.
  - Two backends sharing one WGSL kernel: a software compute-shader BVH traversal and a hardware ray-query path, selected automatically at runtime (the hardware path is used when the GPU supports it, otherwise it falls back to the compute backend).
  - Two-level (instanced) BVH so instanced scene nodes are traced once per mesh.
  - Unified PBR/BSDF surface model (diffuse / metal / glass), per-object materials and textures, area-light next-event estimation with multiple-importance sampling, thin-lens depth of field, and image-based lighting from an equirectangular HDRI.
  - Alpha (coverage) transparency; ambient acts as a uniform fill light; the window background color is shown on directly-seen ray misses without lighting the scene.
  - Edge-aware à-trous denoiser with first-hit albedo/normal guides (`RayTracer::set_denoise`).
  - BSDF setters on `SceneNode3d` (`set_metallic`, `set_roughness`, `set_emissive`, …).
  - Examples: `raytracing`, `raytracing_bsdf`, `raytracing_denoise`, `raytracing_offscreen`, `raytracing_transparency`.

### HDR rendering pipeline

- The rasterizer now renders into a linear `Rgba16Float` **HDR film** and resolves it with a configurable tonemap operator (`Window::set_tonemap` / `set_exposure`).
  - Operators: `None`, `Aces`, `Reinhard`, `AgX`, `Neutral` (Khronos PBR Neutral, the default), and `TonyMcMapface` (baked CC0 LUT). The same operator applies to the path tracer's resolve.
  - Physically-weighted **bloom** on the HDR film (`Window::set_bloom_enabled` and related settings).
  - Weighted-blended **order-independent transparency** for the rasterizer (correct blending of overlapping transparent surfaces without sorting).
  - Examples: `hdr_bloom`, `tonemapping`, `transparency`.

### Real-time shadows

- Added real-time **shadow mapping** for directional, spot and point lights, applied in the PBR lighting pass.
  - Directional lights use **cascaded shadow maps** (the camera frustum is split logarithmically, each cascade fit with a rotation-invariant, texel-snapped projection derived from the camera so shadows don't shimmer or degrade with scene size, and cascades are cross-faded across boundaries) for crisp near shadows with bounded far coverage.
  - Spot lights use a perspective map; point lights an unrolled cube map.
  - Castaño 2013 optimized PCF (tent-weighted 9-tap) for smooth, crisp edges.
  - Configurable: `Window::set_shadows_enabled` / `set_shadow_resolution`, plus cascade count, shadow distance, first-cascade bound and depth bias on the shadow mapper. Per-light `Light::with_casts_shadows`.
  - Example: `shadows`.

### Auxiliary render outputs (AOVs)

- Added auxiliary render outputs produced by re-rendering the scene with dedicated materials (no path tracer required): linear **depth**, surface **normals**, and per-object **segmentation** ids (with a colorized variant).
  - `Window::snap_depth` / `snap_normals` / `snap_segmentation` / `snap_segmentation_colored`, a per-object segmentation id on `Object3d`, and an `aov` example.

# v0.42.0

## Breaking Changes

- Bumped `glamx` dependency: 0.2 → 0.3. ([#384](https://github.com/dimforge/kiss3d/pull/384))

## New Features

### Off-screen rendering on the web + zero-copy egui display

- `OffscreenSurface` now exists on wasm: creation and all GPU-side rendering (`render_3d`, `raytrace_3d`, ...) work in the browser. Only the CPU read-backs (`snap_*`, `render_image_*`) remain native-only (they must block on the GPU).
- `OffscreenSurface::output_view` exposes the surface's final (post-tonemap) texture, and `Window::register_egui_texture` / `unregister_egui_texture` register any wgpu texture view with the window's egui renderer — so an offscreen surface can be displayed live in an egui UI with zero GPU→CPU copies, on native and web alike.
- GPU-only AOV visualization: `Window::render_aov_3d` / `OffscreenSurface::render_aov_3d` render depth (fixed-range grayscale), normals or colorized segmentation as a display-ready image into the surface's output texture, with no read-back. The `robot_view` example uses all of the above.

### Off-screen Rendering ([#382](https://github.com/dimforge/kiss3d/pull/382))

- Added `OffscreenSurface`: a truly headless render target with no window and no event loop — works on CI, servers and other environments without a display server (native only).
  - `OffscreenSurface::new(width, height)` / `with_setup(width, height, CanvasSetup)`
  - `render_3d` / `render_2d` / `render` — share the same scene graph, cameras, lights and materials as `Window`
  - `render_image_3d` — render a frame and capture it in one call
  - `snap`, `snap_rect`, `snap_image`, `resize`, `size`, `width`, `height`, `set_background_color`
- Hidden windows (`Window::new_hidden*`) now render into an off-screen texture, so `snap*` and the `recording` feature work on them too. ([#381](https://github.com/dimforge/kiss3d/pull/381))
- Added the `offscreen` example.

## Bug Fixes

- Fixed the first frame occasionally failing to render. A freshly created window — particularly on macOS — may need the event loop to be pumped a few times before its surface becomes presentable; surface acquisition now retries during a short startup grace period before giving up, then skips transient failures immediately on subsequent frames. ([#381](https://github.com/dimforge/kiss3d/pull/381))
- Ensured the selected surface format is supported by the device features. ([#379](https://github.com/dimforge/kiss3d/pull/379))
- Replaced stray `println!` / `eprintln!` calls in the rendering path and the OBJ/MTL loaders with `log` macros. ([#381](https://github.com/dimforge/kiss3d/pull/381))

# v0.41.0

## Breaking Changes

- Removed the `parry` feature flag and `parry3d` dependency to avoid circular dependencies when publishing.
  `SceneNode3d::trimesh`/`add_trimesh` and `MeshManager3d::add_trimesh` now take `(Vec<Vec3>, Vec<[u32; 3]>, ...)` instead of a `parry3d::shape::TriMesh`. The `From<TriMesh> for RenderMesh` conversion and the `parry3d` re-export are removed.
- `CanvasSetup` no longer implements `Copy` (now contains a `String` field) and has a new required field `canvas_id: String`. Use `..Default::default()` to fill it in. ([#372](https://github.com/dimforge/kiss3d/pull/372))
- `FixedView2d::new()` now takes `(CoordinateSystem2d, bool)` parameters instead of no arguments. Use `FixedView2d::default()` for the previous behavior. ([#354](https://github.com/dimforge/kiss3d/pull/354))
- Removed the `decomp` example (depended on `parry`).
- Bumped dependencies: `wgpu` 27 → 29, `glamx` 0.1 → 0.2, `egui`/`egui-wgpu` 0.33 → 0.34, `getrandom` 0.3 → 0.4, `oneshot` 0.1 → 0.2, `rand` (dev) 0.9 → 0.10.

## New Features

- `FixedView2d`: added `CoordinateSystem2d` enum with `CenterUp` (default, unchanged) and `TopLeftDown` (top-left origin, Y-down) coordinate systems, and a configurable `apply_hidpi` flag. ([#354](https://github.com/dimforge/kiss3d/pull/354))
- Added new `dda_raycast2d` example demonstrating 2D ray casting with the top-left coordinate system. ([#354](https://github.com/dimforge/kiss3d/pull/354))
- `Window::new_with_window_attributes()`: create a window from a `winit::window::WindowAttributes` for fine-grained control. ([#364](https://github.com/dimforge/kiss3d/pull/364))
- `Window::new_hidden_with_size()`: create a hidden window with custom dimensions. ([#365](https://github.com/dimforge/kiss3d/pull/365))
- `Window::rebind_close_key()` / `rebind_close_modifiers()`: customize or disable the window-close keybinding (default: Escape). ([#367](https://github.com/dimforge/kiss3d/pull/367))
- `PanZoomCamera2d`: added `zoom_step()` / `set_zoom_step()` ([#362](https://github.com/dimforge/kiss3d/pull/362)), and `rebind_drag_modifier()` / `rebind_zoom_modifier()` for modifier-gated drag and zoom. ([#360](https://github.com/dimforge/kiss3d/pull/360))
- `OrbitCamera3d`: added `fov()` / `set_fov()`. ([#361](https://github.com/dimforge/kiss3d/pull/361))
- Implemented `Default` for `CanvasSetup`. ([#372](https://github.com/dimforge/kiss3d/pull/372))

# v0.40.0

## Breaking Changes

- Switched to parry 0.26.0.

### SceneNode3d and SceneNode2d: Recursive vs Non-Recursive Methods

Methods that previously modified both a node and all its descendants now only modify the current node.
Use the new `_recursive` suffix variants for the previous behavior.

**Renamed methods (now non-recursive by default):**

`SceneNode3d`:
- `set_material`, `set_material_with_name`
- `set_color`, `set_texture`, `set_texture_from_file`, `set_texture_from_memory`, `set_texture_with_name`
- `set_lines_width`, `set_lines_color`, `set_points_size`, `set_points_color`
- `set_surface_rendering_activation`, `enable_backface_culling`
- `set_local_scale`, `set_visible`
- `set_metallic`, `set_roughness`, `set_emissive`
- `set_normal_map`, `set_normal_map_from_file`, `set_normal_map_from_memory`, `set_normal_map_with_name`
- `set_ao_map`, `set_ao_map_from_file`, `set_ao_map_from_memory`, `set_ao_map_with_name`
- `set_emissive_map`, `set_emissive_map_from_file`, `set_emissive_map_from_memory`, `set_emissive_map_with_name`
- `modify_vertices`, `read_vertices`, `recompute_normals`, `modify_normals`, `read_normals`
- `modify_uvs`, `read_uvs`, `modify_faces`, `read_faces`

`SceneNode2d`:
- `set_material`, `set_material_with_name`
- `set_color`, `set_texture`, `set_texture_from_file`, `set_texture_from_memory`, `set_texture_with_name`
- `set_lines_width`, `set_lines_color`, `set_points_size`, `set_points_color`
- `set_surface_rendering_activation`
- `set_local_scale`, `set_visible`
- `modify_vertices`, `read_vertices`, `modify_uvs`, `read_uvs`, `modify_faces`, `read_faces`

**New recursive variants:**
All the above methods now have `_recursive` suffix versions (e.g., `set_color_recursive`) that apply to the node and all descendants.

**Helper method renames:**
- `apply_to_objects_mut` → `apply_to_objects_mut_recursive` / `apply_to_object_mut` (new)
- `apply_to_objects` → `apply_to_objects_recursive` / `apply_to_object` (new)
- `apply_to_scene_nodes_mut` → `apply_to_scene_nodes_mut_recursive`
- `apply_to_scene_nodes` → `apply_to_scene_nodes_recursive`

# v0.39.1

Update website links in documentations.

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
