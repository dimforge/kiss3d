//! A built-in, one-call egui inspector for showcasing and debugging.
//!
//! Create your own [`Inspector`] and call [`Window::draw_inspector`] inside a
//! render loop to overlay a floating panel that exposes essentially every
//! global rendering knob, lets you toggle and tune the GPU path tracer (passed
//! in as an `Option<&mut RayTracer>`), and provides a live tree view of the
//! scene graph with per-node / per-object editing:
//!
//! ```no_run
//! use kiss3d::prelude::*;
//! use kiss3d::renderer::RayTracer;
//! use kiss3d::window::Inspector;
//!
//! #[kiss3d::main]
//! async fn main() {
//!     let mut window = Window::new("Inspector").await;
//!     let mut camera = OrbitCamera3d::default();
//!     let mut scene = SceneNode3d::empty();
//!     scene.add_cube(1.0, 1.0, 1.0);
//!     scene.add_point_light(100.0).set_position(Vec3::new(3.0, 5.0, 3.0));
//!     let mut raytracer = RayTracer::new();
//!     let mut inspector = Inspector::new();
//!
//!     while window.raytrace_3d(&mut scene, &mut camera, &mut raytracer).await {
//!         // The single line that draws the inspector. Pass `None` instead of
//!         // `Some(&mut raytracer)` for a rasterizer-only panel.
//!         window.draw_inspector(&mut inspector, &mut scene, Some(&mut raytracer));
//!     }
//! }
//! ```

use std::path::Path;

use glamx::{EulerRot, Quat, Vec3};

use crate::color::Color;
use crate::light::LightType;
use crate::post_processing::{HdrSettings, Tonemap};
use crate::renderer::{RayTracer, RenderTimings};
use crate::scene::{Bsdf, SceneNode3d};
use crate::window::NumSamples;

use super::Window;

/// Path-tracer-specific knobs mirrored by the inspector UI (exposure and tonemap
/// are *not* here: they are shared with the rasterizer, see [`WinSettings::hdr`]).
///
/// These are edited every frame and pushed onto the live [`RayTracer`] through
/// setters that already guard against redundant accumulation resets, so holding
/// them in a plain `Copy` struct is safe.
#[derive(Clone, Copy, Debug)]
struct RtKnobs {
    max_bounces: u32,
    samples_per_frame: u32,
    denoise: bool,
    denoise_iterations: u32,
    interactive_scale: f32,
    f_number: f32,
    focus_distance: f32,
    env_rotation_deg: f32,
    env_intensity: f32,
}

impl Default for RtKnobs {
    fn default() -> Self {
        // Mirrors the defaults in `RayTracer::new`.
        RtKnobs {
            max_bounces: 8,
            samples_per_frame: 1,
            denoise: false,
            denoise_iterations: 5,
            interactive_scale: 0.5,
            f_number: 0.0,
            focus_distance: 1.0,
            env_rotation_deg: 0.0,
            env_intensity: 1.0,
        }
    }
}

impl RtKnobs {
    /// Pushes these knobs onto `rt`. `prev_env` is the environment orientation
    /// last applied (so we only call the accumulation-resetting orientation
    /// setter when it actually changed); the new applied orientation is returned.
    fn apply(&self, rt: &mut RayTracer, prev_env: Option<(f32, f32)>) -> Option<(f32, f32)> {
        rt.set_max_bounces(self.max_bounces);
        rt.set_samples_per_frame(self.samples_per_frame);
        rt.set_denoise(self.denoise);
        rt.set_denoise_iterations(self.denoise_iterations);
        rt.set_interactive_scale(self.interactive_scale);
        rt.set_f_number(self.f_number, self.focus_distance);

        let env = (self.env_rotation_deg.to_radians(), self.env_intensity);
        if prev_env != Some(env) {
            rt.set_environment_orientation(env.0, env.1);
        }
        Some(env)
    }
}

/// Global rasterizer settings the panel edits, snapshotted from the window
/// before the UI runs and written back (only where changed) afterwards.
struct WinSettings {
    background: [f32; 3],
    ambient: f32,
    samples: NumSamples,
    shadows: bool,
    shadow_res: u32,
    shadow_softness: f32,
    hdr: HdrSettings,
}

/// State backing [`Window::draw_inspector`]: the floating egui panel that
/// configures every global rendering knob, toggles/tunes the path tracer, and
/// edits the scene tree.
///
/// You create and own the inspector ([`Inspector::new`]), keep it alive across
/// frames to preserve its UI state (selection, expanded sections, path-tracer
/// knobs), and pass it to [`Window::draw_inspector`] each frame.
pub struct Inspector {
    open: bool,
    initialized: bool,
    rt: RtKnobs,
    applied_env: Option<(f32, f32)>,
    env_path: String,
    env_status: String,
    /// Set by the UI when the user asks to (re)load the HDRI at `env_path`.
    env_load_requested: bool,
    /// Set by the UI when the user asks to clear the HDRI.
    env_clear_requested: bool,
    samples: NumSamples,
    /// Color used by the "apply to subtree" control for groups.
    recursive_color: [f32; 3],
    selected: Option<SceneNode3d>,
    /// Euler-angle editing buffer `(node id, degrees)` for the selected node, so
    /// the rotation sliders don't jitter through quaternion round-tripping.
    edit_rot: Option<(u64, Vec3)>,
}

impl Default for Inspector {
    fn default() -> Self {
        Inspector {
            open: true,
            initialized: false,
            rt: RtKnobs::default(),
            applied_env: None,
            env_path: String::new(),
            env_status: String::new(),
            env_load_requested: false,
            env_clear_requested: false,
            samples: NumSamples::Zero,
            recursive_color: [1.0, 1.0, 1.0],
            selected: None,
            edit_rot: None,
        }
    }
}

impl Inspector {
    /// Creates a new inspector with default state (panel shown, rasterizer view).
    ///
    /// You own the inspector: keep it alive across frames and drive it from your
    /// render loop with [`Window::draw_inspector`].
    pub fn new() -> Inspector {
        Inspector::default()
    }

    /// Whether the inspector panel is currently shown.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Shows or hides the inspector panel.
    pub fn set_open(&mut self, open: bool) {
        self.open = open;
    }

    /// Builds the whole panel for one frame.
    fn ui(
        &mut self,
        ctx: &egui::Context,
        scene: &mut SceneNode3d,
        win: &mut WinSettings,
        mut raytracer: Option<&mut RayTracer>,
        timings: Option<&RenderTimings>,
    ) {
        if !self.open {
            return;
        }

        // No `[x]` button: the window is always available (collapse it from the
        // title bar). Use `Inspector::set_open` to hide it programmatically.
        egui::Window::new("🛠 kiss3d inspector")
            .default_width(320.0)
            .default_pos([12.0, 12.0])
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(ui.ctx().content_rect().height() - 60.0)
                    .show(ui, |ui| {
                        self.renderer_section(ui, raytracer.as_deref_mut());

                        // Whether the path tracer is the active renderer for this
                        // frame (reflects the toggle just edited above). Decides
                        // which renderer-specific section is shown.
                        let path_tracing =
                            raytracer.as_deref().is_some_and(|rt| rt.enabled());

                        // Settings shared by both renderers.
                        self.common_section(ui, &mut win.hdr);

                        // Only the active renderer's specific settings are shown.
                        if path_tracing {
                            self.path_tracer_section(ui);
                        } else {
                            self.rasterizer_section(ui, win);
                        }

                        Self::timings_section(ui, timings);

                        ui.separator();
                        self.scene_tree_section(ui, scene);
                        ui.separator();
                        self.selection_section(ui, path_tracing);
                    });
            });
    }

    /// Per-step wall-clock timings of the last rendered frame.
    fn timings_section(ui: &mut egui::Ui, timings: Option<&RenderTimings>) {
        egui::CollapsingHeader::new("Timings")
            .default_open(true)
            .show(ui, |ui| match timings {
                Some(t) => {
                    // `RenderTimings` already formats a tidy multi-line summary.
                    ui.monospace(t.to_string());
                }
                None => {
                    ui.label("No frame rendered yet.");
                }
            });
    }

    fn renderer_section(&mut self, ui: &mut egui::Ui, raytracer: Option<&mut RayTracer>) {
        egui::CollapsingHeader::new("Renderer")
            .default_open(true)
            .show(ui, |ui| match raytracer {
                Some(rt) => {
                    let mut enabled = rt.enabled();
                    if ui
                        .checkbox(&mut enabled, "Enable path tracer")
                        .on_hover_text("Off renders with the rasterizer instead")
                        .changed()
                    {
                        rt.set_enabled(enabled);
                    }
                    if rt.enabled() {
                        ui.label(format!("Backend: {:?}", rt.backend()));
                        ui.label(format!("Samples accumulated: {}", rt.samples_accumulated()));
                        if ui.button("Restart accumulation").clicked() {
                            rt.mark_dirty();
                        }
                    }
                }
                None => {
                    ui.label("Rasterizer (no path tracer was passed to draw_inspector).");
                }
            });
    }

    /// Settings that affect both renderers (exposure and tonemap operator).
    fn common_section(&mut self, ui: &mut egui::Ui, hdr: &mut HdrSettings) {
        egui::CollapsingHeader::new("Common")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(
                    egui::Slider::new(&mut hdr.exposure, 0.0..=8.0)
                        .text("Exposure")
                        .logarithmic(true),
                );
                tonemap_combo(ui, "inspector_tonemap", &mut hdr.tonemap);
            });
    }

    fn rasterizer_section(&mut self, ui: &mut egui::Ui, win: &mut WinSettings) {
        egui::CollapsingHeader::new("Rasterizer")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Background");
                    ui.color_edit_button_rgb(&mut win.background);
                });
                ui.add(egui::Slider::new(&mut win.ambient, 0.0..=1.0).text("Ambient"));

                ui.horizontal(|ui| {
                    ui.label("MSAA");
                    egui::ComboBox::from_id_salt("inspector_msaa")
                        .selected_text(msaa_label(win.samples))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut win.samples, NumSamples::Zero, "Off");
                            ui.selectable_value(&mut win.samples, NumSamples::Four, "4×");
                        });
                });

                ui.checkbox(&mut win.shadows, "Shadows");
                ui.add_enabled_ui(win.shadows, |ui| {
                    ui.add(
                        egui::Slider::new(&mut win.shadow_res, 256..=8192)
                            .text("Shadow resolution")
                            .step_by(256.0),
                    );
                    ui.add(
                        egui::Slider::new(&mut win.shadow_softness, 0.0..=8.0)
                            .text("Shadow softness"),
                    );
                });

                // Bloom is a rasterizer post-process (the path tracer ignores it).
                let hdr = &mut win.hdr;
                ui.checkbox(&mut hdr.bloom_enabled, "Bloom");
                ui.add_enabled_ui(hdr.bloom_enabled, |ui| {
                    ui.add(egui::Slider::new(&mut hdr.bloom_threshold, 0.0..=4.0).text("Threshold"));
                    ui.add(egui::Slider::new(&mut hdr.bloom_knee, 0.0..=1.0).text("Knee"));
                    ui.add(egui::Slider::new(&mut hdr.bloom_intensity, 0.0..=1.0).text("Intensity"));
                });
            });
    }

    fn path_tracer_section(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Path tracer")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(egui::Slider::new(&mut self.rt.max_bounces, 1..=32).text("Max bounces"));
                ui.add(
                    egui::Slider::new(&mut self.rt.samples_per_frame, 1..=64)
                        .text("Samples / frame"),
                );

                ui.checkbox(&mut self.rt.denoise, "Denoise");
                ui.add_enabled(
                    self.rt.denoise,
                    egui::Slider::new(&mut self.rt.denoise_iterations, 1..=10)
                        .text("Denoise iterations"),
                );

                ui.add(
                    egui::Slider::new(&mut self.rt.interactive_scale, 0.05..=1.0)
                        .text("Interactive scale"),
                );

                ui.collapsing("Depth of field", |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.rt.f_number, 0.0..=32.0)
                            .text("f-number (0 = pinhole)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.rt.focus_distance, 0.01..=100.0)
                            .text("Focus distance")
                            .logarithmic(true),
                    );
                });

                ui.collapsing("Environment", |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.rt.env_rotation_deg, -180.0..=180.0)
                            .text("Rotation°"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.rt.env_intensity, 0.0..=8.0).text("Intensity"),
                    );
                    ui.horizontal(|ui| {
                        ui.label("HDRI");
                        ui.text_edit_singleline(&mut self.env_path);
                    });
                    ui.horizontal(|ui| {
                        // Applied against the live path tracer after the UI runs
                        // (see `draw_inspector`).
                        if ui.button("Load").clicked() {
                            self.env_load_requested = true;
                        }
                        if ui.button("Clear").clicked() {
                            self.env_clear_requested = true;
                        }
                    });
                    if !self.env_status.is_empty() {
                        ui.label(&self.env_status);
                    }
                });
            });
    }

    fn scene_tree_section(&mut self, ui: &mut egui::Ui, scene: &mut SceneNode3d) {
        egui::CollapsingHeader::new("Scene tree")
            .default_open(true)
            .show(ui, |ui| {
                if ui.button("Clear selection").clicked() {
                    self.selected = None;
                }
                tree_ui(ui, scene, 0, true, &mut self.selected);
            });
    }

    fn selection_section(&mut self, ui: &mut egui::Ui, path_tracing: bool) {
        let Some(node) = self.selected.clone() else {
            ui.label("Select a node in the tree to edit it.");
            return;
        };

        egui::CollapsingHeader::new("Selection")
            .default_open(true)
            .show(ui, |ui| {
                self.transform_ui(ui, &node);
                if node.data().has_object() {
                    self.material_ui(ui, &node, path_tracing);
                }
                if node.data().has_light() {
                    self.light_ui(ui, &node);
                }
                if !node.data().children().is_empty() {
                    self.subtree_ui(ui, &node);
                }
            });
    }

    /// Operations that apply to a node and its whole subtree (per-group editing).
    fn subtree_ui(&mut self, ui: &mut egui::Ui, node: &SceneNode3d) {
        egui::CollapsingHeader::new("Subtree")
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Color");
                    ui.color_edit_button_rgb(&mut self.recursive_color);
                    if ui.button("Apply to subtree").clicked() {
                        let c = self.recursive_color;
                        node.clone()
                            .set_color_recursive(Color::new(c[0], c[1], c[2], 1.0));
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Show subtree").clicked() {
                        node.clone()
                            .apply_to_scene_nodes_mut_recursive(&mut |n| {
                                n.set_visible(true);
                            });
                    }
                    if ui.button("Hide subtree").clicked() {
                        node.clone()
                            .apply_to_scene_nodes_mut_recursive(&mut |n| {
                                n.set_visible(false);
                            });
                    }
                });
            });
    }

    fn transform_ui(&mut self, ui: &mut egui::Ui, node: &SceneNode3d) {
        let mut node = node.clone();
        ui.label("Transform");

        // Position.
        let mut pos = node.position();
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Pos");
            changed |= drag(ui, &mut pos.x);
            changed |= drag(ui, &mut pos.y);
            changed |= drag(ui, &mut pos.z);
        });
        if changed {
            node.set_position(pos);
        }

        // Scale.
        let mut scale = node.local_scale();
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Scale");
            changed |= drag(ui, &mut scale.x);
            changed |= drag(ui, &mut scale.y);
            changed |= drag(ui, &mut scale.z);
        });
        if changed {
            node.set_local_scale(scale.x, scale.y, scale.z);
        }

        // Rotation (Euler degrees), buffered so the slider round-trips cleanly.
        let id = node.ptr_id();
        let mut euler = match self.edit_rot {
            Some((nid, e)) if nid == id => e,
            _ => {
                let (x, y, z) = node.rotation().to_euler(EulerRot::XYZ);
                Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees())
            }
        };
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Rot°");
            changed |= drag(ui, &mut euler.x);
            changed |= drag(ui, &mut euler.y);
            changed |= drag(ui, &mut euler.z);
        });
        if changed {
            node.set_rotation(Quat::from_euler(
                EulerRot::XYZ,
                euler.x.to_radians(),
                euler.y.to_radians(),
                euler.z.to_radians(),
            ));
        }
        self.edit_rot = Some((id, euler));
    }

    fn material_ui(&mut self, ui: &mut egui::Ui, node: &SceneNode3d, path_tracing: bool) {
        let mut node = node.clone();
        egui::CollapsingHeader::new("Material")
            .default_open(true)
            .show(ui, |ui| {
                // Base color and its opacity (the color's alpha; below 1.0 makes
                // the object transparent).
                if let Some(mut c) = obj_get(&node, |o| o.data().color()) {
                    let mut changed = color_edit(ui, "Color", &mut c);
                    changed |= slider(ui, "Opacity", &mut c.a, 0.0..=1.0);
                    if changed {
                        node.set_color(c);
                    }
                }
                if let Some(mut c) = obj_get(&node, |o| o.data().emissive()) {
                    if color_edit(ui, "Emissive", &mut c) {
                        node.set_emissive(c);
                    }
                }

                // PBR scalars.
                if let Some(mut v) = obj_get(&node, |o| o.data().metallic()) {
                    if slider(ui, "Metallic", &mut v, 0.0..=1.0) {
                        node.set_metallic(v);
                    }
                }
                if let Some(mut v) = obj_get(&node, |o| o.data().roughness()) {
                    if slider(ui, "Roughness", &mut v, 0.0..=1.0) {
                        node.set_roughness(v);
                    }
                }

                // BSDF and its parameters only affect the path tracer, so only
                // show them when path tracing is the active renderer.
                if path_tracing {
                    ui.separator();
                    ui.label("Path-tracer BSDF");
                    if let Some(mut bsdf) = obj_get(&node, |o| o.data().bsdf()) {
                        if bsdf_combo(ui, &mut bsdf) {
                            node.set_bsdf(bsdf);
                        }
                    }
                    if let Some(mut v) = obj_get(&node, |o| o.data().ior()) {
                        if slider(ui, "IOR", &mut v, 1.0..=3.0) {
                            node.set_ior(v);
                        }
                    }
                    if let Some(mut v) = obj_get(&node, |o| o.data().transmission()) {
                        if slider(ui, "Transmission", &mut v, 0.0..=1.0) {
                            node.set_transmission(v);
                        }
                    }
                    if let Some(mut c) = obj_get(&node, |o| o.data().specular_tint()) {
                        if color_edit(ui, "Specular tint", &mut c) {
                            node.set_specular_tint(c);
                        }
                    }
                    let sub =
                        obj_get(&node, |o| (o.data().subsurface(), o.data().subsurface_radius()));
                    if let Some((mut factor, mut radius)) = sub {
                        let mut changed = slider(ui, "Subsurface", &mut factor, 0.0..=1.0);
                        changed |= slider(ui, "SSS radius", &mut radius, 0.0..=5.0);
                        if changed {
                            node.set_subsurface(factor, radius);
                        }
                    }
                }

                ui.separator();
                // Surface / wireframe / points.
                if let Some(mut on) = obj_get(&node, |o| o.data().surface_rendering_active()) {
                    if ui.checkbox(&mut on, "Draw surface").changed() {
                        node.set_surface_rendering_activation(on);
                    }
                }
                if let Some(mut on) = obj_get(&node, |o| o.data().backface_culling_enabled()) {
                    if ui.checkbox(&mut on, "Backface culling").changed() {
                        node.enable_backface_culling(on);
                    }
                }

                wireframe_ui(ui, &mut node);
                points_ui(ui, &mut node);

                if let Some(mut id) = obj_get(&node, |o| o.data().segmentation_id()) {
                    ui.horizontal(|ui| {
                        ui.label("Segmentation id");
                        if ui.add(egui::DragValue::new(&mut id)).changed() {
                            if let Some(obj) = node.data_mut().object_mut() {
                                obj.set_segmentation_id(id);
                            }
                        }
                    });
                }
            });
    }

    fn light_ui(&mut self, ui: &mut egui::Ui, node: &SceneNode3d) {
        let mut node = node.clone();
        let Some(mut light) = node.light() else {
            return;
        };

        egui::CollapsingHeader::new("Light")
            .default_open(true)
            .show(ui, |ui| {
                let mut changed = false;
                changed |= ui.checkbox(&mut light.enabled, "Enabled").changed();
                changed |= slider(ui, "Intensity", &mut light.intensity, 0.0..=50.0);
                changed |= color_edit(ui, "Color", &mut light.color);
                changed |= slider(ui, "Soft radius", &mut light.radius, 0.0..=10.0);
                changed |= ui
                    .checkbox(&mut light.casts_shadows, "Casts shadows (raster)")
                    .changed();

                match &mut light.light_type {
                    LightType::Point { attenuation_radius } => {
                        ui.label("Point light");
                        changed |= slider(ui, "Attenuation", attenuation_radius, 0.0..=1000.0);
                    }
                    LightType::Spot {
                        inner_cone_angle,
                        outer_cone_angle,
                        attenuation_radius,
                    } => {
                        ui.label("Spot light");
                        changed |=
                            slider(ui, "Inner cone (rad)", inner_cone_angle, 0.0..=std::f32::consts::FRAC_PI_2);
                        changed |=
                            slider(ui, "Outer cone (rad)", outer_cone_angle, 0.0..=std::f32::consts::FRAC_PI_2);
                        changed |= slider(ui, "Attenuation", attenuation_radius, 0.0..=1000.0);
                    }
                    LightType::Directional(dir) => {
                        ui.label("Directional light");
                        ui.horizontal(|ui| {
                            ui.label("Dir");
                            changed |= drag(ui, &mut dir.x);
                            changed |= drag(ui, &mut dir.y);
                            changed |= drag(ui, &mut dir.z);
                        });
                    }
                }

                if changed {
                    node.set_light(Some(light));
                }
            });
    }
}

/// Recursively renders one scene-graph node (and its subtree) as a tree row.
fn tree_ui(
    ui: &mut egui::Ui,
    node: &SceneNode3d,
    index: usize,
    is_root: bool,
    selected: &mut Option<SceneNode3d>,
) {
    let (icon, kind) = if node.data().has_object() {
        ("◆", "object")
    } else if let Some(light) = node.light() {
        // Show the concrete light type rather than a generic "light".
        match light.light_type {
            LightType::Point { .. } => ("☀", "point light"),
            LightType::Directional(_) => ("☀", "directional light"),
            LightType::Spot { .. } => ("☀", "spot light"),
        }
    } else {
        ("▢", "group")
    };
    let label = if is_root {
        format!("{icon} scene root")
    } else {
        format!("{icon} {kind} #{index}")
    };

    // Collect child handles before recursing so no scene-node borrow is held.
    let children: Vec<SceneNode3d> = node.data().children().to_vec();
    let id = egui::Id::new(node.ptr_id());
    let mut state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, is_root);

    let header = ui.horizontal(|ui| {
        if children.is_empty() {
            ui.add_space(18.0);
        } else {
            state.show_toggle_button(ui, egui::collapsing_header::paint_default_icon);
        }

        let mut vis = node.is_visible();
        if ui.checkbox(&mut vis, "").on_hover_text("Visible").changed() {
            node.clone().set_visible(vis);
        }

        let is_sel = selected.as_ref().is_some_and(|s| s.same_node(node));
        if ui.selectable_label(is_sel, label).clicked() {
            *selected = Some(node.clone());
        }
    });

    if !children.is_empty() {
        state.show_body_indented(&header.response, ui, |ui| {
            for (i, child) in children.iter().enumerate() {
                tree_ui(ui, child, i, false, selected);
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Small widget helpers.
// ---------------------------------------------------------------------------

/// Reads a value from `node`'s object, or `None` if the node has no object.
fn obj_get<T>(node: &SceneNode3d, f: impl FnOnce(&crate::scene::Object3d) -> T) -> Option<T> {
    node.data().object().map(f)
}

fn drag(ui: &mut egui::Ui, v: &mut f32) -> bool {
    ui.add(egui::DragValue::new(v).speed(0.01)).changed()
}

fn slider(ui: &mut egui::Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>) -> bool {
    ui.add(egui::Slider::new(v, range).text(label)).changed()
}

fn color_edit(ui: &mut egui::Ui, label: &str, c: &mut Color) -> bool {
    let mut rgb = [c.r, c.g, c.b];
    let resp = ui
        .horizontal(|ui| {
            ui.label(label);
            ui.color_edit_button_rgb(&mut rgb)
        })
        .inner;
    if resp.changed() {
        *c = Color::new(rgb[0], rgb[1], rgb[2], c.a);
        true
    } else {
        false
    }
}

fn msaa_label(s: NumSamples) -> &'static str {
    match s {
        NumSamples::Zero | NumSamples::One => "Off",
        NumSamples::Four => "4×",
    }
}

fn tonemap_combo(ui: &mut egui::Ui, id: &str, tonemap: &mut Tonemap) -> bool {
    let before = *tonemap;
    egui::ComboBox::from_id_salt(id)
        .selected_text(tonemap_label(*tonemap))
        .show_ui(ui, |ui| {
            for &t in &[
                Tonemap::None,
                Tonemap::Neutral,
                Tonemap::Aces,
                Tonemap::Reinhard,
                Tonemap::AgX,
                Tonemap::TonyMcMapface,
            ] {
                ui.selectable_value(tonemap, t, tonemap_label(t));
            }
        });
    *tonemap != before
}

fn tonemap_label(t: Tonemap) -> &'static str {
    match t {
        Tonemap::None => "None",
        Tonemap::Aces => "ACES",
        Tonemap::Reinhard => "Reinhard",
        Tonemap::AgX => "AgX",
        Tonemap::Neutral => "Neutral",
        Tonemap::TonyMcMapface => "Tony McMapface",
    }
}

fn bsdf_combo(ui: &mut egui::Ui, bsdf: &mut Bsdf) -> bool {
    let before = *bsdf;
    egui::ComboBox::from_id_salt("inspector_bsdf")
        .selected_text(format!("{bsdf:?}"))
        .show_ui(ui, |ui| {
            for b in [Bsdf::Opaque, Bsdf::Glass, Bsdf::Metal, Bsdf::Emissive] {
                ui.selectable_value(bsdf, b, format!("{b:?}"));
            }
        });
    *bsdf != before
}

fn wireframe_ui(ui: &mut egui::Ui, node: &mut SceneNode3d) {
    let mut width = obj_get(node, |o| o.data().lines_width()).unwrap_or(0.0);
    let mut persp = obj_get(node, |o| o.data().lines_use_perspective()).unwrap_or(false);
    let cur = obj_get(node, |o| o.data().lines_color()).flatten();
    let mut enabled = cur.is_some();
    let mut color = cur.unwrap_or(crate::color::WHITE);

    if ui.checkbox(&mut enabled, "Wireframe").changed() {
        node.set_lines_color(if enabled { Some(color) } else { None });
    }
    if enabled {
        if color_edit(ui, "Wire color", &mut color) {
            node.set_lines_color(Some(color));
        }
        let mut changed = slider(ui, "Wire width", &mut width, 0.0..=20.0);
        changed |= ui.checkbox(&mut persp, "Wire perspective").changed();
        if changed {
            node.set_lines_width(width, persp);
        }
    }
}

fn points_ui(ui: &mut egui::Ui, node: &mut SceneNode3d) {
    let mut size = obj_get(node, |o| o.data().points_size()).unwrap_or(0.0);
    let mut persp = obj_get(node, |o| o.data().points_use_perspective()).unwrap_or(false);
    let cur = obj_get(node, |o| o.data().points_color()).flatten();
    let mut enabled = cur.is_some();
    let mut color = cur.unwrap_or(crate::color::WHITE);

    if ui.checkbox(&mut enabled, "Points").changed() {
        node.set_points_color(if enabled { Some(color) } else { None });
    }
    if enabled {
        if color_edit(ui, "Point color", &mut color) {
            node.set_points_color(Some(color));
        }
        let mut changed = slider(ui, "Point size", &mut size, 0.0..=20.0);
        changed |= ui.checkbox(&mut persp, "Point perspective").changed();
        if changed {
            node.set_points_size(size, persp);
        }
    }
}

impl Window {
    /// Draws the built-in inspector overlay for the current frame (the `egui`
    /// feature must be enabled).
    ///
    /// Call this **inside** your render loop, after the frame's render call,
    /// just like [`draw_ui`](Self::draw_ui), passing your own [`Inspector`]. It
    /// overlays a floating panel that exposes every global rendering knob
    /// (background, ambient, MSAA, shadows, HDR/tonemap, bloom) plus a tree view
    /// of the scene graph with per-node transform, material, and light editing.
    ///
    /// Pass `Some(&mut raytracer)` to also toggle and tune the GPU path tracer
    /// from the panel (bounces, samples, denoising, depth of field, environment,
    /// and an "Enable path tracer" switch — the switch is just
    /// [`RayTracer::set_enabled`], which [`raytrace_3d`](Self::raytrace_3d)
    /// honors by falling back to the rasterizer). Pass `None` when rendering with
    /// the rasterizer only.
    ///
    /// Edits are applied to the window / scene / path tracer immediately and take
    /// effect on the next rendered frame.
    ///
    /// ```no_run
    /// use kiss3d::prelude::*;
    /// use kiss3d::renderer::RayTracer;
    /// use kiss3d::window::Inspector;
    ///
    /// #[kiss3d::main]
    /// async fn main() {
    ///     let mut window = Window::new("Inspector").await;
    ///     let mut camera = OrbitCamera3d::default();
    ///     let mut scene = SceneNode3d::empty();
    ///     scene.add_cube(1.0, 1.0, 1.0);
    ///     let mut raytracer = RayTracer::new();
    ///     let mut inspector = Inspector::new();
    ///
    ///     while window.raytrace_3d(&mut scene, &mut camera, &mut raytracer).await {
    ///         window.draw_inspector(&mut inspector, &mut scene, Some(&mut raytracer));
    ///     }
    /// }
    /// ```
    pub fn draw_inspector(
        &mut self,
        inspector: &mut Inspector,
        scene: &mut SceneNode3d,
        mut raytracer: Option<&mut RayTracer>,
    ) {
        // One-time seeding of UI state from the live window.
        if !inspector.initialized {
            inspector.samples = NumSamples::from_u32(self.samples()).unwrap_or(NumSamples::Zero);
            inspector.initialized = true;
        }

        // Snapshot the global rasterizer settings the panel can edit.
        let mut settings = WinSettings {
            background: [self.background.r, self.background.g, self.background.b],
            ambient: self.ambient(),
            samples: inspector.samples,
            shadows: self.shadows_enabled(),
            shadow_res: self.shadow_resolution(),
            shadow_softness: self.shadow_softness(),
            hdr: *self.hdr_settings(),
        };

        // Snapshot the last frame's timings (from the render call that preceded
        // this `draw_inspector` in the loop) for the Timings panel.
        let timings = self.last_timings.clone();

        // Build the panel (queues egui shapes drawn by the next render call).
        // `as_deref_mut` reborrows the path tracer for the duration of the UI so
        // it can still be configured below.
        self.draw_ui(|ctx| {
            inspector.ui(
                ctx,
                scene,
                &mut settings,
                raytracer.as_deref_mut(),
                timings.as_ref(),
            )
        });

        // Push edited global settings back onto the window, only where they
        // changed (e.g. changing the sample count rebuilds the render targets).
        let bg = Color::new(
            settings.background[0],
            settings.background[1],
            settings.background[2],
            self.background.a,
        );
        if bg != self.background {
            self.set_background_color(bg);
        }
        if settings.ambient != self.ambient() {
            self.set_ambient(settings.ambient);
        }
        if settings.shadows != self.shadows_enabled() {
            self.set_shadows_enabled(settings.shadows);
        }
        if settings.shadow_res != self.shadow_resolution() {
            self.set_shadow_resolution(settings.shadow_res);
        }
        if settings.shadow_softness != self.shadow_softness() {
            self.set_shadow_softness(settings.shadow_softness);
        }
        if (settings.samples as u32).max(1) != self.samples() {
            self.set_samples(settings.samples);
        }
        inspector.samples = settings.samples;
        *self.hdr_settings_mut() = settings.hdr;

        // Apply the UI edits to the path tracer (effective on the next frame).
        if let Some(rt) = raytracer {
            if inspector.env_load_requested {
                inspector.env_load_requested = false;
                inspector.env_status =
                    if rt.set_environment_from_file(Path::new(&inspector.env_path)) {
                        "loaded".to_string()
                    } else {
                        "failed to load".to_string()
                    };
            }
            if inspector.env_clear_requested {
                inspector.env_clear_requested = false;
                rt.clear_environment();
                inspector.env_status = "procedural sky".to_string();
            }

            let knobs = inspector.rt;
            inspector.applied_env = knobs.apply(rt, inspector.applied_env);
        }
    }
}
