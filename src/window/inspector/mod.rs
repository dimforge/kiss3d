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
//!         // `Some(&mut raytracer)` for a rasterizer-only panel, and an optional
//!         // 2D scene as the third argument.
//!         window.draw_inspector(&mut inspector, Some(&mut scene), None, Some(&mut raytracer));
//!     }
//! }
//! ```
//!
//! The UI is organised into one module per top-level tab — [`global`], [`scene3d`]
//! and [`scene2d`] — with shared egui/scene helpers in [`widgets`]. This module
//! holds the [`Inspector`] state, the tab dispatcher, and [`Window::draw_inspector`].

mod global;
mod scene2d;
mod scene3d;
mod widgets;

use std::path::Path;

use glamx::Vec3;

use crate::color::Color;
use crate::post_processing::HdrSettings;
use crate::renderer::{DofSettings, RayTracer, RenderTimings, SsaoSettings, SsrSettings};
use crate::scene::{SceneNode2d, SceneNode3d};
use crate::window::NumSamples;

use self::widgets::{node_is_empty, node_is_empty_2d};
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
        // Depth of field is shared with the rasterizer (driven by the window's DoF
        // settings in `raytrace_3d_frame`), so it is not a separate path-tracer knob.

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
    /// Whether vsync is enabled (off = uncapped presentation; pairs with the
    /// wall-clock frame time in the Timings panel for GPU-bound measurement).
    vsync: bool,
}

/// The inspector's top-level tab: global rendering settings, the 3D scene tree, or
/// the 2D scene tree. The scene tabs are only shown when their scene is present and
/// non-empty; see [`Inspector::set_tab`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InspectorTab {
    /// Renderer, common, rasterizer / path-tracer and timing settings.
    Global,
    /// The 3D scene tree and the selected node's editor.
    Scene3d,
    /// The 2D scene tree and the selected node's editor.
    Scene2d,
}

/// Which image map of the selected object the inspector's loader writes to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MapTarget {
    BaseColor,
    Normal,
    MetallicRoughness,
    Ao,
    Emissive,
    Height,
}

impl MapTarget {
    fn label(self) -> &'static str {
        match self {
            MapTarget::BaseColor => "Base color",
            MapTarget::Normal => "Normal",
            MapTarget::MetallicRoughness => "Metallic / roughness",
            MapTarget::Ao => "Ambient occlusion",
            MapTarget::Emissive => "Emissive",
            MapTarget::Height => "Height (relief)",
        }
    }

    const ALL: [MapTarget; 6] = [
        MapTarget::BaseColor,
        MapTarget::Normal,
        MapTarget::MetallicRoughness,
        MapTarget::Ao,
        MapTarget::Emissive,
        MapTarget::Height,
    ];
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
    /// Active top-level tab.
    tab: InspectorTab,
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
    /// Rasterizer post-process effects, persisted across frames (like [`RtKnobs`])
    /// and pushed onto the window each frame; their settings are kept here so the
    /// sliders keep their values even while the effect is toggled off.
    ssao_enabled: bool,
    ssao: SsaoSettings,
    ssr_enabled: bool,
    ssr: SsrSettings,
    dof_enabled: bool,
    dof: DofSettings,
    /// Image-file path + status + apply flags for the equirectangular skybox loader.
    skybox_path: String,
    skybox_status: String,
    skybox_load_requested: bool,
    skybox_clear_requested: bool,
    skybox_rotation_deg: f32,
    skybox_intensity: f32,
    /// Set once the user touches a skybox orientation slider, so opening the
    /// inspector doesn't reset an orientation set in code.
    skybox_orient_dirty: bool,
    /// Scratch image-path buffer + status + target for the object map loader.
    tex_path: String,
    tex_status: String,
    tex_target: MapTarget,
    /// Currently selected 3D node (a cheap `Rc` handle).
    selected: Option<SceneNode3d>,
    /// Currently selected 2D node.
    selected_2d: Option<SceneNode2d>,
    /// Euler-angle editing buffer `(node id, degrees)` for the selected node, so
    /// the rotation sliders don't jitter through quaternion round-tripping.
    edit_rot: Option<(u64, Vec3)>,
}

impl Default for Inspector {
    fn default() -> Self {
        Inspector {
            open: true,
            tab: InspectorTab::Global,
            initialized: false,
            rt: RtKnobs::default(),
            applied_env: None,
            env_path: String::new(),
            env_status: String::new(),
            env_load_requested: false,
            env_clear_requested: false,
            samples: NumSamples::One,
            ssao_enabled: false,
            ssao: SsaoSettings::default(),
            ssr_enabled: false,
            ssr: SsrSettings::default(),
            dof_enabled: false,
            dof: DofSettings::default(),
            skybox_path: String::new(),
            skybox_status: String::new(),
            skybox_load_requested: false,
            skybox_clear_requested: false,
            skybox_rotation_deg: 0.0,
            skybox_intensity: 1.0,
            skybox_orient_dirty: false,
            tex_path: String::new(),
            tex_status: String::new(),
            tex_target: MapTarget::BaseColor,
            selected: None,
            selected_2d: None,
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

    /// The currently active top-level tab.
    pub fn tab(&self) -> InspectorTab {
        self.tab
    }

    /// Switches the active top-level tab. A scene tab whose scene is absent or
    /// empty is hidden, and selecting it falls back to [`InspectorTab::Global`] on
    /// the next frame.
    pub fn set_tab(&mut self, tab: InspectorTab) {
        self.tab = tab;
    }

    /// The 3D node currently selected in the scene tree, if any.
    pub fn selected(&self) -> Option<&SceneNode3d> {
        self.selected.as_ref()
    }

    /// Programmatically selects a 3D node in the scene tree (the same state a
    /// click in the tree sets), or clears the selection with `None`. Useful to
    /// focus the inspector on an object you just created.
    pub fn select(&mut self, node: Option<SceneNode3d>) {
        self.selected = node;
    }

    /// The 2D node currently selected in the 2D scene tree, if any.
    pub fn selected_2d(&self) -> Option<&SceneNode2d> {
        self.selected_2d.as_ref()
    }

    /// Programmatically selects a node in the 2D scene tree, or clears it.
    pub fn select_2d(&mut self, node: Option<SceneNode2d>) {
        self.selected_2d = node;
    }

    /// Builds the whole panel for one frame.
    fn ui(
        &mut self,
        ctx: &egui::Context,
        scene: Option<&mut SceneNode3d>,
        scene_2d: Option<&mut SceneNode2d>,
        win: &mut WinSettings,
        mut raytracer: Option<&mut RayTracer>,
        timings: Option<&RenderTimings>,
    ) {
        if !self.open {
            return;
        }

        // The scene tabs are shown only for a scene that is present and non-empty
        // (an empty root has no object and no children — e.g. a 3D-only app passes
        // no 2D scene, or passes one it never populated).
        let show_3d = scene.as_deref().is_some_and(|s| !node_is_empty(s));
        let show_2d = scene_2d.as_deref().is_some_and(|s| !node_is_empty_2d(s));

        // If the active tab is a scene tab that is no longer available, fall back
        // to the always-present Global tab.
        if (self.tab == InspectorTab::Scene3d && !show_3d)
            || (self.tab == InspectorTab::Scene2d && !show_2d)
        {
            self.tab = InspectorTab::Global;
        }

        // Path-tracing state (for the 3D material editor's BSDF section); read
        // before the renderer toggle is edited below.
        let path_tracing = raytracer.as_deref().is_some_and(|rt| rt.enabled());

        // No `[x]` button: the window is always available (collapse it from the
        // title bar). Use `Inspector::set_open` to hide it programmatically.
        egui::Window::new("🛠 kiss3d inspector")
            .default_width(320.0)
            .default_pos([12.0, 12.0])
            .show(ctx, |ui| {
                // Tab bar. "Global" is always present; the scene tabs appear only
                // when their scene is present and non-empty.
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.tab, InspectorTab::Global, "Global");
                    if show_3d {
                        ui.selectable_value(&mut self.tab, InspectorTab::Scene3d, "3D scene");
                    }
                    if show_2d {
                        ui.selectable_value(&mut self.tab, InspectorTab::Scene2d, "2D scene");
                    }
                });
                ui.separator();

                egui::ScrollArea::vertical()
                    .max_height(ui.ctx().content_rect().height() - 60.0)
                    .show(ui, |ui| match self.tab {
                        InspectorTab::Global => {
                            self.renderer_section(ui, raytracer.as_deref_mut());

                            // Re-read after the toggle above so the renderer-specific
                            // section below reflects the change this same frame.
                            let pt = raytracer.as_deref().is_some_and(|rt| rt.enabled());

                            self.common_section(ui, &mut win.hdr, &mut win.vsync);

                            if pt {
                                self.path_tracer_section(ui);
                            } else {
                                self.rasterizer_section(ui, win);
                            }

                            Self::timings_section(ui, timings);
                        }
                        InspectorTab::Scene3d => {
                            if let Some(scene) = scene {
                                self.scene_tree_section(ui, scene);
                                ui.separator();
                                self.selection_section(ui, path_tracing);
                            }
                        }
                        InspectorTab::Scene2d => {
                            if let Some(scene_2d) = scene_2d {
                                self.scene_tree_section_2d(ui, scene_2d);
                                ui.separator();
                                self.selection_section_2d(ui);
                            }
                        }
                    });
            });
    }
}

impl Window {
    /// Draws the built-in inspector overlay for the current frame (the `egui`
    /// feature must be enabled).
    ///
    /// Call this **inside** your render loop, after the frame's render call,
    /// just like [`draw_ui`](Self::draw_ui), passing your own [`Inspector`]. It
    /// overlays a floating panel that exposes every global rendering knob
    /// (background, ambient, MSAA, shadows, HDR/tonemap, bloom, SSAO, SSR, depth of
    /// field) plus a tree view of the scene graph with per-node transform,
    /// material, and light editing — including per-object SSR and reflectors, and
    /// whole-subtree editing when a group is selected.
    ///
    /// Both the 3D scene and a 2D scene are optional: pass `Some` for the ones you
    /// render so the panel shows a tree for each (a pure-2D app passes `None` for
    /// the 3D scene).
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
    ///         window.draw_inspector(&mut inspector, Some(&mut scene), None, Some(&mut raytracer));
    ///     }
    /// }
    /// ```
    pub fn draw_inspector(
        &mut self,
        inspector: &mut Inspector,
        scene: Option<&mut SceneNode3d>,
        scene_2d: Option<&mut SceneNode2d>,
        mut raytracer: Option<&mut RayTracer>,
    ) {
        // One-time seeding of UI state from the live window.
        if !inspector.initialized {
            inspector.samples = NumSamples::from_u32(self.samples()).unwrap_or(NumSamples::One);
            inspector.ssao_enabled = self.ssao_enabled();
            inspector.ssr_enabled = self.ssr_enabled();
            inspector.dof_enabled = self.dof_enabled();
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
            vsync: self.vsync(),
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
                scene_2d,
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
        if settings.vsync != self.vsync() {
            self.set_vsync(settings.vsync);
        }
        if (settings.samples as u32).max(1) != self.samples() {
            self.set_samples(settings.samples);
        }
        inspector.samples = settings.samples;
        *self.hdr_settings_mut() = settings.hdr;

        // Apply the screen-space post-process effects. The settings live in the
        // inspector (so they persist while toggled off); they are only pushed —
        // which lazily allocates the effect's GPU targets — while the effect is on.
        self.set_ssao_enabled(inspector.ssao_enabled);
        if inspector.ssao_enabled {
            *self.ssao_settings_mut() = inspector.ssao;
        }
        self.set_ssr_enabled(inspector.ssr_enabled);
        if inspector.ssr_enabled {
            *self.ssr_settings_mut() = inspector.ssr;
        }
        self.set_dof_enabled(inspector.dof_enabled);
        if inspector.dof_enabled {
            *self.dof_settings_mut() = inspector.dof;
        }

        // Apply the skybox loader. Loading/clearing happens only on the frame the
        // button was clicked; the orientation is pushed only once the user has
        // touched a slider (so opening the inspector never resets an orientation
        // set in code).
        if inspector.skybox_load_requested {
            inspector.skybox_load_requested = false;
            let path = inspector.skybox_path.trim().to_string();
            inspector.skybox_status =
                if !path.is_empty() && self.set_skybox_from_file(Path::new(&path)) {
                    format!("loaded {path}")
                } else {
                    "failed to load".to_string()
                };
        }
        if inspector.skybox_clear_requested {
            inspector.skybox_clear_requested = false;
            self.clear_skybox();
            inspector.skybox_status = "cleared".to_string();
        }
        if inspector.skybox_orient_dirty && self.has_skybox() {
            self.set_skybox_orientation(
                inspector.skybox_rotation_deg.to_radians(),
                inspector.skybox_intensity,
            );
        }

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
