//! Small egui widget helpers and scene-graph accessors shared by the inspector
//! tabs.

use crate::color::Color;
use crate::post_processing::Tonemap;
use crate::renderer::DepthOfFieldMode;
use crate::scene::{Bsdf, Object2d, Object3d, SceneNode2d, SceneNode3d};
use crate::window::NumSamples;

/// Whether a 3D scene root is empty (no object and no children) — used to hide its
/// tab. A scene with lights or shapes has child nodes, so it is not empty.
pub(super) fn node_is_empty(node: &SceneNode3d) -> bool {
    let d = node.data();
    !d.has_object() && d.children().is_empty()
}

/// 2D analogue of [`node_is_empty`].
pub(super) fn node_is_empty_2d(node: &SceneNode2d) -> bool {
    let d = node.data();
    !d.has_object() && d.children().is_empty()
}

/// Reads a value from `node`'s object, or `None` if the node has no object.
pub(super) fn obj_get<T>(node: &SceneNode3d, f: impl FnOnce(&Object3d) -> T) -> Option<T> {
    node.data().object().map(f)
}

/// Reads a representative value: from `node`'s own object, or — when `recursive`
/// (a group) — from the first object found anywhere in its subtree. `None` when no
/// object exists. Mirrors the read side of the recursive setters used to apply
/// edits to a whole group.
pub(super) fn mat_get<T>(
    node: &SceneNode3d,
    recursive: bool,
    f: impl Fn(&Object3d) -> T,
) -> Option<T> {
    if recursive {
        let mut out = None;
        node.apply_to_objects_recursive(&mut |o| {
            if out.is_none() {
                out = Some(f(o));
            }
        });
        out
    } else {
        node.data().object().map(f)
    }
}

/// Applies an object mutation to `node`'s own object, or — when `recursive` (a
/// group) — to every object in its subtree (the `_recursive` application path).
pub(super) fn apply3d(node: &SceneNode3d, recursive: bool, mut f: impl FnMut(&mut Object3d)) {
    let mut n = node.clone();
    if recursive {
        n.apply_to_objects_mut_recursive(&mut f);
    } else {
        n.apply_to_object_mut(&mut f);
    }
}

/// 2D analogue of [`mat_get`].
pub(super) fn mat_get_2d<T>(
    node: &SceneNode2d,
    recursive: bool,
    f: impl Fn(&Object2d) -> T,
) -> Option<T> {
    if recursive {
        let mut out = None;
        node.apply_to_objects_recursive(&mut |o| {
            if out.is_none() {
                out = Some(f(o));
            }
        });
        out
    } else {
        node.data().object().map(f)
    }
}

/// 2D analogue of [`apply3d`].
pub(super) fn apply2d(node: &SceneNode2d, recursive: bool, mut f: impl FnMut(&mut Object2d)) {
    let mut n = node.clone();
    if recursive {
        n.apply_to_objects_mut_recursive(&mut f);
    } else {
        n.apply_to_object_mut(&mut f);
    }
}

pub(super) fn dof_mode_combo(ui: &mut egui::Ui, mode: &mut DepthOfFieldMode) {
    egui::ComboBox::from_id_salt("inspector_dof_mode")
        .selected_text(match mode {
            DepthOfFieldMode::Bokeh => "Bokeh",
            DepthOfFieldMode::Gaussian => "Gaussian",
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(mode, DepthOfFieldMode::Bokeh, "Bokeh");
            ui.selectable_value(mode, DepthOfFieldMode::Gaussian, "Gaussian");
        });
}

pub(super) fn drag(ui: &mut egui::Ui, v: &mut f32) -> bool {
    ui.add(egui::DragValue::new(v).speed(0.01)).changed()
}

pub(super) fn slider(
    ui: &mut egui::Ui,
    label: &str,
    v: &mut f32,
    range: std::ops::RangeInclusive<f32>,
) -> bool {
    ui.add(egui::Slider::new(v, range).text(label)).changed()
}

pub(super) fn color_edit(ui: &mut egui::Ui, label: &str, c: &mut Color) -> bool {
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

pub(super) fn msaa_label(s: NumSamples) -> &'static str {
    match s {
        NumSamples::Zero | NumSamples::One => "Off",
        NumSamples::Four => "4×",
    }
}

pub(super) fn tonemap_combo(ui: &mut egui::Ui, id: &str, tonemap: &mut Tonemap) -> bool {
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

pub(super) fn bsdf_combo(ui: &mut egui::Ui, bsdf: &mut Bsdf) -> bool {
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

/// Opens a native file-open dialog filtered to image files and returns the chosen
/// path. On wasm there is no synchronous dialog, so this is absent and the
/// inspector falls back to its path text field.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn pick_image_path() -> Option<String> {
    rfd::FileDialog::new()
        .add_filter(
            "Images",
            &[
                "png", "jpg", "jpeg", "bmp", "tga", "tiff", "tif", "webp", "hdr", "exr", "gif",
                "dds", "ktx2",
            ],
        )
        .pick_file()
        .map(|p| p.to_string_lossy().into_owned())
}
