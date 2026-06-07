//! The "2D scene" inspector tab: the 2D scene-graph tree and the selected node's
//! editor (transform, material and base-color texture), with whole-subtree editing
//! when a group is selected.

use crate::scene::SceneNode2d;

use super::widgets::*;
use super::Inspector;

impl Inspector {
    pub(super) fn scene_tree_section_2d(&mut self, ui: &mut egui::Ui, scene: &mut SceneNode2d) {
        egui::CollapsingHeader::new("2D scene tree")
            .default_open(true)
            .show(ui, |ui| {
                if ui.button("Clear selection").clicked() {
                    self.selected_2d = None;
                }
                tree_ui_2d(ui, scene, 0, true, &mut self.selected_2d);
            });
    }

    pub(super) fn selection_section_2d(&mut self, ui: &mut egui::Ui) {
        let Some(node) = self.selected_2d.clone() else {
            ui.label("Select a node in the 2D tree to edit it.");
            return;
        };

        // Borrow the texture-loader scratch fields up front (the closure below must
        // not capture all of `self`).
        let tex_path = &mut self.tex_path;
        let tex_status = &mut self.tex_status;
        egui::CollapsingHeader::new("2D selection")
            .default_open(true)
            .show(ui, |ui| {
                transform_ui_2d(ui, &node);

                let has_object = node.data().has_object();
                let recursive = !has_object;
                if has_object || mat_get_2d(&node, true, |_| ()).is_some() {
                    material_ui_2d(ui, &node, recursive);
                    texture_ui_2d(ui, &node, recursive, tex_path, tex_status);
                }
                if !node.data().children().is_empty() {
                    egui::CollapsingHeader::new("Subtree")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                if ui.button("Show subtree").clicked() {
                                    node.clone().apply_to_scene_nodes_mut_recursive(&mut |n| {
                                        n.set_visible(true);
                                    });
                                }
                                if ui.button("Hide subtree").clicked() {
                                    node.clone().apply_to_scene_nodes_mut_recursive(&mut |n| {
                                        n.set_visible(false);
                                    });
                                }
                            });
                        });
                }
            });
    }
}

/// 2D analogue of [`tree_ui`](super::scene3d) (no lights in 2D).
fn tree_ui_2d(
    ui: &mut egui::Ui,
    node: &SceneNode2d,
    index: usize,
    is_root: bool,
    selected: &mut Option<SceneNode2d>,
) {
    let (icon, kind) = if node.data().has_object() {
        ("◆", "object")
    } else {
        ("▢", "group")
    };
    let label = if is_root {
        format!("{icon} 2D scene root")
    } else {
        format!("{icon} {kind} #{index}")
    };

    let children: Vec<SceneNode2d> = node.data().children().to_vec();
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
                tree_ui_2d(ui, child, i, false, selected);
            }
        });
    }
}

fn transform_ui_2d(ui: &mut egui::Ui, node: &SceneNode2d) {
    let mut node = node.clone();
    ui.label("Transform");

    let mut pos = node.position();
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Pos");
        changed |= drag(ui, &mut pos.x);
        changed |= drag(ui, &mut pos.y);
    });
    if changed {
        node.set_position(pos);
    }

    let mut scale = node.local_scale();
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Scale");
        changed |= drag(ui, &mut scale.x);
        changed |= drag(ui, &mut scale.y);
    });
    if changed {
        node.set_local_scale(scale.x, scale.y);
    }

    // 2D rotation is a single angle; reading it back round-trips cleanly, so no
    // editing buffer is needed (unlike the 3D Euler case).
    let mut deg = node.rotation().angle().to_degrees();
    let changed = ui
        .horizontal(|ui| {
            ui.label("Rot°");
            drag(ui, &mut deg)
        })
        .inner;
    if changed {
        node.set_rotation(deg.to_radians());
    }
}

fn material_ui_2d(ui: &mut egui::Ui, node: &SceneNode2d, recursive: bool) {
    let header = if recursive {
        "Material (subtree)"
    } else {
        "Material"
    };
    egui::CollapsingHeader::new(header)
        .default_open(true)
        .show(ui, |ui| {
            if let Some(mut c) = mat_get_2d(node, recursive, |o| o.data().color()) {
                let mut changed = color_edit(ui, "Color", &mut c);
                changed |= slider(ui, "Opacity", &mut c.a, 0.0..=1.0);
                if changed {
                    apply2d(node, recursive, |o| o.set_color(c));
                }
            }

            ui.separator();
            if let Some(mut on) =
                mat_get_2d(node, recursive, |o| o.data().surface_rendering_active())
            {
                if ui.checkbox(&mut on, "Draw surface").changed() {
                    apply2d(node, recursive, |o| o.set_surface_rendering_activation(on));
                }
            }
            if let Some(mut on) =
                mat_get_2d(node, recursive, |o| o.data().backface_culling_enabled())
            {
                if ui.checkbox(&mut on, "Backface culling").changed() {
                    apply2d(node, recursive, |o| o.enable_backface_culling(on));
                }
            }

            wireframe_ui_2d(ui, node, recursive);
            points_ui_2d(ui, node, recursive);
        });
}

fn wireframe_ui_2d(ui: &mut egui::Ui, node: &SceneNode2d, recursive: bool) {
    let mut width = mat_get_2d(node, recursive, |o| o.data().lines_width()).unwrap_or(0.0);
    let mut persp =
        mat_get_2d(node, recursive, |o| o.data().lines_use_perspective()).unwrap_or(false);
    let cur = mat_get_2d(node, recursive, |o| o.data().lines_color()).flatten();
    let mut enabled = cur.is_some();
    let mut color = cur.unwrap_or(crate::color::WHITE);

    if ui.checkbox(&mut enabled, "Wireframe").changed() {
        let c = enabled.then_some(color);
        apply2d(node, recursive, |o| o.set_lines_color(c));
    }
    if enabled {
        if color_edit(ui, "Wire color", &mut color) {
            apply2d(node, recursive, |o| o.set_lines_color(Some(color)));
        }
        let mut changed = slider(ui, "Wire width", &mut width, 0.0..=20.0);
        changed |= ui.checkbox(&mut persp, "Wire perspective").changed();
        if changed {
            apply2d(node, recursive, |o| o.set_lines_width(width, persp));
        }
    }
}

fn points_ui_2d(ui: &mut egui::Ui, node: &SceneNode2d, recursive: bool) {
    let mut size = mat_get_2d(node, recursive, |o| o.data().points_size()).unwrap_or(0.0);
    let mut persp =
        mat_get_2d(node, recursive, |o| o.data().points_use_perspective()).unwrap_or(false);
    let cur = mat_get_2d(node, recursive, |o| o.data().points_color()).flatten();
    let mut enabled = cur.is_some();
    let mut color = cur.unwrap_or(crate::color::WHITE);

    if ui.checkbox(&mut enabled, "Points").changed() {
        let c = enabled.then_some(color);
        apply2d(node, recursive, |o| o.set_points_color(c));
    }
    if enabled {
        if color_edit(ui, "Point color", &mut color) {
            apply2d(node, recursive, |o| o.set_points_color(Some(color)));
        }
        let mut changed = slider(ui, "Point size", &mut size, 0.0..=20.0);
        changed |= ui.checkbox(&mut persp, "Point perspective").changed();
        if changed {
            apply2d(node, recursive, |o| o.set_points_size(size, persp));
        }
    }
}

/// Base-color texture loader for the selected 2D object/group.
fn texture_ui_2d(
    ui: &mut egui::Ui,
    node: &SceneNode2d,
    recursive: bool,
    path: &mut String,
    status: &mut String,
) {
    ui.separator();
    ui.label("Texture");
    ui.horizontal(|ui| {
        ui.label("Image");
        ui.text_edit_singleline(path);
    });
    ui.horizontal(|ui| {
        #[cfg(not(target_arch = "wasm32"))]
        if ui.button("Open…").clicked() {
            if let Some(p) = pick_image_path() {
                *status = if apply_texture_2d(node, recursive, &p) {
                    format!("loaded {p}")
                } else {
                    format!("not found: {p}")
                };
                *path = p;
            }
        }
        if ui.button("Load").clicked() {
            let p = path.trim().to_string();
            *status = if apply_texture_2d(node, recursive, &p) {
                format!("loaded {p}")
            } else {
                format!("not found: {p}")
            };
        }
    });
    if !status.is_empty() {
        ui.label(status.as_str());
    }
}

/// Loads `p` as the base-color texture of a 2D `node` (recursively for a group).
fn apply_texture_2d(node: &SceneNode2d, recursive: bool, p: &str) -> bool {
    if p.is_empty() || !std::path::Path::new(p).is_file() {
        return false;
    }
    apply2d(node, recursive, |o| {
        o.set_texture_from_file(std::path::Path::new(p), p)
    });
    true
}
