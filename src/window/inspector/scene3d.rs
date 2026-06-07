//! The "3D scene" inspector tab: the scene-graph tree and the selected node's
//! editor — transform, material (including per-object SSR, reflector and the
//! image-map loader), and light — with whole-subtree editing when a group is
//! selected.

use glamx::{EulerRot, Quat, Vec3};

use crate::light::LightType;
use crate::renderer::{Reflector, SsrMaterial};
use crate::scene::{ParallaxMethod, SceneNode3d};

use super::widgets::*;
use super::{Inspector, MapTarget};

impl Inspector {
    pub(super) fn scene_tree_section(&mut self, ui: &mut egui::Ui, scene: &mut SceneNode3d) {
        egui::CollapsingHeader::new("Scene tree")
            .default_open(true)
            .show(ui, |ui| {
                if ui.button("Clear selection").clicked() {
                    self.selected = None;
                }
                tree_ui(ui, scene, 0, true, &mut self.selected);
            });
    }

    pub(super) fn selection_section(&mut self, ui: &mut egui::Ui, path_tracing: bool) {
        let Some(node) = self.selected.clone() else {
            ui.label("Select a node in the tree to edit it.");
            return;
        };

        egui::CollapsingHeader::new("Selection")
            .default_open(true)
            .show(ui, |ui| {
                self.transform_ui(ui, &node);

                // A node that holds an object is edited directly; a group (no own
                // object) edits its whole subtree — every descendant object — via
                // the recursive setters, so selecting a group behaves like selecting
                // an object but applies to all of it.
                let has_object = node.data().has_object();
                let recursive = !has_object;
                if has_object || mat_get(&node, true, |_| ()).is_some() {
                    self.material_ui(ui, &node, path_tracing, recursive);
                }

                if node.data().has_light() {
                    self.light_ui(ui, &node);
                }
                if !node.data().children().is_empty() {
                    Self::subtree_ui(ui, &node);
                }
            });
    }

    /// Whole-subtree visibility toggles for a group (material edits are handled by
    /// [`material_ui`](Self::material_ui) in recursive mode).
    fn subtree_ui(ui: &mut egui::Ui, node: &SceneNode3d) {
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

    /// Material editor for a 3D node. When `recursive` (the node is a group) every
    /// edit applies to the whole subtree via the recursive object visitor, and the
    /// widgets seed from the first object found in the subtree.
    fn material_ui(
        &mut self,
        ui: &mut egui::Ui,
        node: &SceneNode3d,
        path_tracing: bool,
        recursive: bool,
    ) {
        let header = if recursive {
            "Material (subtree)"
        } else {
            "Material"
        };
        // Borrow the image-loader scratch fields up front so the closure below does
        // not need to capture all of `self`.
        let tex_path = &mut self.tex_path;
        let tex_target = &mut self.tex_target;
        let tex_status = &mut self.tex_status;
        egui::CollapsingHeader::new(header)
            .default_open(true)
            .show(ui, |ui| {
                // Base color and its opacity (the color's alpha; below 1.0 makes
                // the object transparent).
                if let Some(mut c) = mat_get(node, recursive, |o| o.data().color()) {
                    let mut changed = color_edit(ui, "Color", &mut c);
                    changed |= slider(ui, "Opacity", &mut c.a, 0.0..=1.0);
                    if changed {
                        apply3d(node, recursive, |o| o.set_color(c));
                    }
                }
                if let Some(mut c) = mat_get(node, recursive, |o| o.data().emissive()) {
                    if color_edit(ui, "Emissive", &mut c) {
                        apply3d(node, recursive, |o| o.set_emissive(c));
                    }
                }

                // PBR scalars.
                if let Some(mut v) = mat_get(node, recursive, |o| o.data().metallic()) {
                    if slider(ui, "Metallic", &mut v, 0.0..=1.0) {
                        apply3d(node, recursive, |o| o.set_metallic(v));
                    }
                }
                if let Some(mut v) = mat_get(node, recursive, |o| o.data().roughness()) {
                    if slider(ui, "Roughness", &mut v, 0.0..=1.0) {
                        apply3d(node, recursive, |o| o.set_roughness(v));
                    }
                }

                // BSDF and its parameters only affect the path tracer, so only
                // show them when path tracing is the active renderer.
                if path_tracing {
                    ui.separator();
                    ui.label("Path-tracer BSDF");
                    if let Some(mut bsdf) = mat_get(node, recursive, |o| o.data().bsdf()) {
                        if bsdf_combo(ui, &mut bsdf) {
                            apply3d(node, recursive, |o| o.set_bsdf(bsdf));
                        }
                    }
                    if let Some(mut v) = mat_get(node, recursive, |o| o.data().ior()) {
                        if slider(ui, "IOR", &mut v, 1.0..=3.0) {
                            apply3d(node, recursive, |o| o.set_ior(v));
                        }
                    }
                    if let Some(mut v) = mat_get(node, recursive, |o| o.data().transmission()) {
                        if slider(ui, "Transmission", &mut v, 0.0..=1.0) {
                            apply3d(node, recursive, |o| o.set_transmission(v));
                        }
                    }
                    if let Some(mut c) = mat_get(node, recursive, |o| o.data().specular_tint()) {
                        if color_edit(ui, "Specular tint", &mut c) {
                            apply3d(node, recursive, |o| o.set_specular_tint(c));
                        }
                    }
                    let sub = mat_get(node, recursive, |o| {
                        (o.data().subsurface(), o.data().subsurface_radius())
                    });
                    if let Some((mut factor, mut radius)) = sub {
                        let mut changed = slider(ui, "Subsurface", &mut factor, 0.0..=1.0);
                        changed |= slider(ui, "SSS radius", &mut radius, 0.0..=5.0);
                        if changed {
                            apply3d(node, recursive, |o| o.set_subsurface(factor, radius));
                        }
                    }
                }

                maps_ui(ui, node, recursive, tex_path, tex_target, tex_status);

                Self::ssr_ui(ui, node, recursive);
                // A reflector is a heavy per-object GPU mirror; only offer it for a
                // single object, not as a bulk subtree edit.
                if !recursive {
                    Self::reflector_ui(ui, node);
                }

                ui.separator();
                // Surface / wireframe / points.
                if let Some(mut on) =
                    mat_get(node, recursive, |o| o.data().surface_rendering_active())
                {
                    if ui.checkbox(&mut on, "Draw surface").changed() {
                        apply3d(node, recursive, |o| o.set_surface_rendering_activation(on));
                    }
                }
                if let Some(mut on) =
                    mat_get(node, recursive, |o| o.data().backface_culling_enabled())
                {
                    if ui.checkbox(&mut on, "Backface culling").changed() {
                        apply3d(node, recursive, |o| o.enable_backface_culling(on));
                    }
                }

                wireframe_ui(ui, node, recursive);
                points_ui(ui, node, recursive);

                if let Some(mut id) = mat_get(node, recursive, |o| o.data().segmentation_id()) {
                    ui.horizontal(|ui| {
                        ui.label("Segmentation id");
                        if ui.add(egui::DragValue::new(&mut id)).changed() {
                            apply3d(node, recursive, |o| o.set_segmentation_id(id));
                        }
                    });
                }
            });
    }

    /// Per-object screen-space-reflection controls ([`SsrMaterial`]). Only takes
    /// effect when window SSR is enabled.
    fn ssr_ui(ui: &mut egui::Ui, node: &SceneNode3d, recursive: bool) {
        let Some(cur) = mat_get(node, recursive, |o| o.data().ssr()) else {
            return;
        };
        ui.separator();
        let mut enabled = cur.is_some();
        if ui
            .checkbox(&mut enabled, "Screen-space reflections")
            .on_hover_text("Per-object; needs window SSR enabled")
            .changed()
        {
            let v = enabled.then(SsrMaterial::default);
            apply3d(node, recursive, |o| o.set_ssr(v));
        }
        if let Some(mut m) = cur {
            let mut changed = slider(ui, "SSR intensity", &mut m.intensity, 0.0..=2.0);
            changed |= ui
                .checkbox(&mut m.infinite_thick, "Infinite thickness")
                .changed();
            changed |= ui
                .checkbox(&mut m.distance_attenuation, "Distance attenuation")
                .changed();
            changed |= ui.checkbox(&mut m.fresnel, "Fresnel boost").changed();
            if changed {
                apply3d(node, recursive, |o| o.set_ssr(Some(m)));
            }
        }
    }

    /// Per-object planar reflector (mirror) controls.
    fn reflector_ui(ui: &mut egui::Ui, node: &SceneNode3d) {
        let has = obj_get(node, |o| o.reflector().is_some()).unwrap_or(false);
        // No object at all → nothing to do.
        if obj_get(node, |_| ()).is_none() {
            return;
        }
        ui.separator();
        let mut on = has;
        if ui
            .checkbox(&mut on, "Planar reflector (mirror)")
            .on_hover_text("Best on a flat surface (e.g. a quad)")
            .changed()
        {
            let mut n = node.clone();
            n.set_reflector(on.then(Reflector::new));
        }
        if has {
            let mut intensity = obj_get(node, |o| o.reflector().map(|r| r.intensity()))
                .flatten()
                .unwrap_or(1.0);
            if slider(ui, "Reflection intensity", &mut intensity, 0.0..=1.0) {
                node.clone().set_reflector_intensity(intensity);
            }
            let mut falloff = obj_get(node, |o| o.reflector().map(|r| r.normal_falloff()))
                .flatten()
                .unwrap_or(0.0);
            if slider(ui, "Normal falloff", &mut falloff, 0.0..=8.0) {
                node.clone().set_reflector_normal_falloff(falloff);
            }
            let mut n = obj_get(node, |o| o.reflector().map(|r| r.local_normal()))
                .flatten()
                .unwrap_or(Vec3::Z);
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.label("Plane normal");
                changed |= drag(ui, &mut n.x);
                changed |= drag(ui, &mut n.y);
                changed |= drag(ui, &mut n.z);
            });
            if changed && n.length_squared() > 1e-6 {
                node.clone().set_reflector_normal(n);
            }
        }
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
                        changed |= slider(
                            ui,
                            "Inner cone (rad)",
                            inner_cone_angle,
                            0.0..=std::f32::consts::FRAC_PI_2,
                        );
                        changed |= slider(
                            ui,
                            "Outer cone (rad)",
                            outer_cone_angle,
                            0.0..=std::f32::consts::FRAC_PI_2,
                        );
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

fn wireframe_ui(ui: &mut egui::Ui, node: &SceneNode3d, recursive: bool) {
    let mut width = mat_get(node, recursive, |o| o.data().lines_width()).unwrap_or(0.0);
    let mut persp = mat_get(node, recursive, |o| o.data().lines_use_perspective()).unwrap_or(false);
    let cur = mat_get(node, recursive, |o| o.data().lines_color()).flatten();
    let mut enabled = cur.is_some();
    let mut color = cur.unwrap_or(crate::color::WHITE);

    if ui.checkbox(&mut enabled, "Wireframe").changed() {
        let c = enabled.then_some(color);
        apply3d(node, recursive, |o| o.set_lines_color(c));
    }
    if enabled {
        if color_edit(ui, "Wire color", &mut color) {
            apply3d(node, recursive, |o| o.set_lines_color(Some(color)));
        }
        let mut changed = slider(ui, "Wire width", &mut width, 0.0..=20.0);
        changed |= ui.checkbox(&mut persp, "Wire perspective").changed();
        if changed {
            apply3d(node, recursive, |o| o.set_lines_width(width, persp));
        }
    }
}

fn points_ui(ui: &mut egui::Ui, node: &SceneNode3d, recursive: bool) {
    let mut size = mat_get(node, recursive, |o| o.data().points_size()).unwrap_or(0.0);
    let mut persp =
        mat_get(node, recursive, |o| o.data().points_use_perspective()).unwrap_or(false);
    let cur = mat_get(node, recursive, |o| o.data().points_color()).flatten();
    let mut enabled = cur.is_some();
    let mut color = cur.unwrap_or(crate::color::WHITE);

    if ui.checkbox(&mut enabled, "Points").changed() {
        let c = enabled.then_some(color);
        apply3d(node, recursive, |o| o.set_points_color(c));
    }
    if enabled {
        if color_edit(ui, "Point color", &mut color) {
            apply3d(node, recursive, |o| o.set_points_color(Some(color)));
        }
        let mut changed = slider(ui, "Point size", &mut size, 0.0..=20.0);
        changed |= ui.checkbox(&mut persp, "Point perspective").changed();
        if changed {
            apply3d(node, recursive, |o| o.set_points_size(size, persp));
        }
    }
}

/// Loads `p` into `target` for `node` (recursively for a group). Returns whether
/// the file existed and was applied.
fn apply_map(node: &SceneNode3d, recursive: bool, target: MapTarget, p: &str) -> bool {
    if p.is_empty() || !std::path::Path::new(p).is_file() {
        return false;
    }
    apply3d(node, recursive, |o| {
        let pth = std::path::Path::new(p);
        match target {
            MapTarget::BaseColor => o.set_texture_from_file(pth, p),
            MapTarget::Normal => o.set_normal_map_from_file(pth, p),
            MapTarget::MetallicRoughness => o.set_metallic_roughness_map_from_file(pth, p),
            MapTarget::Ao => o.set_ao_map_from_file(pth, p),
            MapTarget::Emissive => o.set_emissive_map_from_file(pth, p),
            MapTarget::Height => o.set_height_map_from_file(pth, p),
        }
    });
    true
}

fn load_map_status(ok: bool, p: &str, target: MapTarget) -> String {
    if ok {
        format!("loaded {} as {}", p, target.label())
    } else {
        format!("not found: {p}")
    }
}

/// Image-map loader for the selected 3D object/group: pick a target slot, open or
/// type an image path, and Load (or Clear) it. Applies recursively to the whole
/// subtree when a group is selected. For the height map it also exposes the
/// parallax (relief) settings.
fn maps_ui(
    ui: &mut egui::Ui,
    node: &SceneNode3d,
    recursive: bool,
    path: &mut String,
    target: &mut MapTarget,
    status: &mut String,
) {
    ui.separator();
    ui.label("Image maps");

    egui::ComboBox::from_id_salt("inspector_map_target")
        .selected_text(target.label())
        .show_ui(ui, |ui| {
            for t in MapTarget::ALL {
                ui.selectable_value(target, t, t.label());
            }
        });
    ui.horizontal(|ui| {
        ui.label("Image");
        ui.text_edit_singleline(path);
    });
    let tgt = *target;
    ui.horizontal(|ui| {
        // Native file-open dialog (wasm has no synchronous picker; type a path).
        #[cfg(not(target_arch = "wasm32"))]
        if ui.button("Open…").clicked() {
            if let Some(p) = pick_image_path() {
                *status = load_map_status(apply_map(node, recursive, tgt, &p), &p, tgt);
                *path = p;
            }
        }
        if ui.button("Load").clicked() {
            let p = path.trim().to_string();
            *status = load_map_status(apply_map(node, recursive, tgt, &p), &p, tgt);
        }
        // The base color texture has no clear (objects always carry one).
        if tgt != MapTarget::BaseColor && ui.button("Clear").clicked() {
            apply3d(node, recursive, |o| match tgt {
                MapTarget::Normal => o.clear_normal_map(),
                MapTarget::MetallicRoughness => o.clear_metallic_roughness_map(),
                MapTarget::Ao => o.clear_ao_map(),
                MapTarget::Emissive => o.clear_emissive_map(),
                MapTarget::Height => o.clear_height_map(),
                MapTarget::BaseColor => {}
            });
            *status = format!("cleared {}", tgt.label());
        }
    });
    if !status.is_empty() {
        ui.label(status.as_str());
    }

    // Relief / parallax settings, relevant to the height map.
    if tgt == MapTarget::Height {
        if let Some(mut scale) = mat_get(node, recursive, |o| o.data().parallax_scale()) {
            if slider(ui, "Parallax scale", &mut scale, 0.0..=0.2) {
                apply3d(node, recursive, |o| o.set_parallax_scale(scale));
            }
        }
        if let Some(mut layers) = mat_get(node, recursive, |o| o.data().parallax_layers()) {
            if slider(ui, "Parallax layers", &mut layers, 1.0..=64.0) {
                apply3d(node, recursive, |o| o.set_parallax_layers(layers));
            }
        }
        if let Some(cur) = mat_get(node, recursive, |o| o.data().parallax_method()) {
            parallax_method_ui(ui, node, recursive, cur);
        }
    }
}

/// Combo + (for relief) max-steps control for the parallax method.
fn parallax_method_ui(
    ui: &mut egui::Ui,
    node: &SceneNode3d,
    recursive: bool,
    current: ParallaxMethod,
) {
    let is_relief = matches!(current, ParallaxMethod::Relief { .. });
    let mut steps = match current {
        ParallaxMethod::Relief { max_steps } => max_steps,
        ParallaxMethod::Occlusion => 8,
    };

    let mut selected_relief = is_relief;
    egui::ComboBox::from_id_salt("inspector_parallax_method")
        .selected_text(if is_relief { "Relief" } else { "Occlusion" })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut selected_relief, false, "Occlusion");
            ui.selectable_value(&mut selected_relief, true, "Relief");
        });
    if selected_relief != is_relief {
        let m = if selected_relief {
            ParallaxMethod::Relief { max_steps: steps }
        } else {
            ParallaxMethod::Occlusion
        };
        apply3d(node, recursive, |o| o.set_parallax_method(m));
    } else if selected_relief
        && ui
            .add(egui::Slider::new(&mut steps, 1..=16).text("Relief steps"))
            .changed()
    {
        apply3d(node, recursive, |o| {
            o.set_parallax_method(ParallaxMethod::Relief { max_steps: steps })
        });
    }
}
