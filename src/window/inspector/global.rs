//! The "Global" inspector tab: the renderer toggle, settings shared by both
//! renderers (exposure/tonemap), the rasterizer settings (background, MSAA,
//! shadows, bloom, SSAO/SSR/DoF and the skybox loader), the path-tracer settings,
//! and the per-frame timings.

use crate::post_processing::HdrSettings;
use crate::renderer::{RayTracer, RenderTimings};
use crate::window::NumSamples;

use super::widgets::*;
use super::{Inspector, WinSettings};

impl Inspector {
    /// Per-step wall-clock timings of the last rendered frame.
    pub(super) fn timings_section(ui: &mut egui::Ui, timings: Option<&RenderTimings>) {
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

    pub(super) fn renderer_section(
        &mut self,
        ui: &mut egui::Ui,
        raytracer: Option<&mut RayTracer>,
    ) {
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

    /// Settings that affect both renderers (exposure, tonemap operator, vsync).
    pub(super) fn common_section(
        &mut self,
        ui: &mut egui::Ui,
        hdr: &mut HdrSettings,
        vsync: &mut bool,
    ) {
        egui::CollapsingHeader::new("Common")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(
                    egui::Slider::new(&mut hdr.exposure, 0.0..=8.0)
                        .text("Exposure")
                        .logarithmic(true),
                );
                tonemap_combo(ui, "inspector_tonemap", &mut hdr.tonemap);
                ui.checkbox(vsync, "VSync").on_hover_text(
                    "Off presents frames uncapped (not paced to the display refresh) \
                     — compare with the wall-clock frame time in Timings to see the \
                     real GPU-bound throughput.",
                );
            });
    }

    pub(super) fn rasterizer_section(&mut self, ui: &mut egui::Ui, win: &mut WinSettings) {
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
                            ui.selectable_value(&mut win.samples, NumSamples::One, "Off");
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
                    ui.add(
                        egui::Slider::new(&mut hdr.bloom_threshold, 0.0..=4.0).text("Threshold"),
                    );
                    ui.add(egui::Slider::new(&mut hdr.bloom_knee, 0.0..=1.0).text("Knee"));
                    ui.add(
                        egui::Slider::new(&mut hdr.bloom_intensity, 0.0..=1.0).text("Intensity"),
                    );
                });

                self.effects_ui(ui);
                self.skybox_ui(ui);
            });
    }

    /// Equirectangular skybox loader (the actual load/clear is applied to the
    /// window in `draw_inspector`; the request flags are set here).
    fn skybox_ui(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Skybox", |ui| {
            ui.horizontal(|ui| {
                ui.label("Image");
                ui.text_edit_singleline(&mut self.skybox_path);
            });
            ui.horizontal(|ui| {
                // Native file-open dialog (wasm: type a path instead).
                #[cfg(not(target_arch = "wasm32"))]
                if ui.button("Open…").clicked() {
                    if let Some(p) = pick_image_path() {
                        self.skybox_path = p;
                        self.skybox_load_requested = true;
                    }
                }
                if ui.button("Load").clicked() {
                    self.skybox_load_requested = true;
                }
                if ui.button("Clear").clicked() {
                    self.skybox_clear_requested = true;
                }
            });
            let mut changed = ui
                .add(
                    egui::Slider::new(&mut self.skybox_rotation_deg, -180.0..=180.0)
                        .text("Rotation°"),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.skybox_intensity, 0.0..=8.0).text("Intensity"))
                .changed();
            if changed {
                self.skybox_orient_dirty = true;
            }
            if !self.skybox_status.is_empty() {
                ui.label(&self.skybox_status);
            }
        });
    }

    /// Screen-space effects sharing the geometry prepass (SSAO / SSR / DoF). All
    /// are rasterizer-only post-processes; the path tracer ignores them.
    fn effects_ui(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Ambient occlusion (SSAO)", |ui| {
            ui.checkbox(&mut self.ssao_enabled, "Enabled");
            ui.add_enabled_ui(self.ssao_enabled, |ui| {
                let s = &mut self.ssao;
                ui.add(egui::Slider::new(&mut s.radius, 0.0..=4.0).text("Radius"));
                ui.add(egui::Slider::new(&mut s.bias, 0.0..=0.2).text("Bias"));
                ui.add(egui::Slider::new(&mut s.intensity, 0.0..=4.0).text("Intensity"));
                ui.add(egui::Slider::new(&mut s.power, 0.1..=8.0).text("Power"));
            });
        });

        ui.collapsing("Screen-space reflections (SSR)", |ui| {
            ui.checkbox(&mut self.ssr_enabled, "Enabled")
                .on_hover_text("Native / WebGPU only (needs compute + storage buffers)");
            ui.add_enabled_ui(self.ssr_enabled, |ui| {
                let s = &mut self.ssr;
                ui.add(egui::Slider::new(&mut s.intensity, 0.0..=2.0).text("Intensity"));
                ui.add(egui::Slider::new(&mut s.max_steps, 8..=128).text("Max steps"));
                ui.add(egui::Slider::new(&mut s.thickness, 0.01..=3.0).text("Thickness"));
                ui.add(egui::Slider::new(&mut s.max_distance, 1.0..=200.0).text("Max distance"));
                ui.add(
                    egui::Slider::new(&mut s.roughness_cutoff, 0.0..=1.0).text("Roughness cutoff"),
                );
                ui.add(egui::Slider::new(&mut s.edge_fade, 0.0..=0.5).text("Edge fade"));
            });
        });

        ui.collapsing("Depth of field", |ui| {
            ui.checkbox(&mut self.dof_enabled, "Enabled");
            ui.add_enabled_ui(self.dof_enabled, |ui| {
                let s = &mut self.dof;
                dof_mode_combo(ui, &mut s.mode);
                ui.add(
                    egui::Slider::new(&mut s.focal_distance, 0.1..=100.0)
                        .text("Focal distance")
                        .logarithmic(true),
                );
                ui.add(
                    egui::Slider::new(&mut s.aperture_f_stops, 0.05..=16.0)
                        .text("Aperture (f-stops)")
                        .logarithmic(true),
                );
                ui.add(
                    egui::Slider::new(&mut s.max_coc_diameter, 4.0..=96.0).text("Max blur (px)"),
                );
                ui.add(egui::Slider::new(&mut s.num_taps, 8..=96).text("Gather taps"));
            });
        });
    }

    pub(super) fn path_tracer_section(&mut self, ui: &mut egui::Ui) {
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

                // Depth of field is shared with the rasterizer — edit it in the
                // global "Depth of field" section above; the path tracer mirrors it.

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
}
