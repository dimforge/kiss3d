//! Loads a glTF / GLB model and plays its animations.
//!
//! Pass a path to load your own model, otherwise the bundled animated, skinned
//! Fox is shown:
//!
//! ```bash
//! cargo run --release --example gltf -- path/to/model.glb
//! ```
//!
//! Build with `--features egui` for an in-window panel to switch the active
//! animation clip and tweak playback speed:
//!
//! ```bash
//! cargo run --release --example gltf --features egui
//! ```

use kiss3d::prelude::*;
use std::path::Path;
use kiss3d::window::Inspector;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: glTF").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(2.6, 1.4, 2.6), Vec3::new(0.0, 0.5, 0.0));

    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(200.0))
        .set_position(Vec3::new(3.0, 6.0, -6.0));
    scene
        .add_light(Light::point(120.0))
        .set_position(Vec3::new(-4.0, 3.0, 4.0));
    scene.add_cube(4.0, 0.1, 4.0)
        .set_position(Vec3::new(0.0, -0.07, 0.0));

    // Default to the bundled Fox (animated + skinned); a CLI arg overrides it.
    let arg = std::env::args().nth(1);
    let path = arg.as_deref().unwrap_or("examples/media/gltf/Fox.glb");

    // The Fox is authored in centimeters; scale it down to a comfortable size.
    let scale = if arg.is_some() {
        Vec3::ONE
    } else {
        Vec3::splat(0.012)
    };
    let mut fox = scene.add_gltf(Path::new(path), scale);

    let names: Vec<String> = fox.player.clip_names().map(|s| s.to_string()).collect();
    println!("Loaded `{path}` with {} animation(s): {names:?}", names.len());
    if let Some(first) = names.first() {
        fox.player.play(first);
        fox.player.set_looping(true);
    }

    // A model with morph atergts
    let mut morph_cube = scene.add_gltf(Path::new("examples/media/gltf/AnimatedMorphCube.glb"), Vec3::splat(0.3));

    morph_cube.root.set_position(Vec3::new(0.8, 0.35, 0.0));
    morph_cube.player.play_index(0);
    morph_cube.player.set_looping(true);

    // egui panel state: the index of the selected clip plus playback controls.
    #[cfg(feature = "egui")]
    let mut selected = 0usize;
    #[cfg(feature = "egui")]
    let mut speed = 1.0f32;
    #[cfg(feature = "egui")]
    let mut looping = true;
    #[cfg(feature = "egui")]
    let mut opacity = 1.0f32;

    // No per-frame delta is exposed by the window, so advance at a fixed timestep.
    let dt = 1.0 / 60.0;

    let mut raytracer = RayTracer::with_enabled(false);
    let mut inspector = Inspector::default();

    while window.raytrace_3d(&mut scene, &mut camera, &mut raytracer).await {
        fox.player.update(dt);
        morph_cube.player.update(dt);

        #[cfg(feature = "egui")]
        {
            let mut pick = selected;
            window.draw_ui(|ctx| {
                egui::Window::new("Animation")
                    .default_width(260.0)
                    .show(ctx, |ui| {
                        if names.is_empty() {
                            ui.label("This model has no animations.");
                            return;
                        }
                        egui::ComboBox::from_label("Clip")
                            .selected_text(names.get(selected).map(String::as_str).unwrap_or("—"))
                            .show_ui(ui, |ui| {
                                for (i, name) in names.iter().enumerate() {
                                    let label = if name.is_empty() {
                                        format!("clip {i}")
                                    } else {
                                        name.clone()
                                    };
                                    ui.selectable_value(&mut pick, i, label);
                                }
                            });
                        ui.add(egui::Slider::new(&mut speed, -3.0..=3.0).text("Speed"));
                        ui.checkbox(&mut looping, "Loop");
                        ui.add(egui::Slider::new(&mut opacity, 0.0..=1.0).text("Opacity"));
                        if ui.button("Restart").clicked() {
                            fox.player.play_index(pick);
                        }
                    });
            });

            if pick != selected {
                selected = pick;
                if let Some(name) = names.get(selected) {
                    fox.player.play(name);
                }
            }
            fox.player.set_speed(speed);
            fox.player.set_looping(looping);

            // Drive the model's opacity from the slider (keep RGB, set alpha). An
            // alpha < 1 routes the mesh through the transparent / transmittance-shadow
            // path; the texture still tints both the surface and its shadow.
            fox.root.apply_to_scene_nodes_mut_recursive(&mut |n| {
                if n.data().has_object() {
                    let c = n.data().get_object().data().color();
                    n.set_color(Color::new(c.r, c.g, c.b, opacity));
                }
            });

            window.draw_inspector(&mut inspector, &mut scene, Some(&mut raytracer));
        }
    }
}
