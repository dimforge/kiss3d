//! A "robot" assembled from primitives drives a figure-eight through a small
//! arena. A picture-in-picture panel in the top-right corner shows the scene
//! through the robot's onboard camera, with an egui combo box to switch the
//! sensor between regular color, path-traced, depth, normals (world or camera
//! space) and segmentation rendering.
//!
//! The obstacles use a variety of materials — chrome and copper mirrors, clear
//! and frosted glass, gold, emissive panels, subsurface scattering, and a
//! mirror wall panel — that show their full effect (reflections, refraction,
//! soft shadows, bounce light) in the raytraced mode. An equirectangular
//! skybox provides the environment: it enables image-based lighting in the
//! rasterized view (so metals reflect the sky there too) and serves as the
//! path tracer's environment light.
//!
//! The robot's eye is an [`OffscreenSurface`] sharing the window's GPU context
//! and scene graph: the color mode re-renders the scene from the robot camera,
//! the raytraced mode runs the path tracer (accumulation keeps refining while
//! the robot is paused), and the other modes use the GPU AOV visualization
//! (`render_aov_3d`). The surface's output texture is registered directly with
//! the window's egui renderer (`register_egui_texture`), so the feed never
//! leaves the GPU — which is also what makes this demo web-compatible.

#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!("The 'egui' feature must be enabled for this example to work.")
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    use kiss3d::prelude::*;
    use kiss3d::renderer::RayTracer;
    use std::f32::consts::{FRAC_PI_2, PI, TAU};
    #[cfg(not(target_arch = "wasm32"))]
    use std::path::Path;

    /// What the robot's onboard camera displays.
    #[derive(PartialEq, Clone, Copy)]
    enum ViewMode {
        Color,
        Raytraced,
        Depth,
        Normals,
        CameraNormals,
        Segmentation,
    }

    impl ViewMode {
        const ALL: [ViewMode; 6] = [
            ViewMode::Color,
            ViewMode::Raytraced,
            ViewMode::Depth,
            ViewMode::Normals,
            ViewMode::CameraNormals,
            ViewMode::Segmentation,
        ];

        fn label(self) -> &'static str {
            match self {
                ViewMode::Color => "Color",
                ViewMode::Raytraced => "Raytraced",
                ViewMode::Depth => "Depth",
                ViewMode::Normals => "Normals (world)",
                ViewMode::CameraNormals => "Normals (camera)",
                ViewMode::Segmentation => "Segmentation",
            }
        }
    }

    const PIP_W: u32 = 320;
    const PIP_H: u32 = 240;
    // Depth is mapped to grayscale over a fixed range (in world units) so the
    // visualization doesn't flicker as the per-frame min/max changes.
    const DEPTH_RANGE: f32 = 16.0;

    let mut window = Window::new_with_size("Kiss3d: robot view", 1280, 800).await;
    window.set_background_color(Color::new(0.05, 0.06, 0.09, 1.0));
    window.set_ambient(0.2);

    // The robot's eye: a small headless surface. It reuses the window's wgpu
    // context, so both can render the same scene graph.
    let mut pip = OffscreenSurface::new(PIP_W, PIP_H).await;
    pip.set_background_color(Color::new(0.05, 0.06, 0.09, 1.0));
    pip.set_ambient(0.2);

    // The skybox enables image-based lighting in the rasterized views and is
    // the path tracer's environment. It is per-surface, so set it on both the
    // main window and the robot's eye.
    #[cfg(not(target_arch = "wasm32"))]
    {
        window.set_skybox_from_file(Path::new("./examples/media/skybox.png"));
        pip.window_mut()
            .set_skybox_from_file(Path::new("./examples/media/skybox.png"));
    }
    #[cfg(target_arch = "wasm32")]
    {
        window.set_skybox_from_memory(include_bytes!("media/skybox.png"));
        pip.window_mut()
            .set_skybox_from_memory(include_bytes!("media/skybox.png"));
    }

    let mut scene = SceneNode3d::empty();

    scene
        .add_light(
            Light::directional(Vec3::new(-0.5, -1.0, -0.4))
                .with_color(Color::new(1.0, 0.96, 0.85, 1.0))
                .with_intensity(2.5),
        )
        .set_position(Vec3::new(0.0, 8.0, 0.0));
    scene
        .add_light(
            Light::point(40.0)
                .with_color(Color::new(0.4, 0.45, 0.6, 1.0))
                .with_intensity(1.5)
                .with_casts_shadows(false),
        )
        .set_position(Vec3::new(6.0, 5.0, -6.0));

    // === Arena: ground, walls and a few obstacles ===
    // Slightly glossy ground so the raytraced view picks up reflections of the
    // emissive and metallic obstacles.
    let mut ground = scene
        .add_cube(20.0, 0.2, 20.0)
        .set_position(Vec3::new(0.0, -0.1, 0.0))
        .set_color(Color::new(0.45, 0.47, 0.5, 1.0))
        .set_roughness(0.35);
    ground.apply_to_object_mut(&mut |o| o.set_segmentation_id(1));

    let walls = vec![
        (Vec3::new(20.0, 1.6, 0.4), Vec3::new(0.0, 0.8, 9.8)),
        (Vec3::new(20.0, 1.6, 0.4), Vec3::new(0.0, 0.8, -9.8)),
        (Vec3::new(0.4, 1.6, 20.0), Vec3::new(9.8, 0.8, 0.0)),
        (Vec3::new(0.4, 1.6, 20.0), Vec3::new(-9.8, 0.8, 0.0)),
    ];
    for (i, (size, pos)) in walls.into_iter().enumerate() {
        let mut wall = scene
            .add_cube(size.x, size.y, size.z)
            .set_position(pos)
            .set_color(Color::new(0.3, 0.32, 0.38, 1.0));
        wall.apply_to_object_mut(&mut |o| o.set_segmentation_id(2 + i as u32));
    }

    // Obstacles with a variety of materials. Their full effect (mirror
    // reflections, refraction, emission, subsurface) shows in the raytraced
    // view mode. Positions are chosen to stay clear of the robot's
    // figure-eight path (the two pillars sit at the centers of its lobes).

    // Transparent colored pillar.
    scene
        .add_cylinder(0.6, 2.8)
        .set_position(Vec3::new(4.0, 1.4, 0.0))
        .set_color(Color::new(0.1, 0.92, 0.7, 0.75))
        .set_metallic(1.0)
        .set_roughness(0.03)
        .apply_to_object_mut(&mut |o| o.set_segmentation_id(10));

    // Emissive pillar: an area light in the raytraced view.
    scene
        .add_cylinder(0.6, 2.8)
        .set_position(Vec3::new(-4.0, 1.4, 0.0))
        .set_color(ORANGE)
        .set_emissive(Color::new(3.0, 1.4, 0.4, 1.0))
        .apply_to_object_mut(&mut |o| o.set_segmentation_id(11));

    // Clear glass sphere: refraction.
    scene
        .add_sphere(0.8)
        .set_position(Vec3::new(8.0, 0.8, 2.5))
        .set_color(WHITE)
        .set_bsdf(Bsdf::Glass)
        .set_ior(1.5)
        .set_transmission(1.0)
        .set_roughness(0.0)
        .apply_to_object_mut(&mut |o| o.set_segmentation_id(12));

    // Brushed gold cone.
    scene
        .add_cone(0.8, 1.8)
        .set_position(Vec3::new(-8.0, 0.9, -2.0))
        .set_color(Color::new(1.0, 0.85, 0.4, 1.0))
        .set_bsdf(Bsdf::Metal)
        .set_metallic(1.0)
        .set_specular_tint(Color::new(1.0, 0.82, 0.45, 1.0))
        .set_roughness(0.15)
        .apply_to_object_mut(&mut |o| o.set_segmentation_id(13));

    // Frosted glass cube, slightly aqua-tinted: rough refraction. Sits near the
    // middle of the arena so the pillars and the passing robot are seen blurred
    // through it.
    scene
        .add_cube(1.5, 1.5, 1.5)
        .set_position(Vec3::new(0.0, 0.75, -3.0))
        .set_color(Color::new(0.75, 0.2, 0.92, 1.0))
        .set_bsdf(Bsdf::Glass)
        .set_ior(2.45)
        .set_thickness(1.0)
        .set_transmission(1.0)
        .set_roughness(0.3)
        .apply_to_object_mut(&mut |o| o.set_segmentation_id(14));

    // Waxy capsule: subsurface scattering.
    scene
        .add_capsule(0.45, 1.2)
        .set_position(Vec3::new(-7.0, 1.05, 4.5))
        .set_color(Color::new(0.9, 0.4, 0.4, 1.0))
        .set_subsurface(0.8, 0.5)
        .set_roughness(0.5)
        .apply_to_object_mut(&mut |o| o.set_segmentation_id(15));

    // Matte plastic cube.
    scene
        .add_cube(2.5, 1.5, 1.0)
        .set_position(Vec3::new(0.0, 0.75, 5.8))
        .set_color(TEAL)
        .set_roughness(0.9)
        .apply_to_object_mut(&mut |o| o.set_segmentation_id(16));

    // Copper mirror sphere.
    scene
        .add_sphere(1.0)
        .set_position(Vec3::new(0.0, 1.0, -6.0))
        .set_color(Color::new(0.95, 0.55, 0.35, 1.0))
        .set_metallic(1.0)
        .set_specular_tint(Color::new(0.95, 0.6, 0.4, 1.0))
        .set_roughness(0.08)
        .apply_to_object_mut(&mut |o| o.set_segmentation_id(17));

    // Glossy plastic cone: sharp specular highlights.
    scene
        .add_cone(0.7, 1.2)
        .set_position(Vec3::new(3.2, 0.6, -5.5))
        .set_color(DODGER_BLUE)
        .set_roughness(0.15)
        .apply_to_object_mut(&mut |o| o.set_segmentation_id(18));

    // Small emissive "bulb".
    scene
        .add_sphere(0.5)
        .set_position(Vec3::new(-3.0, 0.5, 5.5))
        .set_color(YELLOW_GREEN)
        .set_emissive(Color::new(1.2, 2.5, 0.6, 1.0))
        .apply_to_object_mut(&mut |o| o.set_segmentation_id(19));

    // Mirror panel on the north wall: a planar reflector quad, rotated so its
    // normal faces the arena (-Z). The rasterized views render a true mirror
    // image of the scene — robot included — into it every frame, and the path
    // tracer reflects it like polished metal. The robot slows down and stares
    // at it for a few seconds when passing nearby.
    let mirror_pos = Vec3::new(4.0, 1.2, 9.55);
    scene
        .add_reflector(4.5, 2.4)
        .set_position(mirror_pos)
        .set_rotation(Quat::from_rotation_y(PI))
        .set_color(Color::new(0.85, 0.88, 0.95, 1.0))
        .set_metallic(1.0)
        .set_roughness(0.05)
        .apply_to_object_mut(&mut |o| o.set_segmentation_id(20));

    // === The robot, built from primitives (local forward is +Z) ===
    const WHEEL_RADIUS: f32 = 0.25;

    let mut robot = scene.add_group();
    robot
        .add_cube(0.9, 0.4, 1.3)
        .set_position(Vec3::new(0.0, 0.45, 0.0))
        .set_color(Color::new(0.75, 0.78, 0.82, 1.0));
    robot
        .add_cylinder(0.08, 0.2)
        .set_position(Vec3::new(0.0, 0.75, 0.1))
        .set_color(Color::new(0.2, 0.2, 0.22, 1.0));

    // The head swivels left/right to "scan" the environment; the onboard
    // camera follows it.
    let mut head = robot.add_group();
    head.set_position(Vec3::new(0.0, 0.95, 0.1));
    head.add_cube(0.5, 0.4, 0.5)
        .set_color(Color::new(0.85, 0.87, 0.9, 1.0));
    head.add_cylinder(0.09, 0.1)
        .set_rotation(Quat::from_rotation_x(FRAC_PI_2))
        .set_position(Vec3::new(0.0, 0.05, 0.3))
        .set_color(Color::new(0.1, 0.6, 0.8, 1.0));

    robot
        .add_cylinder(0.02, 0.5)
        .set_position(Vec3::new(-0.25, 0.9, -0.5))
        .set_color(Color::new(0.2, 0.2, 0.22, 1.0));
    robot
        .add_sphere(0.05)
        .set_position(Vec3::new(-0.25, 1.18, -0.5))
        .set_color(RED);

    let mut wheels = Vec::new();
    for (x, z) in [(-0.55, 0.45), (0.55, 0.45), (-0.55, -0.45), (0.55, -0.45)] {
        wheels.push(
            robot
                .add_cylinder(WHEEL_RADIUS, 0.14)
                .set_position(Vec3::new(x, WHEEL_RADIUS, z))
                .set_color(Color::new(0.12, 0.12, 0.14, 1.0)),
        );
    }
    robot.apply_to_objects_mut_recursive(&mut |o| o.set_segmentation_id(99));

    // === Cameras ===
    let mut main_camera = OrbitCamera3d::new(Vec3::new(10.0, 8.0, 10.0), Vec3::ZERO);
    // The robot's eye; never receives input events, it is repositioned
    // programmatically every frame.
    let mut robot_camera = FirstPersonCamera3d::new_with_frustum(
        60.0f32.to_radians(),
        0.05,
        100.0,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::Z,
    );

    // Path tracer for the raytraced view mode. Accumulation resets
    // automatically whenever the robot camera moves, so the feed stays live;
    // pausing the robot lets it converge.
    let mut raytracer = RayTracer::new();
    raytracer.set_max_bounces(6);

    // The PiP display: register the offscreen surface's output texture
    // directly with the window's egui renderer. The robot's view never leaves
    // the GPU — no per-frame read-back or re-upload.
    let mut pip_tex = window.register_egui_texture(&pip.output_view(), wgpu::FilterMode::Linear);
    // Allocated PiP resolution vs the size wanted by the (resizable) panel.
    let mut pip_size = (PIP_W, PIP_H);
    let mut pip_desired = pip_size;

    // === UI state ===
    let mut mode = ViewMode::Color;
    let mut paused = false;
    let mut show_robot_in_pip = true;
    let mut rt_samples = 8u32;
    // Set when the trajectory slider scrubbed `u`, so the wheels don't roll
    // across the jump.
    let mut scrubbed = false;
    // User-controlled angular offset added to the head's scripted orientation.
    let mut look_offset = 0.0f32;

    // The figure-eight the robot patrols.
    let path = |u: f32| Vec3::new(6.0 * u.sin(), 0.0, 3.5 * (2.0 * u).sin());

    let mut t = 0.0f32;
    let mut u = 0.0f32; // path parameter (integrated, so the speed can vary)
    let mut wheel_spin = 0.0f32;
    let mut last_pos = path(0.0);

    // Head-scan state: the head normally sweeps left/right, but when the wall
    // mirror comes roughly ahead the robot slows down and stares at it for a
    // few seconds, the head tracking it well past the heading as it drives by.
    const GAZE_ACQUIRE: f32 = 1.0; // bearing under which the robot notices the mirror
    const GAZE_TRACK_LIMIT: f32 = 2.0; // max head swivel while staring, radians
    const GAZE_TIME: f32 = 7.0; // how long one stare lasts, seconds
    const GAZE_SLOWDOWN: f32 = 0.70; // path-speed factor while staring
    let mut head_yaw = 0.0f32;
    let mut gaze_timer = 0.0f32; // > 0 while staring at the mirror
    let mut gaze_cooldown = 0.0f32; // delay before the next lock-on

    while window.render_3d(&mut scene, &mut main_camera).await {
        if !paused {
            t += 0.016;
            // Drive along the path; the robot slows down while staring at the
            // mirror, which also stretches the stare's geometric window.
            let speed = if gaze_timer > 0.0 { GAZE_SLOWDOWN } else { 1.0 };
            u += 0.016 * 0.45 * speed;
        }

        // Heading along the path tangent.
        let pos = path(u);
        let dir = (path(u + 0.01) - pos).normalize();
        let yaw = dir.x.atan2(dir.z);
        let rot = Quat::from_rotation_y(yaw);
        robot.set_position(pos).set_rotation(rot);

        // Don't roll the wheels across a trajectory-slider jump.
        if scrubbed {
            last_pos = pos;
            scrubbed = false;
        }

        // Roll the wheels by the distance traveled.
        wheel_spin += (pos - last_pos).length() / WHEEL_RADIUS;
        last_pos = pos;
        for wheel in &mut wheels {
            wheel
                .set_rotation(Quat::from_rotation_x(wheel_spin) * Quat::from_rotation_z(FRAC_PI_2));
        }

        // Swivel the head, and place the onboard camera just in front of its lens.
        // Bearing of the mirror relative to the robot's heading, in [-pi, pi].
        let to_mirror = mirror_pos - (pos + Vec3::Y);
        let rel_mirror_yaw = (to_mirror.x.atan2(to_mirror.z) - yaw + PI).rem_euclid(TAU) - PI;
        let mirror_in_range = rel_mirror_yaw.abs() < GAZE_ACQUIRE && to_mirror.length() < 12.0;
        if !paused {
            let dt = 0.016;
            if gaze_timer > 0.0 {
                gaze_timer -= dt;
                // Done staring (or the robot turned so far away that even the
                // fully swiveled head lost the mirror): resume scanning, and
                // don't re-lock immediately.
                if gaze_timer <= 0.0 || rel_mirror_yaw.abs() > GAZE_TRACK_LIMIT + 0.3 {
                    gaze_timer = 0.0;
                    gaze_cooldown = 8.0;
                }
            } else if gaze_cooldown > 0.0 {
                gaze_cooldown -= dt;
            } else if mirror_in_range {
                gaze_timer = GAZE_TIME;
            }

            let target = if gaze_timer > 0.0 {
                rel_mirror_yaw.clamp(-GAZE_TRACK_LIMIT, GAZE_TRACK_LIMIT)
            } else {
                (t * 1.2).sin() * 0.6
            };
            // Ease toward the target so lock-on/release looks like a head turn.
            head_yaw += (target - head_yaw) * 0.08;
        }
        // The head's final orientation: the scripted scan/gaze angle plus the
        // user-controlled offset from the "Look offset" slider.
        let scan = head_yaw + look_offset;
        head.set_rotation(Quat::from_rotation_y(scan));
        let scan_rot = rot * Quat::from_rotation_y(scan);
        let eye = pos + rot * Vec3::new(0.0, 1.0, 0.1) + scan_rot * Vec3::new(0.0, 0.0, 0.42);
        let look = scan_rot * Vec3::new(0.0, -0.12, 1.0);
        robot_camera.look_at(eye, eye + look * 5.0);

        // Apply a PiP panel resize: re-render at the displayed resolution so an
        // enlarged panel stays sharp. The resize reallocates the surface's
        // output texture, so the egui registration must be refreshed.
        if pip_desired != pip_size {
            pip.resize(pip_desired.0, pip_desired.1);
            window.unregister_egui_texture(pip_tex);
            pip_tex = window.register_egui_texture(&pip.output_view(), wgpu::FilterMode::Linear);
            pip_size = pip_desired;
        }

        // Render what the robot sees into the PiP texture. Every mode stays on
        // the GPU: the beauty/raytraced passes render into the surface's output
        // target, and the AOV modes use the GPU visualization.
        if !show_robot_in_pip {
            robot.set_visible(false);
        }
        match mode {
            ViewMode::Color => {
                pip.render_3d(&mut scene, &mut robot_camera).await;
            }
            ViewMode::Raytraced => {
                raytracer.set_samples_per_frame(rt_samples);
                pip.raytrace_3d(&mut scene, &mut robot_camera, &mut raytracer)
                    .await;
            }
            ViewMode::Depth => {
                pip.render_aov_3d(AovKind::Depth, &mut scene, &mut robot_camera, DEPTH_RANGE);
            }
            ViewMode::Normals => {
                pip.render_aov_3d(AovKind::Normals, &mut scene, &mut robot_camera, 0.0);
            }
            ViewMode::CameraNormals => {
                pip.render_aov_3d(AovKind::CameraNormals, &mut scene, &mut robot_camera, 0.0);
            }
            ViewMode::Segmentation => {
                pip.render_aov_3d(AovKind::Segmentation, &mut scene, &mut robot_camera, 0.0);
            }
        }
        robot.set_visible(true);

        window.draw_ui(|ctx| {
            egui::Window::new("Robot eye")
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-10.0, 10.0))
                .resizable(true)
                .default_width(PIP_W as f32)
                .collapsible(false)
                .show(ctx, |ui| {
                    // The image fills the panel width at a 4:3 aspect; the
                    // surface is re-rendered at this size (in physical pixels)
                    // so enlarging the panel keeps the feed sharp.
                    let img_w = ui.available_width().max(120.0);
                    let img_h = img_w * (PIP_H as f32 / PIP_W as f32);
                    ui.image((pip_tex, egui::vec2(img_w, img_h)));
                    let ppp = ui.ctx().pixels_per_point();
                    pip_desired = (
                        ((img_w * ppp).round() as u32).max(64),
                        ((img_h * ppp).round() as u32).max(48),
                    );
                    egui::ComboBox::from_label("Mode")
                        .selected_text(mode.label())
                        .show_ui(ui, |ui| {
                            for m in ViewMode::ALL {
                                ui.selectable_value(&mut mode, m, m.label());
                            }
                        });
                    // Scrub the robot along its figure-eight (one lap = 1.0).
                    // Auto-play keeps advancing from wherever it is dropped.
                    let mut lap = u.rem_euclid(TAU) / TAU;
                    if ui
                        .add(egui::Slider::new(&mut lap, 0.0..=1.0).text("Position"))
                        .changed()
                    {
                        u = lap * TAU;
                        scrubbed = true;
                    }
                    // Turn the head relative to its scripted orientation.
                    let mut look_deg = look_offset.to_degrees();
                    if ui
                        .add(
                            egui::Slider::new(&mut look_deg, -180.0..=180.0)
                                .suffix("°")
                                .text("Look offset"),
                        )
                        .changed()
                    {
                        look_offset = look_deg.to_radians();
                    }
                    if mode == ViewMode::Raytraced {
                        ui.add(egui::Slider::new(&mut rt_samples, 1..=32).text("Samples/frame"));
                    }
                    ui.checkbox(&mut show_robot_in_pip, "Show robot body");
                    ui.checkbox(&mut paused, "Pause robot");
                });
        });
    }
}
