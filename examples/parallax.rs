//! Parallax-occlusion mapping.
//!
//! Uses three textures (a warm plaid base color, and
//! a normal + depth map describing a beveled frame around a 2x2 grid of carved
//! features). A cube rotates at the center, lit by a point light (shown as a
//! small sphere), on a ground plane and surrounded by four large cubes. Parallax
//! gives the flat faces real depth that shifts as the cube turns.
//!
//! The egui panel exposes controls for depth scale, max
//! layer count, the occlusion-vs-relief mapping method, and the relief search
//! step count).
//!
//! Textures are from Bevy (MIT/Apache-2.0) — see
//! `examples/media/parallax/CREDITS.md`.
//!
//! Run with the `egui` feature: `cargo run --features egui --example parallax`.

#[cfg(not(feature = "egui"))]
#[kiss3d::main]
async fn main() {
    panic!("The 'egui' feature must be enabled: cargo run --features egui --example parallax");
}

#[cfg(feature = "egui")]
use kiss3d::prelude::*;
#[cfg(feature = "egui")]
use kiss3d::resource::Texture;
#[cfg(feature = "egui")]
use kiss3d::scene::ParallaxMethod;
#[cfg(feature = "egui")]
use std::path::Path;
#[cfg(feature = "egui")]
use std::sync::Arc;

/// Loads a texture from a file. `srgb` selects sRGB color vs. linear data
/// (normal/height maps must be linear and are loaded without sRGB decoding).
///
/// `invert` flips the grayscale value — used for the depth map, since the source
/// treats it as depth (white = deepest) whereas kiss3d's height map uses white =
/// at the surface. `flip_green` flips the normal map's Y, since the source's normal
/// maps are OpenGL (+Y) while kiss3d's tangent frame expects the opposite.
///
/// Textures use `ClampToEdge`: parallax can push the displaced UV past the [0,1]
/// tile at grazing angles, and clamping extends the edge color
/// instead of wrapping the whole pattern back in, which would look like an
/// infinitely repeating "portal" on the relief walls.
#[cfg(feature = "egui")]
fn load(path: &str, srgb: bool, invert: bool, flip_green: bool) -> Arc<Texture> {
    let mut img = image::open(Path::new(path))
        .unwrap_or_else(|e| panic!("failed to load {}: {}", path, e))
        .to_rgba8();
    if invert || flip_green {
        for p in img.pixels_mut() {
            if invert {
                p[0] = 255 - p[0];
                p[1] = 255 - p[1];
                p[2] = 255 - p[2];
            }
            if flip_green {
                p[1] = 255 - p[1];
            }
        }
    }
    let format = if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    Texture::new(
        img.width(),
        img.height(),
        img.as_raw(),
        format,
        wgpu::AddressMode::ClampToEdge,
        true,
    )
}

#[cfg(feature = "egui")]
fn apply(node: &mut SceneNode3d, m: &(Arc<Texture>, Arc<Texture>, Arc<Texture>), rough: f32) {
    node.set_texture(m.0.clone());
    node.set_normal_map(m.1.clone());
    node.set_height_map(m.2.clone());
    node.set_roughness(rough);
}

#[cfg(feature = "egui")]
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: parallax").await;
    window.set_background_color(Color::new(0.13, 0.13, 0.17, 1.0));
    window.set_ambient(0.3);

    let mut camera = OrbitCamera3d::new(Vec3::new(1.5, 1.5, 1.5), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();

    let maps = (
        load("examples/media/parallax/cube_color.png", true, false, false),
        load(
            "examples/media/parallax/cube_normal.png",
            false,
            false,
            true,
        ),
        load("examples/media/parallax/cube_depth.png", false, true, false),
    );

    let light_pos = Vec3::new(2.0, 1.0, -1.1);
    scene
        .add_light(Light::point(60.0).with_intensity(5.0))
        .set_position(light_pos);
    scene
        .add_sphere(0.05)
        .translate(light_pos)
        .set_emissive(Color::new(6.0, 6.0, 6.0, 1.0));

    // Every brick node, so the UI can update the parallax settings on all of them.
    let mut bricks = Vec::new();

    let mut cube = scene.add_cube(1.0, 1.0, 1.0);
    apply(&mut cube, &maps, 0.4);
    bricks.push(cube.clone());

    let mut ground = scene.add_cube(10.0, 0.1, 10.0);
    ground.translate(Vec3::new(0.0, -1.0, 0.0));
    apply(&mut ground, &maps, 0.45);
    bricks.push(ground);

    let mut background = Vec::new();
    for (dx, dz) in [(45.0, 0.0), (-45.0, 0.0), (0.0, 45.0), (0.0, -45.0)] {
        let mut c = scene.add_cube(40.0, 40.0, 40.0);
        c.translate(Vec3::new(dx, 0.0, dz));
        apply(&mut c, &maps, 0.5);
        bricks.push(c.clone());
        background.push(c);
    }

    // UI state for the parallax controls.
    let mut depth_scale = 0.1f32;
    let mut layers = 32.0f32;
    let mut use_relief = true;
    let mut relief_steps = 8u32;

    let spin = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), 0.006);
    let spin_back = Quat::from_axis_angle(Vec3::Y, -0.002);
    while window.render_3d(&mut scene, &mut camera).await {
        let method = if use_relief {
            ParallaxMethod::Relief {
                max_steps: relief_steps,
            }
        } else {
            ParallaxMethod::Occlusion
        };
        for b in bricks.iter_mut() {
            b.set_parallax_scale(depth_scale);
            b.set_parallax_layers(layers);
            b.set_parallax_method(method);
        }
        cube.append_rotation(spin);
        for c in background.iter_mut() {
            c.append_rotation(spin_back);
        }

        window.draw_ui(|ctx| {
            egui::Window::new("Parallax")
                .default_width(280.0)
                .show(ctx, |ui| {
                    ui.add(egui::Slider::new(&mut depth_scale, 0.0..=0.3).text("depth scale"));
                    ui.add(egui::Slider::new(&mut layers, 1.0..=64.0).text("max layers"));
                    ui.separator();
                    ui.label("Mapping method:");
                    ui.radio_value(&mut use_relief, false, "Parallax occlusion");
                    ui.radio_value(&mut use_relief, true, "Relief (binary search)");
                    ui.add_enabled(
                        use_relief,
                        egui::Slider::new(&mut relief_steps, 1..=32).text("relief steps"),
                    );
                });
        });
    }
}
