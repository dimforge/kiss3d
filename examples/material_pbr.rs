//! Extended PBR surface properties on the rasterizer.
//!
//! A 6x6 grid of spheres lit by an equirectangular skybox (so every material
//! reflects a real environment). Each row isolates one StandardMaterial extension
//! and sweeps it left-to-right across the six columns, so you can read off how the
//! parameter changes the look. Rows top-to-bottom:
//!   - roughness     0 -> 1   (metallic base: sharp -> blurry reflections)
//!   - metallic      0 -> 1   (dielectric -> tinted metal)
//!   - clearcoat     0 -> 1   (matte base gains a glossy coat)
//!   - anisotropy   -1 -> 1   (highlight stretches along the tangent)
//!   - reflectance   0 -> 1   (dielectric specular brightness)
//!   - transmission  0 -> 1   (refractive glass: opaque -> clear, refracts the scene)
//!
//! The bottom row is real screen-space refractive glass: each sphere refracts the
//! rendered scene behind it (the skybox), bent by its index of refraction and
//! thickness and tinted by a volume attenuation color, with a Fresnel reflection
//! that strengthens toward the rim. Raise a sphere's `roughness` for frosted glass.
//!
//! Run with: `cargo run --example material_pbr`.

use kiss3d::prelude::*;
use std::path::Path;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: material_pbr").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 0.0, 18.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();

    // The skybox is the dominant light source (image-based lighting): the material
    // differences come through in its reflections, and the glass row refracts it. A
    // soft directional key adds specular highlights.
    window.set_skybox_from_file(Path::new("./examples/media/skybox.png"));
    window.set_ambient(0.0);
    window.transmission_settings_mut().steps = 4;
    scene.add_light(Light::directional(Vec3::new(-0.4, -0.8, -0.6)).with_intensity(2.0));

    let cols = 6;
    let rows = 6;
    let spacing = 2.2;
    for r in 0..rows {
        for c in 0..cols {
            let x = (c as f32 - (cols as f32 - 1.0) * 0.5) * spacing;
            let y = ((rows as f32 - 1.0) * 0.5 - r as f32) * spacing;
            let t = c as f32 / (cols as f32 - 1.0); // 0 -> 1 across the columns

            let mut s = scene.add_sphere(0.9)
             .translate(Vec3::new(x, y, 0.0));

            // Each row picks a base material that makes its swept extension most
            // visible, then overrides that one parameter with the column ramp `t`.
            match r {
                // Roughness: a polished metal smears the reflected skybox as it rises.
                0 => {
                    s.set_color(Color::new(0.95, 0.95, 0.97, 1.0))
                     .set_metallic(1.0)
                     .set_roughness(t);
                }
                // Metallic: a glossy dielectric turns into tinted gold.
                1 => {
                    s.set_color(Color::new(0.95, 0.70, 0.30, 1.0))
                     .set_roughness(0.2)
                     .set_metallic(t);
                }
                // Clearcoat: a matte red surface gains a sharp glossy coat on top.
                2 => {
                    s.set_color(Color::new(0.80, 0.15, 0.12, 1.0))
                     .set_metallic(0.0)
                     .set_roughness(0.75)
                     .set_clearcoat(t, 0.05);
                }
                // Anisotropy: a brushed metal stretches its highlight with the tangent.
                3 => {
                    s.set_color(Color::new(0.85, 0.85, 0.88, 1.0))
                     .set_metallic(1.0)
                     .set_roughness(0.4)
                     .set_anisotropy(t * 2.0 - 1.0, 0.0); // -1 -> 1
                }
                // Reflectance: F0 of a smooth blue dielectric, dim to bright specular.
                4 => {
                    s.set_color(Color::new(0.15, 0.30, 0.80, 1.0))
                     .set_metallic(0.0)
                     .set_roughness(0.15)
                     .set_reflectance(t);
                }
                // Transmission: refractive glass. As it rises the sphere goes from a
                // solid dielectric to clear glass that refracts the scene behind it
                // (bent by ior/thickness), with a cool volume tint and a Fresnel rim
                // reflection. Smooth here; raise roughness for frosted glass.
                5 => {
                    s.set_color(Color::new(0.85, 0.92, 1.0, 1.0))
                     .set_reflectance(0.8)
                     .set_roughness(0.2)
                     .set_ior(1.5)
                     .set_thickness(0.4)
                     .set_attenuation(Color::new(0.82, 0.92, 1.0, 1.0), 8.0)
                     .set_transmission(t);
                }
                _ => {}
            }
        }
    }

    // A row of bright colored spheres behind the bottom (glass) row, so its
    // refraction has vivid structure to bend and magnify — clear glass over the
    // blank sky is nearly invisible. The other rows keep the open sky behind them.
    {
        let bottom_y = -((rows as f32 - 1.0) * 0.5) * spacing;
        let palette = [
            Color::new(0.9, 0.15, 0.15, 1.0),
            Color::new(0.95, 0.55, 0.1, 1.0),
            Color::new(0.95, 0.9, 0.15, 1.0),
            Color::new(0.2, 0.75, 0.25, 1.0),
            Color::new(0.15, 0.55, 0.95, 1.0),
            Color::new(0.6, 0.2, 0.8, 1.0),
        ];
        for (i, c) in palette.iter().enumerate() {
            let x = (i as f32 - (palette.len() as f32 - 1.0) * 0.5) * 2.2;
            let mut b = scene.add_sphere(0.9);
            b.translate(Vec3::new(x, bottom_y, -3.0));
            b.set_color(*c);
            b.set_roughness(0.5);
            // A touch of emission so the refracted colors stay vivid.
            b.set_emissive(Color::new(c.r * 0.4, c.g * 0.4, c.b * 0.4, 1.0));
        }
    }

    // One label per row, listed top-to-bottom to match the grid.
    let font = Font::default();
    let labels = [
        "row 1  roughness     0 -> 1",
        "row 2  metallic      0 -> 1",
        "row 3  clearcoat     0 -> 1",
        "row 4  anisotropy   -1 -> 1",
        "row 5  reflectance   0 -> 1",
        "row 6  transmission  0 -> 1",
    ];

    while window.render_3d(&mut scene, &mut camera).await {
        let scale = 40.0;
        for (i, label) in labels.iter().enumerate() {
            let y = 10.0 + i as f32 * scale;
            window.draw_text(label, Vec2::new(10.0, y), scale, &font, WHITE);
        }
    }
}
