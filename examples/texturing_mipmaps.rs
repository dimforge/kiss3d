use kiss3d::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
// `web_time::Instant` is `std::time::Instant` on native but a working shim on
// wasm, where `std::time::Instant::now()` panics ("time not implemented").
use web_time::Instant;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: texturing-mipmaps").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, -10.0));

    // Show two spheres that are scaled up and down, one without mipmaps and one
    // with mipmaps. The checkerboard is read from the path on native and embedded
    // into the binary on wasm (no filesystem).
    TextureManager::get_global_manager(|tm| tm.set_generate_mipmaps(false));
    let mut q1 = scene.add_sphere(1.0);
    #[cfg(not(target_arch = "wasm32"))]
    q1.set_texture_from_file(Path::new("./examples/media/checkerboard.png"), "no-mipmaps");
    #[cfg(target_arch = "wasm32")]
    q1.set_texture_from_memory(include_bytes!("media/checkerboard.png"), "no-mipmaps");
    q1.translate(Vec3::new(0.3, 0.0, 0.0));

    TextureManager::get_global_manager(|tm| tm.set_generate_mipmaps(true));
    let mut q2 = scene.add_sphere(1.0);
    #[cfg(not(target_arch = "wasm32"))]
    q2.set_texture_from_file(
        Path::new("./examples/media/checkerboard.png"),
        "with-mipmaps",
    );
    #[cfg(target_arch = "wasm32")]
    q2.set_texture_from_memory(include_bytes!("media/checkerboard.png"), "with-mipmaps");
    q2.translate(Vec3::new(-0.3, 0.0, 0.0));

    let start = Instant::now();
    while window.render_3d(&mut scene, &mut camera).await {
        let scale = 0.25 + 0.2 * (Instant::now() - start).as_secs_f32().cos();
        for c in [&mut q1, &mut q2] {
            c.set_local_scale(scale, scale, scale);
        }
    }
}
