extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::window::Window;
use kiss3d::{light::Light, resource::TextureManager};
use na::Translation3;
use std::path::Path;
use std::time::Instant;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: texturing-mipmaps").await;
    let tex_path = Path::new("./examples/media/checkerboard.png");

    // Show two spheres that are scaled up and down, one without mipmaps and one
    // without mipmaps.
    TextureManager::get_global_manager(|tm| tm.set_generate_mipmaps(false));
    let mut q1 = window.add_sphere(1.0);
    q1.set_texture_from_file(tex_path, "no-mipmaps");
    q1.set_local_translation(Translation3::new(0.3, 0.0, 0.0));

    TextureManager::get_global_manager(|tm| tm.set_generate_mipmaps(true));
    let mut q2 = window.add_sphere(1.0);
    q2.set_texture_from_file(tex_path, "with-mipmaps");
    q2.set_local_translation(Translation3::new(-0.3, 0.0, 0.0));

    window.set_light(Light::StickToCamera);

    let start = Instant::now();
    while window.render().await {
        let scale = 0.25 + 0.2 * (Instant::now() - start).as_secs_f32().cos();
        for c in [&mut q1, &mut q2] {
            c.set_local_scale(scale, scale, scale);
        }
    }
}
