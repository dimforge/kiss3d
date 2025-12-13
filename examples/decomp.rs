extern crate instant;
extern crate kiss3d;
extern crate nalgebra as na;
extern crate rand;

use instant::Instant;
use kiss3d::light::Light;
use kiss3d::loader::obj;
use kiss3d::window::Window;
use na::Vector3;
use parry3d::shape::TriMesh;
use parry3d::transformation;
use parry3d::transformation::vhacd::VHACDParameters;
use rand::random;
use std::env;
use std::path::Path;
use std::str::FromStr;

fn usage(exe_name: &str) {
    println!("Usage: {} obj_file scale clusters concavity", exe_name);
    println!();
    println!("Options:");
    println!("    obj_file  - the obj file to decompose.");
    println!("    scale     - the scale to apply to the displayed model.");
    println!("    concavity - the maximum concavity accepted by the decomposition.");
}

#[kiss3d::main]
async fn main() {
    /*
     * Parse arguments.
     */
    let mut args = env::args();
    let exname = args.next().unwrap();

    if args.len() != 3 {
        usage(&exname[..]);
        return;
    }

    let path = &args.next().unwrap()[..];
    let scale: f32 = FromStr::from_str(&args.next().unwrap()[..]).unwrap();
    let concavity: f32 = FromStr::from_str(&args.next().unwrap()[..]).unwrap();

    let scale = Vector3::from_element(scale);

    /*
     * Create the window.
     */
    let mut window = Window::new("Kiss3d: convex decomposition").await;

    /*
     * Convex decomposition.
     */
    let obj_path = Path::new(path);
    let mtl_path = Path::new("none");
    let teapot = obj::parse_file(obj_path, mtl_path, "none").unwrap();

    let mut m = window.add_obj(obj_path, mtl_path, scale);
    m.set_surface_rendering_activation(false);

    let mut total_time = 0.0f64;
    for &(_, ref mesh, _) in teapot.iter() {
        match mesh.to_render_mesh() {
            Some(mut trimesh) => {
                trimesh.split_index_buffer(true);
                let idx: Vec<[u32; 3]> = trimesh
                    .indices
                    .as_split()
                    .iter()
                    .map(|idx| [idx.x.x, idx.y.x, idx.z.x])
                    .collect();
                let begin = Instant::now();
                let params = VHACDParameters {
                    concavity,
                    ..VHACDParameters::default()
                };
                let decomp =
                    transformation::vhacd::VHACD::decompose(&params, &trimesh.coords, &idx, true);
                let elapsed = begin.elapsed();
                total_time =
                    elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1000000000.0;

                for (vtx, idx) in decomp.compute_exact_convex_hulls(&trimesh.coords, &idx) {
                    let r = random();
                    let g = random();
                    let b = random();

                    if let Ok(trimesh) = TriMesh::new(vtx, idx) {
                        let mut m = window.add_trimesh(trimesh, scale);
                        m.set_color(r, g, b);
                    }
                }
            }
            None => {}
        }
    }

    println!("Decomposition time: {}", total_time);

    /*
     *
     * Rendering.
     *
     */
    window.set_light(Light::StickToCamera);

    while window.render().await {}
}
