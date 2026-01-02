use kiss3d::prelude::*;
use web_time::Instant;
use kiss3d::parry3d::shape::TriMesh;
use kiss3d::parry3d::transformation;
use kiss3d::parry3d::transformation::vhacd::VHACDParameters;
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

    let scale = Vec3::splat(scale);

    /*
     * Create the window.
     */
    let mut window = Window::new("Kiss3d: convex decomposition").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));

    /*
     * Convex decomposition.
     */
    let obj_path = Path::new(path);
    let mtl_path = Path::new("none");
    let teapot = obj::parse_file(obj_path, mtl_path, "none").unwrap();

    scene
        .add_obj(obj_path, mtl_path, scale)
        .set_surface_rendering_activation(false);

    let mut total_time = 0.0f64;
    for &(_, ref mesh, _) in teapot.iter() {
        match mesh.to_render_mesh() {
            Some(mut trimesh) => {
                trimesh.split_index_buffer(true);
                let idx: Vec<[u32; 3]> = trimesh
                    .indices
                    .as_split()
                    .iter()
                    .map(|idx| [idx[0][0], idx[1][0], idx[2][0]])
                    .collect();
                let coords = trimesh.coords.clone();
                let begin = Instant::now();
                let params = VHACDParameters {
                    concavity,
                    ..VHACDParameters::default()
                };
                let decomp = transformation::vhacd::VHACD::decompose(&params, &coords, &idx, true);
                let elapsed = begin.elapsed();
                total_time =
                    elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1000000000.0;

                for (vtx, idx) in decomp.compute_exact_convex_hulls(&coords, &idx) {
                    let r = random();
                    let g = random();
                    let b = random();

                    if let Ok(trimesh) = TriMesh::new(vtx, idx) {
                        scene.add_trimesh(trimesh, scale, true).set_color(Color::new(r, g, b, 1.0));
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
    while window.render_3d(&mut scene, &mut camera).await {}
}
