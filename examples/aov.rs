use kiss3d::prelude::*;
use std::path::Path;

// Renders a scene headlessly and saves the beauty image alongside three
// auxiliary render outputs (AOVs): linear depth, surface normals and colorized
// segmentation. They are produced by re-rendering the scene with dedicated
// materials — no path tracer involved.
#[kiss3d::main]
async fn main() {
    let mut surface = OffscreenSurface::new(1024, 768).await;
    surface.set_background_color(DARK_BLUE);

    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));

    // A few objects with explicit segmentation ids so the mask is reproducible.
    let mut cube = scene.add_cube(0.3, 0.3, 0.3);
    cube.set_color(RED)
        .rotate(Quat::from_axis_angle(Vec3::Y, 0.785))
        .rotate(Quat::from_axis_angle(Vec3::X, -0.6f32));
    cube.apply_to_object_mut(&mut |o| o.set_segmentation_id(1));

    let mut sphere = scene.add_sphere(0.2);
    sphere
        .set_color(LIME)
        .set_position(Vec3::new(0.5, 0.0, 0.0));
    sphere.apply_to_object_mut(&mut |o| o.set_segmentation_id(2));

    // Beauty pass (regular RGB), unchanged from normal rendering.
    let rgb = surface.render_image_3d(&mut scene, &mut camera).await;
    rgb.save(Path::new("aov_rgb.png")).unwrap();

    // Auxiliary outputs. Each re-renders the same scene/camera into a dedicated
    // target and reads it back.
    let depth = surface.snap_depth(&mut scene, &mut camera);
    depth.save(Path::new("aov_depth.png")).unwrap();

    let normals = surface.snap_normals(&mut scene, &mut camera);
    normals.save(Path::new("aov_normals.png")).unwrap();

    let seg = surface.snap_segmentation_colored(&mut scene, &mut camera);
    seg.save(Path::new("aov_segmentation.png")).unwrap();

    // Raw outputs are available too, e.g. metric depth and integer ids.
    let raw_depth = surface.snap_depth_raw(&mut scene, &mut camera);
    let raw_ids = surface.snap_segmentation(&mut scene, &mut camera);
    let max_id = raw_ids.iter().copied().max().unwrap_or(0);
    let nearest = raw_depth
        .iter()
        .copied()
        .filter(|d| *d > 0.0)
        .fold(f32::INFINITY, f32::min);

    println!(
        "Rendered aov_rgb.png, aov_depth.png, aov_normals.png, aov_segmentation.png ({:?})",
        surface.size()
    );
    println!("Nearest surface depth: {nearest:.3} world units; max segmentation id: {max_id}");
}
