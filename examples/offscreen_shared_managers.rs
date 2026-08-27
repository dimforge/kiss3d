//! Regression check: a scene built against the main window's resource
//! managers, then rendered through an offscreen surface created afterwards,
//! with shaded renders interleaved with AOV read-backs on more objects than
//! the per-object uniform buffer's initial capacity. Creating the surface
//! used to replace the global managers, orphaning the objects' material so
//! its uniform buffer was never cleared or flushed again.

use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    // Like a viewer that owns a headless main window, builds its scene, and
    // only then opens offscreen sensor surfaces sharing the context: the
    // objects must keep working with the resource managers they were built
    // against.
    let _main = Window::new_headless_with_setup(64, 64, CanvasSetup::default()).await;
    let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 6.0, 12.0), Vec3::ZERO);
    let mut scene = SceneNode3d::empty();
    scene.add_directional_light(Vec3::new(0.3, -0.4, -1.0));
    // 100 objects: four unflushed prepares exceed the 256-entry default.
    for i in 0..100 {
        let mut cube = scene.add_cube(0.3, 0.3, 0.3);
        let x = (i % 10) as f32 - 4.5;
        let z = (i / 10) as f32 - 4.5;
        cube.set_position(Vec3::new(x, 0.0, z));
        cube.apply_to_object_mut(&mut |o| o.set_segmentation_id(i as u32 + 1));
    }
    let mut surface = OffscreenSurface::new(320, 240).await;
    for round in 0..4 {
        surface.render_3d(&mut scene, &mut camera).await;
        let _ = surface.snap_image();
        let depth = surface.snap_depth_raw(&mut scene, &mut camera);
        surface.render_3d(&mut scene, &mut camera).await;
        let ids = surface.snap_segmentation(&mut scene, &mut camera);
        surface.render_3d(&mut scene, &mut camera).await;
        let hit = depth.iter().filter(|d| **d > 0.0).count();
        let max_id = ids.iter().copied().max().unwrap_or(0);
        println!("round {round}: depth hits {hit}, max id {max_id}");
    }
    println!("ok");
}
