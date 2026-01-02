use kiss3d::prelude::*;
use kiss3d::procedural::path::StrokePattern;
use kiss3d::procedural::path::{ArrowheadCap, PolylinePath, PolylinePattern};
use kiss3d::procedural::RenderMesh;
use parry3d::shape::TriMesh;
use std::path::Path;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: procedural").await;
    let mut camera = OrbitCamera3d::new(Vec3::new(2.0, 1.0, 12.0), Vec3::new(2.0, 1.0, 2.0));
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 10.0, 10.0));

    /*
     * A cube.
     */
    let cube = kiss3d::procedural::cuboid(Vec3::new(0.7f32, 0.2, 0.4));
    let mut c = scene
        .add_render_mesh(cube, Vec3::splat(1.0))
        .set_position(Vec3::new(1.0, 0.0, 0.0));
    #[cfg(not(target_arch = "wasm32"))]
    c.set_texture_from_file(Path::new("./examples/media/kitten.png"), "kitten");

    /*
     * A sphere.
     */
    let sphere = kiss3d::procedural::sphere(0.4f32, 20, 20, true);
    let mut s = scene.add_render_mesh(sphere, Vec3::splat(1.0));
    #[cfg(not(target_arch = "wasm32"))]
    s.set_texture_with_name("kitten");

    /*
     * A capsule.
     */
    let capsule = kiss3d::procedural::capsule(0.4f32, 0.4f32, 20, 20);
    scene
        .add_render_mesh(capsule, Vec3::splat(1.0))
        .set_position(Vec3::new(-1.0, 0.0, 0.0))
        .set_color(BLUE);

    // /*
    //  * Triangulation.
    //  */
    // let to_triangulate = ncollide_transformation::triangulate(&[
    //     Vec3::new(5.0f32, 0.0, 0.0),
    //     Vec3::new(6.1, 0.0, 0.5),
    //     Vec3::new(7.4, 0.0, 0.5),
    //     Vec3::new(8.2, 0.0, 0.0),
    //     Vec3::new(5.1f32, 1.0, 0.0),
    //     Vec3::new(6.2, 1.5, 0.5),
    //     Vec3::new(7.2, 1.0, 0.5),
    //     Vec3::new(8.0, 1.3, 0.0),
    //     Vec3::new(5.3f32, 2.0, 0.0),
    //     Vec3::new(6.1, 2.2, 0.5),
    //     Vec3::new(7.3, 2.0, 0.5),
    //     Vec3::new(8.2, 2.4, 0.0),
    //     Vec3::new(5.2f32, 3.0, 0.0),
    //     Vec3::new(6.1, 2.9, 0.5),
    //     Vec3::new(7.4, 3.0, 0.5),
    //     Vec3::new(8.0, 3.1, 0.0),
    // ]);
    // let mut t = scene.add_trimesh(to_triangulate, Vec3::splat(1.0));
    // t.set_surface_rendering_activation(false);
    // t.set_lines_width(2.0);
    // t.set_color(GREEN);

    /*
     * A (non-rational) bicubic Bézier surface.
     */
    let control_points = [
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 2.0),
        Vec3::new(2.0, 0.0, 2.0),
        Vec3::new(3.0, 0.0, 0.0),
        Vec3::new(0.0f32, 1.0, 2.0),
        Vec3::new(1.0, 1.0, 3.0),
        Vec3::new(2.0, 1.0, 3.0),
        Vec3::new(3.0, 1.0, 2.0),
        Vec3::new(0.0f32, 2.0, 2.0),
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(2.0, 2.0, 3.0),
        Vec3::new(3.0, 2.0, 2.0),
        Vec3::new(0.0f32, 3.0, 0.0),
        Vec3::new(1.0, 3.0, 2.0),
        Vec3::new(2.0, 3.0, 2.0),
        Vec3::new(3.0, 3.0, 0.0),
    ];
    let bezier = kiss3d::procedural::bezier_surface(&control_points, 4, 4, 100, 100);
    scene
        .add_render_mesh(bezier, Vec3::splat(1.0))
        .set_position(Vec3::new(-1.5, -1.5, 0.0))
        .enable_backface_culling(false);

    // XXX: replace by an `add_mesh`.
    let control_polyhedra_gfx = scene
        .add_quad_with_vertices(&control_points, 4, 4)
        .set_position(Vec3::new(-1.5, -1.5, 0.0))
        .set_color(BLUE)
        .set_surface_rendering_activation(false)
        .set_lines_width(2.0, false);

    scene
        .add_mesh(
            control_polyhedra_gfx.data().get_object().mesh().clone(),
            Vec3::splat(1.0),
        )
        .set_position(Vec3::new(-1.5, -1.5, 0.0))
        .set_color(RED)
        .set_surface_rendering_activation(false)
        .set_points_size(10.0, false);

    /*
     * Path stroke.
     */
    let control_points = [
        Vec3::new(0.0f32, 1.0, 0.0),
        Vec3::new(2.0f32, 4.0, 2.0),
        Vec3::new(2.0f32, 1.0, 4.0),
        Vec3::new(4.0f32, 4.0, 6.0),
        Vec3::new(2.0f32, 1.0, 8.0),
        Vec3::new(2.0f32, 4.0, 10.0),
        Vec3::new(0.0f32, 1.0, 12.0),
        Vec3::new(-2.0f32, 4.0, 10.0),
        Vec3::new(-2.0f32, 1.0, 8.0),
        Vec3::new(-4.0f32, 4.0, 6.0),
        Vec3::new(-2.0f32, 1.0, 4.0),
        Vec3::new(-2.0f32, 4.0, 2.0),
    ];
    let bezier = kiss3d::procedural::bezier_curve(&control_points, 100);
    let mut path = PolylinePath::new(&bezier);
    let pattern = kiss3d::procedural::unit_circle(100);
    let start_cap = ArrowheadCap::new(1.5f32, 2.0, 0.0);
    let end_cap = ArrowheadCap::new(2.0f32, 2.0, 0.5);
    let mut pattern = PolylinePattern::new(pattern.coords(), true, start_cap, end_cap);
    let mesh = pattern.stroke(&mut path);
    scene
        .add_render_mesh(mesh, Vec3::new(0.5f32, 0.5, 0.5))
        .set_position(Vec3::new(4.0, -1.0, 0.0))
        .set_color(Color::new(1.0, 1.0, 0.0, 1.0));

    /*
     * Convex hull of 100,000 random 3d points.
     */
    let mut points = Vec::new();
    for _ in 0usize..100000 {
        points.push(Vec3::new(
            rand::random::<f32>() * 2.0,
            rand::random::<f32>() * 2.0,
            rand::random::<f32>() * 2.0,
        ));
    }

    let chull = parry3d::transformation::convex_hull(&points);
    scene
        .add_trimesh(
            TriMesh::new(chull.0, chull.1).unwrap(),
            Vec3::splat(1.0),
            false,
        )
        .set_position(Vec3::new(0.0, 2.0, -1.0))
        .set_color(GREEN)
        .set_lines_width(2.0, false)
        .set_surface_rendering_activation(false)
        .set_points_size(10.0, false);
    scene
        .add_render_mesh(RenderMesh::new(points, None, None, None), Vec3::splat(1.0))
        .set_color(BLUE)
        .set_position(Vec3::new(0.0, 2.0, -1.0))
        .set_points_size(2.0, false)
        .set_surface_rendering_activation(false);

    /*
     * Convex hull of 100,000 random 2d points.
     */
    let mut points2d = Vec::new();
    let origin = Vec2::new(3.0f32, 2.0);
    for _ in 0usize..100000 {
        points2d.push(origin + Vec2::new(rand::random::<f32>() * 2.0, rand::random::<f32>() * 2.0));
    }

    let polyline = parry2d::transformation::convex_hull(&points2d);

    /*
     *
     * Rendering.
     *
     */
    while window.render_3d(&mut scene, &mut camera).await {
        draw_polyline(&mut window, &polyline, &points2d);
    }
}

fn draw_polyline(window: &mut Window, polyline: &[Vec2], points: &[Vec2]) {
    for pt in polyline.windows(2) {
        window.draw_line(
            Vec3::new(pt[0].x, pt[0].y, 0.0),
            Vec3::new(pt[1].x, pt[1].y, 0.0),
            GREEN,
            10.0,
            false
        );
    }

    let last = polyline.len() - 1;
    window.draw_line(
        Vec3::new(polyline[0].x, polyline[0].y, 0.0),
        Vec3::new(polyline[last].x, polyline[last].y, 0.0),
        GREEN,
        6.0,
        false
    );

    for pt in points.iter() {
        window.draw_point(Vec3::new(pt.x, pt.y, 0.0), BLUE, 1.0);
    }

    for pt in polyline.iter() {
        window.draw_point(Vec3::new(pt.x, pt.y, 0.0), RED, 8.0);
    }
}
