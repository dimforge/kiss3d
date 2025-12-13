extern crate kiss3d;
extern crate nalgebra as na;
extern crate rand;

use kiss3d::light::Light;
use kiss3d::procedural::path::StrokePattern;
use kiss3d::procedural::path::{ArrowheadCap, PolylinePath, PolylinePattern};
use kiss3d::procedural::RenderMesh;
use kiss3d::window::Window;
use na::{Point2, Point3, Translation3, Vector2, Vector3};
use parry3d::shape::TriMesh;
use std::path::Path;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: procedural").await;

    /*
     * A cube.
     */
    let cube = kiss3d::procedural::cuboid(&Vector3::new(0.7f32, 0.2, 0.4));
    let mut c = window.add_render_mesh(cube, Vector3::from_element(1.0));
    c.append_translation(&Translation3::new(1.0, 0.0, 0.0));
    #[cfg(not(target_arch = "wasm32"))]
    c.set_texture_from_file(Path::new("./examples/media/kitten.png"), "kitten");

    /*
     * A sphere.
     */
    let sphere = kiss3d::procedural::sphere(0.4f32, 20, 20, true);
    let mut s = window.add_render_mesh(sphere, Vector3::from_element(1.0));
    #[cfg(not(target_arch = "wasm32"))]
    s.set_texture_with_name("kitten");

    /*
     * A capsule.
     */
    let capsule = kiss3d::procedural::capsule(0.4f32, 0.4f32, 20, 20);
    let mut c = window.add_render_mesh(capsule, Vector3::from_element(1.0));
    c.append_translation(&Translation3::new(-1.0, 0.0, 0.0));
    c.set_color(0.0, 0.0, 1.0);

    // /*
    //  * Triangulation.
    //  */
    // let to_triangulate = ncollide_transformation::triangulate(&[
    //     Point3::new(5.0f32, 0.0, 0.0),
    //     Point3::new(6.1, 0.0, 0.5),
    //     Point3::new(7.4, 0.0, 0.5),
    //     Point3::new(8.2, 0.0, 0.0),
    //     Point3::new(5.1f32, 1.0, 0.0),
    //     Point3::new(6.2, 1.5, 0.5),
    //     Point3::new(7.2, 1.0, 0.5),
    //     Point3::new(8.0, 1.3, 0.0),
    //     Point3::new(5.3f32, 2.0, 0.0),
    //     Point3::new(6.1, 2.2, 0.5),
    //     Point3::new(7.3, 2.0, 0.5),
    //     Point3::new(8.2, 2.4, 0.0),
    //     Point3::new(5.2f32, 3.0, 0.0),
    //     Point3::new(6.1, 2.9, 0.5),
    //     Point3::new(7.4, 3.0, 0.5),
    //     Point3::new(8.0, 3.1, 0.0),
    // ]);
    // let mut t = window.add_trimesh(to_triangulate, Vector3::from_element(1.0));
    // t.set_surface_rendering_activation(false);
    // t.set_lines_width(2.0);
    // t.set_color(0.0, 1.0, 0.0);

    /*
     * A (non-rational) bicubic Bézier surface.
     */
    let control_points = [
        Point3::new(0.0f32, 0.0, 0.0),
        Point3::new(1.0, 0.0, 2.0),
        Point3::new(2.0, 0.0, 2.0),
        Point3::new(3.0, 0.0, 0.0),
        Point3::new(0.0f32, 1.0, 2.0),
        Point3::new(1.0, 1.0, 3.0),
        Point3::new(2.0, 1.0, 3.0),
        Point3::new(3.0, 1.0, 2.0),
        Point3::new(0.0f32, 2.0, 2.0),
        Point3::new(1.0, 2.0, 3.0),
        Point3::new(2.0, 2.0, 3.0),
        Point3::new(3.0, 2.0, 2.0),
        Point3::new(0.0f32, 3.0, 0.0),
        Point3::new(1.0, 3.0, 2.0),
        Point3::new(2.0, 3.0, 2.0),
        Point3::new(3.0, 3.0, 0.0),
    ];
    let bezier = kiss3d::procedural::bezier_surface(&control_points, 4, 4, 100, 100);
    let mut b = window.add_render_mesh(bezier, Vector3::from_element(1.0));
    b.append_translation(&Translation3::new(-1.5, -1.5, 0.0));
    b.enable_backface_culling(false);

    // XXX: replace by an `add_mesh`.
    let mut control_polyhedra_gfx = window.add_quad_with_vertices(&control_points, 4, 4);
    control_polyhedra_gfx.append_translation(&Translation3::new(-1.5, -1.5, 0.0));
    control_polyhedra_gfx.set_color(0.0, 0.0, 1.0);
    control_polyhedra_gfx.set_surface_rendering_activation(false);
    control_polyhedra_gfx.set_lines_width(2.0, false);

    let mut control_points_gfx = window.add_mesh(
        control_polyhedra_gfx.data().get_object().mesh().clone(),
        Vector3::from_element(1.0),
    );
    control_points_gfx.append_translation(&Translation3::new(-1.5, -1.5, 0.0));
    control_points_gfx.set_color(1.0, 0.0, 0.0);
    control_points_gfx.set_surface_rendering_activation(false);
    control_points_gfx.set_points_size(10.0, false);

    /*
     * Path stroke.
     */
    let control_points = [
        Point3::new(0.0f32, 1.0, 0.0),
        Point3::new(2.0f32, 4.0, 2.0),
        Point3::new(2.0f32, 1.0, 4.0),
        Point3::new(4.0f32, 4.0, 6.0),
        Point3::new(2.0f32, 1.0, 8.0),
        Point3::new(2.0f32, 4.0, 10.0),
        Point3::new(0.0f32, 1.0, 12.0),
        Point3::new(-2.0f32, 4.0, 10.0),
        Point3::new(-2.0f32, 1.0, 8.0),
        Point3::new(-4.0f32, 4.0, 6.0),
        Point3::new(-2.0f32, 1.0, 4.0),
        Point3::new(-2.0f32, 4.0, 2.0),
    ];
    let bezier = kiss3d::procedural::bezier_curve(&control_points, 100);
    let mut path = PolylinePath::new(&bezier);
    let pattern = kiss3d::procedural::unit_circle(100);
    let start_cap = ArrowheadCap::new(1.5f32, 2.0, 0.0);
    let end_cap = ArrowheadCap::new(2.0f32, 2.0, 0.5);
    let mut pattern = PolylinePattern::new(pattern.coords(), true, start_cap, end_cap);
    let mesh = pattern.stroke(&mut path);
    let mut m = window.add_render_mesh(mesh, Vector3::new(0.5f32, 0.5, 0.5));
    m.append_translation(&Translation3::new(4.0, -1.0, 0.0));
    m.set_color(1.0, 1.0, 0.0);

    /*
     * Convex hull of 100,000 random 3d points.
     */
    let mut points = Vec::new();
    for _ in 0usize..100000 {
        points.push(rand::random::<Point3<f32>>() * 2.0f32);
    }

    let chull = parry3d::transformation::convex_hull(&points[..]);
    let mut mhull = window.add_trimesh(
        TriMesh::new(chull.0, chull.1).unwrap(),
        Vector3::from_element(1.0),
    );
    let mut mpts = window.add_render_mesh(
        RenderMesh::new(points, None, None, None),
        Vector3::from_element(1.0),
    );
    mhull.append_translation(&Translation3::new(0.0, 2.0, -1.0));
    mhull.set_color(0.0, 1.0, 0.0);
    mhull.set_lines_width(2.0, false);
    mhull.set_surface_rendering_activation(false);
    mhull.set_points_size(10.0, false);
    mpts.set_color(0.0, 0.0, 1.0);
    mpts.append_translation(&Translation3::new(0.0, 2.0, -1.0));
    mpts.set_points_size(2.0, false);
    mpts.set_surface_rendering_activation(false);

    /*
     * Convex hull of 100,000 random 2d points.
     */
    let mut points = Vec::new();
    let origin = Point2::new(3.0f32, 2.0);
    for _ in 0usize..100000 {
        points.push(origin + rand::random::<Vector2<f32>>() * 2.0f32);
    }

    let points = &points[..];
    let polyline = parry2d::transformation::convex_hull(points);

    /*
     *
     * Rendering.
     *
     */
    window.set_light(Light::StickToCamera);

    while window.render().await {
        draw_polyline(&mut window, &polyline, points);
    }
}

fn draw_polyline(window: &mut Window, polyline: &[Point2<f32>], points: &[Point2<f32>]) {
    for pt in polyline.windows(2) {
        window.draw_line(
            &Point3::new(pt[0].x, pt[0].y, 0.0),
            &Point3::new(pt[1].x, pt[1].y, 0.0),
            &Point3::new(0.0, 1.0, 0.0),
            1.0,
        );
    }

    let last = polyline.len() - 1;
    window.draw_line(
        &Point3::new(polyline[0].x, polyline[0].y, 0.0),
        &Point3::new(polyline[last].x, polyline[last].y, 0.0),
        &Point3::new(0.0, 1.0, 0.0),
        1.0,
    );

    for pt in points.iter() {
        window.draw_point(
            &Point3::new(pt.x, pt.y, 0.0),
            &Point3::new(0.0, 0.0, 1.0),
            4.0,
        );
    }

    for pt in polyline.iter() {
        window.draw_point(
            &Point3::new(pt.x, pt.y, 0.0),
            &Point3::new(1.0, 0.0, 0.0),
            4.0,
        );
    }
}
