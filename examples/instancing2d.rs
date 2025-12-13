extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::camera::FixedView;
use kiss3d::planar_camera::Sidescroll;
use kiss3d::scene::PlanarInstanceData;
use kiss3d::window::Window;
use na::{Matrix2, Point2, UnitComplex};

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: instancing 2D").await;
    let mut rect = window.add_rectangle(50.0, 150.0);

    let rot_rect = UnitComplex::new(0.014);

    let mut instances = vec![];
    let count = 100;

    for i in 0..=count {
        for j in 0..=count {
            let shift = count as f32 / 2.0;
            let ii = i as f32;
            let jj = j as f32;
            let color = [ii / count as f32, jj / count as f32, 1.0, 1.0];
            let mut lines_color = color.map(|c| 1.0 - c);
            lines_color[3] = 1.0;
            instances.push(PlanarInstanceData {
                position: Point2::new((ii - shift) * 150.0, (jj - shift) * 150.0),
                deformation: Matrix2::new(1.0, ii * 0.004, jj * 0.004, 1.0),
                color,
                lines_color: Some(lines_color),
                ..Default::default()
            });
        }
    }

    rect.set_instances(&instances);
    rect.set_lines_width(2.0, true);
    rect.set_points_size(10.0, true);

    let mut camera2d = Sidescroll::new();
    let mut camera3d = FixedView::new();

    while !window.should_close() {
        rect.prepend_to_local_rotation(&rot_rect);
        window
            .render_with_cameras(&mut camera3d, &mut camera2d)
            .await;
    }
}
