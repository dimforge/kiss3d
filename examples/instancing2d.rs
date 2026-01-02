use kiss3d::prelude::*;

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: instancing 2D").await;
    let mut camera = PanZoomCamera2d::new(Vec2::ZERO, 0.1);
    let mut scene = SceneNode2d::empty();
    let rot_rect = 0.014;

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
            instances.push(InstanceData2d {
                position: Vec2::new((ii - shift) * 150.0, (jj - shift) * 150.0),
                deformation: Mat2::from_cols_array(&[1.0, ii * 0.004, jj * 0.004, 1.0]),
                color,
                lines_color: Some(lines_color),
                ..Default::default()
            });
        }
    }

    let mut rect = scene
        .add_rectangle(50.0, 150.0)
        .set_instances(&instances)
        .set_lines_width(2.0, true)
        .set_points_size(10.0, true);

    while window.render_2d(&mut scene, &mut camera).await {
        rect.append_rotation(rot_rect);
    }
}
