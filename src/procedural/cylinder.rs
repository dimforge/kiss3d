use super::utils;
use super::{IndexBuffer, RenderMesh};
use glamx::{Vec2, Vec3};

/// Generates a cylinder mesh with the specified dimensions.
///
/// Creates a cylinder oriented along the Y axis, with its center at the origin.
/// Both the top and bottom caps are included.
///
/// # Arguments
/// * `diameter` - The diameter of the cylinder
/// * `height` - The height of the cylinder
/// * `nsubdiv` - Number of subdivisions around the cylinder's circumference
///
/// # Returns
/// A `RenderMesh` containing the cylinder geometry with UVs and normals
///
/// # Example
/// ```no_run
/// # use kiss3d::procedural::cylinder;
/// // Create a cylinder with diameter 1.0, height 3.0, using 32 subdivisions
/// let cylinder_mesh = cylinder(1.0, 3.0, 32);
/// ```
pub fn cylinder(diameter: f32, height: f32, nsubdiv: u32) -> RenderMesh {
    let mut cylinder = unit_cylinder(nsubdiv);

    cylinder.scale_by(Vec3::new(diameter, height, diameter));

    cylinder
}

/// Generates a unit cylinder mesh.
///
/// Creates a cylinder with unit height and diameter, oriented along the Y axis.
/// The cylinder extends from y = -0.5 to y = +0.5.
///
/// # Arguments
/// * `nsubdiv` - Number of subdivisions around the cylinder's circumference
///
/// # Returns
/// A `RenderMesh` containing the unit cylinder geometry
///
/// # Example
/// ```no_run
/// # use kiss3d::procedural::unit_cylinder;
/// // Create a unit cylinder with 32 subdivisions
/// let cylinder_mesh = unit_cylinder(32);
/// ```
pub fn unit_cylinder(nsubdiv: u32) -> RenderMesh {
    let two_pi = std::f32::consts::TAU;
    let invsubdiv = 1.0 / (nsubdiv as f32);
    let dtheta = two_pi * invsubdiv;
    let mut coords = Vec::new();
    let mut indices = Vec::new();
    let mut normals: Vec<Vec3>;

    utils::push_circle(0.5, nsubdiv, dtheta, -0.5, &mut coords);

    normals = coords.clone();

    utils::push_circle(0.5, nsubdiv, dtheta, 0.5, &mut coords);

    utils::push_ring_indices(0, nsubdiv, nsubdiv, &mut indices);
    utils::push_filled_circle_indices(0, nsubdiv, &mut indices);
    utils::push_filled_circle_indices(nsubdiv, nsubdiv, &mut indices);

    let len = indices.len();
    let bottom_start_id = len - (nsubdiv as usize - 2);
    utils::reverse_clockwising(&mut indices[bottom_start_id..]);

    let mut indices = utils::split_index_buffer(&indices[..]);

    /*
     * Compute uvs.
     */
    // bottom ring uvs
    let mut uvs = Vec::with_capacity(coords.len());
    let mut curr_u = 0.0;
    for _ in 0..nsubdiv {
        uvs.push(Vec2::new(curr_u, 0.0));
        curr_u += invsubdiv;
    }

    // top ring uvs
    curr_u = 0.0;
    for _ in 0..nsubdiv {
        uvs.push(Vec2::new(curr_u, 1.0));
        curr_u += invsubdiv;
    }

    /*
     * Adjust normals.
     */
    for n in normals.iter_mut() {
        n.y = 0.0;
        *n = n.normalize();
    }

    normals.push(-Vec3::Y);
    normals.push(Vec3::Y);
    let nlen = normals.len() as u32;

    let top_start_id = len - 2 * (nsubdiv as usize - 2);

    for i in indices[..top_start_id].iter_mut() {
        if i[0][1] >= nsubdiv {
            i[0][1] -= nsubdiv;
        }
        if i[1][1] >= nsubdiv {
            i[1][1] -= nsubdiv;
        }
        if i[2][1] >= nsubdiv {
            i[2][1] -= nsubdiv;
        }
    }

    for i in indices[top_start_id..bottom_start_id].iter_mut() {
        i[0][1] = nlen - 2;
        i[1][1] = nlen - 2;
        i[2][1] = nlen - 2;
    }

    for i in indices[bottom_start_id..].iter_mut() {
        i[0][1] = nlen - 1;
        i[1][1] = nlen - 1;
        i[2][1] = nlen - 1;
    }

    RenderMesh::new(
        coords,
        Some(normals),
        Some(uvs),
        Some(IndexBuffer::Split(indices)),
    )
}
