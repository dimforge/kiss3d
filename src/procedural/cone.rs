use super::utils;
use super::{IndexBuffer, RenderMesh};
use glamx::Vec3;

/// Generates a cone mesh with the specified dimensions.
///
/// Creates a cone pointing upward along the positive Y axis, with its base
/// centered at y = -height/2 and apex at y = +height/2.
///
/// # Arguments
/// * `diameter` - The diameter of the cone's base
/// * `height` - The height of the cone (distance from base to apex)
/// * `nsubdiv` - Number of subdivisions around the base circle
///
/// # Returns
/// A `RenderMesh` containing the cone geometry
///
/// # Example
/// ```no_run
/// # use kiss3d::procedural::cone;
/// // Create a cone with base diameter 2.0, height 3.0, using 32 subdivisions
/// let cone_mesh = cone(2.0, 3.0, 32);
/// ```
pub fn cone(diameter: f32, height: f32, nsubdiv: u32) -> RenderMesh {
    let mut cone = unit_cone(nsubdiv);

    cone.scale_by(Vec3::new(diameter, height, diameter));

    cone
}

/// Generates a unit cone mesh.
///
/// Creates a cone with unit height and unit diameter, pointing upward along the Y axis.
/// The base is at y = -0.5 and the apex is at y = +0.5.
///
/// # Arguments
/// * `nsubdiv` - Number of subdivisions around the base circle
///
/// # Returns
/// A `RenderMesh` containing the unit cone geometry
///
/// # Example
/// ```no_run
/// # use kiss3d::procedural::unit_cone;
/// // Create a unit cone with 32 subdivisions
/// let cone_mesh = unit_cone(32);
/// ```
pub fn unit_cone(nsubdiv: u32) -> RenderMesh {
    let two_pi = std::f32::consts::TAU;
    let dtheta = two_pi / (nsubdiv as f32);
    let mut coords = Vec::new();
    let mut indices = Vec::new();
    let mut normals: Vec<Vec3>;

    utils::push_circle(0.5, nsubdiv, dtheta, -0.5, &mut coords);

    normals = coords.clone();

    coords.push(Vec3::new(0.0, 0.5, 0.0));

    utils::push_degenerate_top_ring_indices(0, coords.len() as u32 - 1, nsubdiv, &mut indices);
    utils::push_filled_circle_indices(0, nsubdiv, &mut indices);

    /*
     * Normals.
     */
    let mut indices = utils::split_index_buffer(&indices[..]);

    // Adjust the normals:
    let shift = 0.05f32 / 0.475;
    for n in normals.iter_mut() {
        n.y += shift;
        *n = n.normalize();
    }

    // Normal for the basis.
    normals.push(Vec3::new(0.0, -1.0, 0.0));

    let ilen = indices.len();
    let nlen = normals.len() as u32;
    for (id, i) in indices[..ilen - (nsubdiv as usize - 2)]
        .iter_mut()
        .enumerate()
    {
        i[1][1] = id as u32;
    }

    for i in indices[ilen - (nsubdiv as usize - 2)..].iter_mut() {
        i[0][1] = nlen - 1;
        i[1][1] = nlen - 1;
        i[2][1] = nlen - 1;
    }

    // Normal for the body.

    RenderMesh::new(
        coords,
        Some(normals),
        None,
        Some(IndexBuffer::Split(indices)),
    )

    // TODO: uvs
}
