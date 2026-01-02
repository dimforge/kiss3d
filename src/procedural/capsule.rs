use super::{sphere, utils};
use super::{IndexBuffer, RenderMesh};

/// Generates a capsule mesh.
///
/// A capsule is a cylinder with hemispherical caps on both ends.
/// The capsule is oriented along the Y axis with its center at the origin.
///
/// # Arguments
/// * `caps_diameter` - The diameter of the hemispherical caps (also the cylinder diameter)
/// * `cylinder_height` - The height of the cylindrical section (not including the caps)
/// * `ntheta_subdiv` - Number of subdivisions around the capsule (longitude)
/// * `nphi_subdiv` - Number of subdivisions along each hemisphere (latitude)
///
/// # Returns
/// A `RenderMesh` containing the capsule geometry
///
/// # Example
/// ```no_run
/// # use kiss3d::procedural::capsule;
/// // Create a capsule with cap diameter 1.0, cylinder height 2.0
/// // Total height will be 3.0 (2.0 cylinder + 1.0 from two caps)
/// let capsule_mesh = capsule(1.0, 2.0, 32, 16);
/// ```
pub fn capsule(
    caps_diameter: f32,
    cylinder_height: f32,
    ntheta_subdiv: u32,
    nphi_subdiv: u32,
) -> RenderMesh {
    let top = sphere::unit_hemisphere(ntheta_subdiv, nphi_subdiv);
    let RenderMesh {
        coords,
        normals,
        indices,
        ..
    } = top.clone();
    let mut bottom_coords = coords;
    let mut bottom_normals = normals.unwrap();
    let mut bottom_indices = indices.unwrap_unified();
    utils::reverse_clockwising(&mut bottom_indices[..]);

    let RenderMesh {
        coords,
        normals,
        indices,
        ..
    } = top;
    let mut top_coords = coords;
    let top_normals = normals.unwrap();
    let mut top_indices = indices.unwrap_unified();

    let half_height = cylinder_height * 0.5;

    // shift the top
    for coord in top_coords.iter_mut() {
        coord.x *= caps_diameter;
        coord.y = coord.y * caps_diameter + half_height;
        coord.z *= caps_diameter;
    }

    // flip + shift the bottom
    for coord in bottom_coords.iter_mut() {
        coord.x *= caps_diameter;
        coord.y = -(coord.y * caps_diameter) - half_height;
        coord.z *= caps_diameter;
    }

    // flip the bottom normals
    for normal in bottom_normals.iter_mut() {
        normal.y = -normal.y;
    }

    // shift the top index buffer
    let base_top_coords = bottom_coords.len() as u32;

    for idx in top_indices.iter_mut() {
        idx[0] += base_top_coords;
        idx[1] += base_top_coords;
        idx[2] += base_top_coords;
    }

    // merge all buffers
    bottom_coords.extend(top_coords);
    bottom_normals.extend(top_normals);
    bottom_indices.extend(top_indices);

    // attach the two caps
    utils::push_ring_indices(0, base_top_coords, ntheta_subdiv, &mut bottom_indices);

    // TODO: uvs
    RenderMesh::new(
        bottom_coords,
        Some(bottom_normals),
        None,
        Some(IndexBuffer::Unified(bottom_indices)),
    )
}
