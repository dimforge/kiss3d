use super::utils;
use super::RenderPolyline;
use super::{IndexBuffer, RenderMesh};
use glamx::{Vec2, Vec3};

/// Generates a UV sphere with the specified diameter.
///
/// Creates a sphere mesh subdivided by latitude (theta) and longitude (phi) lines.
/// The sphere is generated using a UV-mapping friendly topology.
///
/// # Arguments
/// * `diameter` - The diameter of the sphere
/// * `ntheta_subdiv` - Number of subdivisions around the sphere (longitude)
/// * `nphi_subdiv` - Number of subdivisions from top to bottom (latitude)
/// * `generate_uvs` - Whether to generate UV texture coordinates
///
/// # Returns
/// A `RenderMesh` containing the sphere geometry
///
/// # Example
/// ```no_run
/// # use kiss3d::procedural::sphere;
/// // Create a sphere with diameter 2.0, 32 longitude divisions, 16 latitude divisions
/// let sphere_mesh = sphere(2.0, 32, 16, true);
/// ```
pub fn sphere(
    diameter: f32,
    ntheta_subdiv: u32,
    nphi_subdiv: u32,
    generate_uvs: bool,
) -> RenderMesh {
    let mut sphere = unit_sphere(ntheta_subdiv, nphi_subdiv, generate_uvs);

    sphere.scale_by_scalar(diameter);

    sphere
}

/// Generates a unit sphere centered at the origin with diameter 1.0.
///
/// Creates a sphere mesh with unit diameter (radius 0.5) subdivided by
/// latitude and longitude lines.
///
/// # Arguments
/// * `ntheta_subdiv` - Number of subdivisions around the sphere (longitude)
/// * `nphi_subdiv` - Number of subdivisions from top to bottom (latitude)
/// * `generate_uvs` - Whether to generate UV texture coordinates
///
/// # Returns
/// A `RenderMesh` containing the unit sphere geometry
///
/// # Example
/// ```no_run
/// # use kiss3d::procedural::unit_sphere;
/// // Create a unit sphere with 32x16 subdivisions and UVs
/// let sphere_mesh = unit_sphere(32, 16, true);
/// ```
pub fn unit_sphere(ntheta_subdiv: u32, nphi_subdiv: u32, generate_uvs: bool) -> RenderMesh {
    if generate_uvs {
        unit_sphere_with_uvs(ntheta_subdiv, nphi_subdiv)
    } else {
        unit_sphere_without_uvs(ntheta_subdiv, nphi_subdiv)
    }
}

// TODO: f32{theta,phi}_subdiv are not the right names.
fn unit_sphere_without_uvs(ntheta_subdiv: u32, nphi_subdiv: u32) -> RenderMesh {
    let pi = std::f32::consts::PI;
    let two_pi = std::f32::consts::TAU;
    let pi_two = std::f32::consts::FRAC_PI_2;
    let dtheta = two_pi / (ntheta_subdiv as f32);
    let dphi = pi / (nphi_subdiv as f32);

    let mut coords = Vec::new();
    let mut curr_phi = -pi_two + dphi;

    // coords.
    coords.push(Vec3::new(0.0, -1.0, 0.0));

    for _ in 0..nphi_subdiv - 1 {
        utils::push_circle(
            curr_phi.cos(),
            ntheta_subdiv,
            dtheta,
            curr_phi.sin(),
            &mut coords,
        );
        curr_phi += dphi;
    }

    coords.push(Vec3::new(0.0, 1.0, 0.0));

    // the normals are the same as the coords.
    let normals: Vec<Vec3> = coords.clone();

    // index buffer
    let mut idx = Vec::new();

    utils::push_degenerate_top_ring_indices(1, 0, ntheta_subdiv, &mut idx);

    utils::reverse_clockwising(&mut idx[..]);

    for i in 0..nphi_subdiv - 2 {
        let bottom = 1 + i * ntheta_subdiv;
        let up = bottom + ntheta_subdiv;
        utils::push_ring_indices(bottom, up, ntheta_subdiv, &mut idx);
    }

    utils::push_degenerate_top_ring_indices(
        1 + (nphi_subdiv - 2) * ntheta_subdiv,
        coords.len() as u32 - 1,
        ntheta_subdiv,
        &mut idx,
    );

    let mut res = RenderMesh::new(coords, Some(normals), None, Some(IndexBuffer::Unified(idx)));
    res.scale_by_scalar(0.5);

    res
}

fn unit_sphere_with_uvs(ntheta_subdiv: u32, nphi_subdiv: u32) -> RenderMesh {
    let pi = std::f32::consts::PI;
    let two_pi = std::f32::consts::TAU;
    let pi_two = std::f32::consts::FRAC_PI_2;
    let duvtheta = 1.0 / (ntheta_subdiv as f32); // step of uv.x coordinates.
    let duvphi = 1.0 / (nphi_subdiv as f32); // step of uv.y coordinates.
    let dtheta = two_pi * duvtheta;
    let dphi = pi * duvphi;

    let mut coords = Vec::new();
    let mut curr_phi = -pi_two;

    for _ in 0..nphi_subdiv + 1 {
        utils::push_circle(
            curr_phi.cos(),
            ntheta_subdiv + 1,
            dtheta,
            curr_phi.sin(),
            &mut coords,
        );
        curr_phi += dphi;
    }

    // the normals are the same as the coords
    let normals: Vec<Vec3> = coords.clone();

    // index buffer
    let mut idx = Vec::new();

    for i in 0..nphi_subdiv {
        let bottom = i * (ntheta_subdiv + 1);
        let up = bottom + (ntheta_subdiv + 1);
        utils::push_open_ring_indices(bottom, up, ntheta_subdiv + 1, &mut idx);
    }

    let mut uvs = Vec::new();
    let mut curr_uvphi = 0.0;

    for _ in 0..nphi_subdiv + 1 {
        let mut curr_uvtheta = 0.0;

        for _ in 0..ntheta_subdiv + 1 {
            uvs.push(Vec2::new(curr_uvtheta, curr_uvphi));
            curr_uvtheta += duvtheta;
        }

        curr_uvphi += duvphi;
    }

    let mut res = RenderMesh::new(
        coords,
        Some(normals),
        Some(uvs),
        Some(IndexBuffer::Unified(idx)),
    );

    res.scale_by_scalar(0.5);

    res
}

/// Creates a hemisphere with unit diameter.
///
/// Generates the upper half of a unit sphere (y >= 0), with diameter 1.0 (radius 0.5).
/// The base of the hemisphere lies on the XZ plane.
///
/// # Arguments
/// * `ntheta_subdiv` - Number of subdivisions around the hemisphere (longitude)
/// * `nphi_subdiv` - Number of subdivisions from base to top (latitude)
///
/// # Returns
/// A `RenderMesh` containing the hemisphere geometry
///
/// # Example
/// ```no_run
/// # use kiss3d::procedural::unit_hemisphere;
/// // Create a hemisphere with 32 longitude and 16 latitude subdivisions
/// let hemisphere_mesh = unit_hemisphere(32, 16);
/// ```
pub fn unit_hemisphere(ntheta_subdiv: u32, nphi_subdiv: u32) -> RenderMesh {
    let two_pi = std::f32::consts::TAU;
    let pi_two = std::f32::consts::FRAC_PI_2;
    let dtheta = two_pi / (ntheta_subdiv as f32);
    let dphi = pi_two / (nphi_subdiv as f32);

    let mut coords = Vec::new();
    let mut curr_phi = 0.0f32;

    for _ in 0..nphi_subdiv - 1 {
        utils::push_circle(
            curr_phi.cos(),
            ntheta_subdiv,
            dtheta,
            curr_phi.sin(),
            &mut coords,
        );
        curr_phi += dphi;
    }

    coords.push(Vec3::new(0.0, 1.0, 0.0));

    let mut idx = Vec::new();

    for i in 0..nphi_subdiv - 2 {
        utils::push_ring_indices(
            i * ntheta_subdiv,
            (i + 1) * ntheta_subdiv,
            ntheta_subdiv,
            &mut idx,
        );
    }

    utils::push_degenerate_top_ring_indices(
        (nphi_subdiv - 2) * ntheta_subdiv,
        coords.len() as u32 - 1,
        ntheta_subdiv,
        &mut idx,
    );

    // Result
    let normals: Vec<Vec3> = coords.clone();
    // TODO: uvs
    let mut out = RenderMesh::new(coords, Some(normals), None, Some(IndexBuffer::Unified(idx)));

    // set the radius to 0.5
    out.scale_by_scalar(0.5);

    out
}

/// Creates a 2D circle polyline lying on the XY plane.
///
/// Generates a circle as a polyline (not a filled mesh) with the specified diameter.
/// The circle lies on the XY plane (z = 0) and is centered at the origin.
///
/// # Arguments
/// * `diameter` - The diameter of the circle
/// * `nsubdivs` - Number of line segments to approximate the circle
///
/// # Returns
/// A `RenderPolyline` containing the circle's vertices
///
/// # Example
/// ```no_run
/// # use kiss3d::procedural::circle;
/// // Create a circle with diameter 2.0 using 64 segments
/// let circle_polyline = circle(2.0, 64);
/// ```
pub fn circle(diameter: f32, nsubdivs: u32) -> RenderPolyline {
    let two_pi = std::f32::consts::TAU;
    let dtheta = two_pi / nsubdivs as f32;

    let mut pts = Vec::with_capacity(nsubdivs as usize);

    utils::push_xy_arc(diameter / 2.0, nsubdivs, dtheta, &mut pts);

    // TODO: normals

    RenderPolyline::new(pts, None)
}

/// Creates a 2D unit circle polyline lying on the XY plane.
///
/// Generates a circle polyline with diameter 1.0 (radius 0.5) on the XY plane.
///
/// # Arguments
/// * `nsubdivs` - Number of line segments to approximate the circle
///
/// # Returns
/// A `RenderPolyline` containing the circle's vertices
///
/// # Example
/// ```no_run
/// # use kiss3d::procedural::unit_circle;
/// // Create a unit circle using 64 segments
/// let circle_polyline = unit_circle(64);
/// ```
pub fn unit_circle(nsubdivs: u32) -> RenderPolyline {
    circle(1.0, nsubdivs)
}
