use super::RenderMesh;
use glamx::Vec3;
use std::ptr;

// De-Casteljau algorithm.
// Evaluates the bezier curve with control points `control_points`.
#[doc(hidden)]
pub fn bezier_curve_at(control_points: &[Vec3], t: f32, cache: &mut Vec<Vec3>) -> Vec3 {
    if control_points.len() > cache.len() {
        let diff = control_points.len() - cache.len();
        cache.extend(std::iter::repeat_n(Vec3::ZERO, diff))
    }

    let cache = &mut cache[..];

    let t_1 = 1.0 - t;

    // XXX: not good if the objects are not POD.
    unsafe {
        ptr::copy_nonoverlapping(
            control_points.as_ptr(),
            cache.as_mut_ptr(),
            control_points.len(),
        );
    }

    for i in 1usize..control_points.len() {
        for j in 0usize..control_points.len() - i {
            cache[j] = cache[j] * t_1 + cache[j + 1] * t;
        }
    }

    cache[0]
}

// Evaluates the bezier curve with control points `control_points`.
#[doc(hidden)]
pub fn bezier_surface_at(
    control_points: &[Vec3],
    nupoints: usize,
    nvpoints: usize,
    u: f32,
    v: f32,
    ucache: &mut Vec<Vec3>,
    vcache: &mut Vec<Vec3>,
) -> Vec3 {
    if vcache.len() < nvpoints {
        let diff = nvpoints - vcache.len();
        vcache.extend(std::iter::repeat_n(Vec3::ZERO, diff));
    }

    // TODO: start with u or v, depending on which dimension has more control points.
    let vcache = &mut vcache[..];

    #[allow(clippy::needless_range_loop)]
    for i in 0..nvpoints {
        let start = i * nupoints;
        let end = start + nupoints;

        vcache[i] = bezier_curve_at(&control_points[start..end], u, ucache);
    }

    bezier_curve_at(&vcache[0..nvpoints], v, ucache)
}

/// Given a set of control points, generates a (non-rational) Bezier curve.
pub fn bezier_curve(control_points: &[Vec3], nsubdivs: usize) -> Vec<Vec3> {
    let mut coords = Vec::with_capacity(nsubdivs);
    let mut cache = Vec::new();
    let tstep = 1.0 / (nsubdivs as f32);
    let mut t = 0.0;

    while t <= 1.0 {
        coords.push(bezier_curve_at(control_points, t, &mut cache));
        t += tstep;
    }

    coords
}

/// Given a set of control points, generates a (non-rational) Bezier surface.
pub fn bezier_surface(
    control_points: &[Vec3],
    nupoints: usize,
    nvpoints: usize,
    usubdivs: usize,
    vsubdivs: usize,
) -> RenderMesh {
    assert!(nupoints * nvpoints == control_points.len());

    let mut surface = super::unit_quad(usubdivs, vsubdivs);

    {
        let uvs = &surface.uvs.as_ref().unwrap()[..];
        let coords = &mut surface.coords[..];

        let mut ucache = Vec::new();
        let mut vcache = Vec::new();

        for j in 0..vsubdivs + 1 {
            for i in 0..usubdivs + 1 {
                let id = i + j * (usubdivs + 1);
                coords[id] = bezier_surface_at(
                    control_points,
                    nupoints,
                    nvpoints,
                    uvs[id].x,
                    uvs[id].y,
                    &mut ucache,
                    &mut vcache,
                )
            }
        }

        // XXX: compute the normals manually.
        surface.normals = None;
    }

    surface
}
