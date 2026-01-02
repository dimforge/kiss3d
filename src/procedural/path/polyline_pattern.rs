use crate::procedural::path::{CurveSampler, PathSample, StrokePattern};
use crate::procedural::render_mesh::{IndexBuffer, RenderMesh};
use crate::procedural::utils;
use glamx::{Pose3, Vec2, Vec3};

/// A pattern composed of polyline and two caps.
pub struct PolylinePattern<C1, C2> {
    pattern: Vec<Vec3>,
    closed: bool,
    last_start_id: u32,
    start_cap: C1,
    end_cap: C2,
}

/// Trait to be implemented by caps compatible with a `PolylinePattern`.
pub trait PolylineCompatibleCap {
    /// Generates the mesh for the cap at the beginning of a path.
    fn gen_start_cap(
        &self,
        attach_id: u32,
        pattern: &[Vec3],
        pt: Vec3,
        dir: Vec3,
        closed: bool,
        coords: &mut Vec<Vec3>,
        indices: &mut Vec<[u32; 3]>,
    );

    /// Generates the mesh for the cap at the end of a path.
    fn gen_end_cap(
        &self,
        attach_id: u32,
        pattern: &[Vec3],
        pt: Vec3,
        dir: Vec3,
        closed: bool,
        coords: &mut Vec<Vec3>,
        indices: &mut Vec<[u32; 3]>,
    );
}

impl<C1, C2> PolylinePattern<C1, C2>
where
    C1: PolylineCompatibleCap,
    C2: PolylineCompatibleCap,
{
    /// Creates a new polyline pattern.
    pub fn new(
        pattern: &[Vec2],
        closed: bool,
        start_cap: C1,
        end_cap: C2,
    ) -> PolylinePattern<C1, C2> {
        let mut coords3d = Vec::with_capacity(pattern.len());

        for v in pattern.iter() {
            coords3d.push(Vec3::new(v.x, v.y, 0.0));
        }

        PolylinePattern {
            pattern: coords3d,
            closed,
            last_start_id: 0,
            start_cap,
            end_cap,
        }
    }
}

impl<C1, C2> StrokePattern for PolylinePattern<C1, C2>
where
    C1: PolylineCompatibleCap,
    C2: PolylineCompatibleCap,
{
    fn stroke<C: CurveSampler>(&mut self, sampler: &mut C) -> RenderMesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let npts = self.pattern.len() as u32;
        // TODO: collect the normals too.
        // let mut normals  = Vec::new();

        loop {
            let next = sampler.next();

            // second match to add the inner triangles.
            match next {
                PathSample::StartPoint(ref pt, ref dir)
                | PathSample::InnerPoint(ref pt, ref dir)
                | PathSample::EndPoint(ref pt, ref dir) => {
                    let mut new_polyline = self.pattern.clone();

                    let transform = if dir.x == 0.0 && dir.z == 0.0 {
                        // TODO: this might not be enough to avoid singularities.
                        Pose3::face_towards(*pt, *pt + *dir, Vec3::X)
                    } else {
                        Pose3::face_towards(*pt, *pt + *dir, Vec3::Y)
                    };

                    for p in &mut new_polyline {
                        *p = transform * *p;
                    }

                    let new_start_id = vertices.len() as u32;

                    vertices.extend(new_polyline);

                    if new_start_id != 0 {
                        if self.closed {
                            utils::push_ring_indices(
                                new_start_id,
                                self.last_start_id,
                                npts,
                                &mut indices,
                            );
                        } else {
                            utils::push_open_ring_indices(
                                new_start_id,
                                self.last_start_id,
                                npts,
                                &mut indices,
                            );
                        }

                        self.last_start_id = new_start_id;
                    }
                }
                PathSample::EndOfSample => {
                    return RenderMesh::new(
                        vertices,
                        None,
                        None,
                        Some(IndexBuffer::Unified(indices)),
                    )
                }
            }

            // third match to add the end cap
            // TODO: this will fail with patterns having multiple starting and end points!
            match next {
                PathSample::StartPoint(pt, dir) => {
                    self.start_cap.gen_start_cap(
                        0,
                        &self.pattern,
                        pt,
                        dir,
                        self.closed,
                        &mut vertices,
                        &mut indices,
                    );
                }
                PathSample::EndPoint(pt, dir) => {
                    self.end_cap.gen_end_cap(
                        vertices.len() as u32 - npts,
                        &self.pattern,
                        pt,
                        dir,
                        self.closed,
                        &mut vertices,
                        &mut indices,
                    );
                }
                _ => {}
            }
        }
    }
}
