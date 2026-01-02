use glamx::Vec3;

use crate::procedural::RenderMesh;

/// A sample point and its associated tangent.
pub enum PathSample {
    /// A point that starts a new path.
    StartPoint(Vec3, Vec3),
    /// A point that is inside of the path currently generated.
    InnerPoint(Vec3, Vec3),
    /// A point that ends the path currently generated.
    EndPoint(Vec3, Vec3),
    /// Used when the sampler does not have any other points to generate.
    EndOfSample,
}

/// A curve sampler.
pub trait CurveSampler {
    /// Returns the next sample point.
    fn next(&mut self) -> PathSample;
}

/// A pattern that is replicated along a path.
///
/// It is responsible of the generation of the whole mesh.
pub trait StrokePattern {
    /// Generates the mesh using this pattern and the curve sampled by `sampler`.
    fn stroke<C: CurveSampler>(&mut self, sampler: &mut C) -> RenderMesh;
}
