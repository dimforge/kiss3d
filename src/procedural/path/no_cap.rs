use crate::procedural::path::PolylineCompatibleCap;
use glamx::Vec3;

/// A cap that renders nothing.
pub struct NoCap;

impl Default for NoCap {
    fn default() -> Self {
        Self::new()
    }
}

impl NoCap {
    /// Creates a new `NoCap`.
    #[inline]
    pub fn new() -> NoCap {
        NoCap
    }
}

impl PolylineCompatibleCap for NoCap {
    fn gen_start_cap(
        &self,
        _: u32,
        _: &[Vec3],
        _: Vec3,
        _: Vec3,
        _: bool,
        _: &mut Vec<Vec3>,
        _: &mut Vec<[u32; 3]>,
    ) {
    }

    fn gen_end_cap(
        &self,
        _: u32,
        _: &[Vec3],
        _: Vec3,
        _: Vec3,
        _: bool,
        _: &mut Vec<Vec3>,
        _: &mut Vec<[u32; 3]>,
    ) {
    }
}
