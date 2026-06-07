//! Lazy, sample-count-keyed render-pipeline cache.
//!
//! Rasterization pipelines are identical except for their
//! [`MultisampleState.count`](wgpu::MultisampleState::count), but that count is
//! only known at render time — it comes from the window's
//! [`CanvasSetup`](crate::window::CanvasSetup) and may differ between windows
//! sharing the same global material. Materials and renderers therefore can't bake
//! it in at construction.
//!
//! [`PipelineCache`] solves this by storing a *builder* closure that produces a
//! pipeline for a given sample count, and building each variant lazily on first
//! use. The result is cached, so toggling MSAA only pays the pipeline-creation
//! cost once per distinct sample count.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// A render pipeline that is (re)built on demand for a specific MSAA sample count
/// and cached thereafter.
///
/// Build a cache from a closure that takes a sample count and returns the matching
/// pipeline, then call [`get`](Self::get) with `context.sample_count` at draw time.
pub struct PipelineCache {
    builder: Box<dyn Fn(u32) -> wgpu::RenderPipeline>,
    cache: RefCell<HashMap<u32, Rc<wgpu::RenderPipeline>>>,
}

impl PipelineCache {
    /// Creates a cache whose pipelines are produced by `builder`. The builder
    /// receives the (clamped, `>= 1`) sample count and must return a pipeline
    /// whose `MultisampleState.count` equals it.
    pub fn new(builder: impl Fn(u32) -> wgpu::RenderPipeline + 'static) -> Self {
        PipelineCache {
            builder: Box::new(builder),
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// Returns the pipeline for `sample_count`, building and caching it on first
    /// use. A `sample_count` of 0 is treated as 1 (no multisampling).
    pub fn get(&self, sample_count: u32) -> Rc<wgpu::RenderPipeline> {
        let sample_count = sample_count.max(1);
        if let Some(pipeline) = self.cache.borrow().get(&sample_count) {
            return pipeline.clone();
        }
        let pipeline = Rc::new((self.builder)(sample_count));
        self.cache
            .borrow_mut()
            .insert(sample_count, pipeline.clone());
        pipeline
    }
}

/// A standard single-color-target [`MultisampleState`](wgpu::MultisampleState) for
/// the given sample count, with the default sample mask and no alpha-to-coverage.
pub fn multisample_state(sample_count: u32) -> wgpu::MultisampleState {
    wgpu::MultisampleState {
        count: sample_count.max(1),
        mask: !0,
        alpha_to_coverage_enabled: false,
    }
}
