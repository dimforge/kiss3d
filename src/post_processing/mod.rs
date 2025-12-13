//! Post-processing effects.

pub use crate::post_processing::grayscales::Grayscales;
pub use crate::post_processing::oculus_stereo::OculusStereo;
pub use crate::post_processing::post_processing_effect::{
    PostProcessingContext, PostProcessingEffect,
};
#[cfg(not(target_arch = "wasm32"))]
pub use crate::post_processing::sobel_edge_highlight::SobelEdgeHighlight;
pub use crate::post_processing::waves::Waves;

mod grayscales;
mod oculus_stereo;
pub mod post_processing_effect;
#[cfg(not(target_arch = "wasm32"))]
mod sobel_edge_highlight;
mod waves;
