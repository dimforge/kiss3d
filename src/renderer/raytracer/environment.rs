//! HDRI environment map for image-based lighting in the path tracer.
//!
//! Loads an equirectangular HDR image (Radiance `.hdr` or any format the `image`
//! crate decodes) into a floating-point texture that the kernel looks up for
//! escaped rays and the background. A 1×1 black fallback is always available so
//! the binding stays valid when no environment is set, in which case the kernel
//! falls back to its procedural gradient sky.

use std::path::Path;

use crate::context::Context;

/// A GPU-resident equirectangular environment map.
pub struct Environment {
    /// The equirect texture view bound at group 1, binding 8.
    pub view: wgpu::TextureView,
    /// The sampler bound at group 1, binding 9.
    pub sampler: wgpu::Sampler,
    /// Whether this holds a real environment (vs. the black fallback).
    pub present: bool,
}

impl Environment {
    /// A 1×1 black fallback used when no environment map is set.
    pub fn fallback() -> Environment {
        Self::from_rgba_f32(1, 1, &[0.0, 0.0, 0.0, 1.0], false)
    }

    /// Loads an equirectangular HDR/LDR image from a file.
    ///
    /// Returns `None` if the file cannot be decoded.
    pub fn from_file(path: &Path) -> Option<Environment> {
        let img = image::open(path).ok()?;
        Some(Self::from_image(&img))
    }

    /// Builds an environment from an already-decoded image.
    pub fn from_image(img: &image::DynamicImage) -> Environment {
        use image::GenericImageView;
        let (w, h) = img.dimensions();
        let rgb = img.to_rgba32f();
        Self::from_rgba_f32(w, h, rgb.as_raw(), true)
    }

    /// Uploads RGBA-f32 pixel data as an `Rgba16Float` texture (broadly supported,
    /// half the bandwidth of f32). `present` flags whether it is a real map.
    pub fn from_rgba_f32(width: u32, height: u32, rgba: &[f32], present: bool) -> Environment {
        let ctxt = Context::get();

        // Convert f32 -> f16 bits for upload.
        let halfs: Vec<u16> = rgba.iter().map(|&v| f32_to_f16(v)).collect();

        let texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("rt_environment"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        ctxt.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&halfs),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 8), // 4 channels * 2 bytes
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rt_environment_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Environment {
            view,
            sampler,
            present,
        }
    }
}

/// Converts an `f32` to IEEE-754 half-precision bits (round-to-nearest-even is
/// approximated by truncation, which is plenty for an environment lookup).
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    if exp <= 0 {
        sign // underflow to (signed) zero
    } else if exp >= 0x1f {
        sign | 0x7c00 // overflow to infinity
    } else {
        sign | ((exp as u16) << 10) | (mant as u16)
    }
}
