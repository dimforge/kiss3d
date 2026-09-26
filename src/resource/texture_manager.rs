//! A resource manager to load textures.

use image::{self, DynamicImage, GenericImageView};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::context::Context;

/// Wrapping parameters for a texture.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextureWrapping {
    /// Repeats the texture when a texture coordinate is out of bounds.
    Repeat,
    /// Repeats the mirrored texture when a texture coordinate is out of bounds.
    MirroredRepeat,
    /// Repeats the nearest edge point texture color when a texture coordinate is out of bounds.
    ClampToEdge,
}

impl From<TextureWrapping> for wgpu::AddressMode {
    #[inline]
    fn from(val: TextureWrapping) -> Self {
        match val {
            TextureWrapping::Repeat => wgpu::AddressMode::Repeat,
            TextureWrapping::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
            TextureWrapping::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        }
    }
}

/// The whole of how one texture is sampled, and how its pixels are prepared
/// before they reach the GPU.
///
/// One struct rather than a call per combination: a caller that reads its
/// settings out of a file cannot pick from `add_image`, `add_image_pixelated`
/// and `add_image_with_color_space` without giving some of them up.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextureSampling {
    /// Wrapping across the horizontal axis.
    pub wrap_u: TextureWrapping,
    /// Wrapping across the vertical axis.
    pub wrap_v: TextureWrapping,
    /// Between texels when the texture is magnified.
    pub mag_filter: wgpu::FilterMode,
    /// Between texels when it is minified.
    pub min_filter: wgpu::FilterMode,
    /// Between mip levels, read only when `mipmaps` is set.
    pub mipmap_filter: wgpu::MipmapFilterMode,
    /// Build the mip chain on upload, box-filtered in linear light.
    pub mipmaps: bool,
    /// Samples per fetch, 1 to 16. wgpu refuses anything above 1 unless all
    /// three filters are linear, so [`TextureSampling::sane`] lowers it.
    pub anisotropy: u16,
    /// Colour authored in sRGB, which the sampler decodes; data — a normal
    /// map, a mask — sets this false and is returned raw.
    pub srgb: bool,
    /// Multiply RGB by alpha before upload, which stops a filtered edge
    /// bleeding the transparent texels' colour into what is drawn.
    ///
    /// A premultiplied texture must be drawn with
    /// [`Blend2d::PremultipliedAlpha`](crate::scene::Blend2d::PremultipliedAlpha);
    /// under ordinary alpha blending it is multiplied a second time and comes
    /// out dark.
    pub premultiply: bool,
}

impl Default for TextureSampling {
    /// What [`TextureManager::add_image`] has always done: clamped, bilinear,
    /// no mip chain, sRGB colour.
    fn default() -> Self {
        TextureSampling {
            wrap_u: TextureWrapping::ClampToEdge,
            wrap_v: TextureWrapping::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            mipmaps: false,
            anisotropy: 1,
            srgb: true,
            premultiply: false,
        }
    }
}

impl TextureSampling {
    /// This sampling with whatever the device would refuse taken back out.
    ///
    /// wgpu validates anisotropy against the three filters and aborts the
    /// process on a mismatch; a texture asked for more than the driver's
    /// sixteen is a settings file, not a bug worth a crash.
    pub fn sane(mut self) -> Self {
        let linear = self.mag_filter == wgpu::FilterMode::Linear
            && self.min_filter == wgpu::FilterMode::Linear
            && self.mipmap_filter == wgpu::MipmapFilterMode::Linear;
        self.anisotropy = if linear {
            self.anisotropy.clamp(1, 16)
        } else {
            1
        };
        self
    }

    /// The pixel format this sampling reads its texels through.
    fn format(self) -> wgpu::TextureFormat {
        if self.srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        }
    }
}

/// sRGB transfer function (IEC 61966-2-1), decoding one stored byte.
///
/// Mip generation and premultiplication both average or scale colour, and
/// both have to do it in linear light or the result comes out dark.
fn srgb_to_linear(u: u8) -> f32 {
    let c = u as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The same transfer function the other way, back to a stored byte.
fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

/// A GPU texture with its view and sampler.
pub struct Texture {
    /// The underlying wgpu texture.
    pub texture: wgpu::Texture,
    /// The texture view for binding.
    pub view: wgpu::TextureView,
    /// The sampler for the texture.
    pub sampler: wgpu::Sampler,
    /// Texture dimensions (width, height).
    pub size: (u32, u32),
    /// Whether the uploaded RGB was multiplied by alpha, so a caller can pick
    /// the blend mode that matches rather than guessing.
    pub premultiplied: bool,
}

impl Texture {
    /// Creates a new texture with the given data, sampled the one way this
    /// call has always offered: the same wrapping and filter on every axis.
    ///
    /// [`Texture::sampled`] takes the rest of the sampler.
    pub fn new(
        width: u32,
        height: u32,
        data: &[u8],
        format: wgpu::TextureFormat,
        address_mode: wgpu::AddressMode,
        filter: wgpu::FilterMode,
        generate_mipmaps: bool,
    ) -> Arc<Texture> {
        let wrapping = match address_mode {
            wgpu::AddressMode::MirrorRepeat => TextureWrapping::MirroredRepeat,
            wgpu::AddressMode::Repeat => TextureWrapping::Repeat,
            _ => TextureWrapping::ClampToEdge,
        };
        Self::build(
            width,
            height,
            data,
            format,
            TextureSampling {
                wrap_u: wrapping,
                wrap_v: wrapping,
                mag_filter: filter,
                min_filter: filter,
                mipmap_filter: if generate_mipmaps {
                    wgpu::MipmapFilterMode::Linear
                } else {
                    wgpu::MipmapFilterMode::Nearest
                },
                mipmaps: generate_mipmaps,
                srgb: format == wgpu::TextureFormat::Rgba8UnormSrgb,
                ..Default::default()
            },
        )
    }

    /// Creates a new RGBA8 texture sampled exactly as `sampling` asks.
    ///
    /// `data` is straight (non-premultiplied) RGBA8; `sampling.premultiply`
    /// is applied here, in linear light for an sRGB texture, so the sampler
    /// returns colour already scaled by its alpha.
    pub fn sampled(
        width: u32,
        height: u32,
        data: &[u8],
        sampling: TextureSampling,
    ) -> Arc<Texture> {
        let sampling = sampling.sane();
        let format = sampling.format();
        if sampling.premultiply {
            let premultiplied = Self::premultiply_rgba(data, sampling.srgb);
            return Self::build(width, height, &premultiplied, format, sampling);
        }
        Self::build(width, height, data, format, sampling)
    }

    fn build(
        width: u32,
        height: u32,
        data: &[u8],
        format: wgpu::TextureFormat,
        sampling: TextureSampling,
    ) -> Arc<Texture> {
        let generate_mipmaps = sampling.mipmaps;
        let ctxt = Context::get();

        let mip_level_count = if generate_mipmaps {
            (width.max(height) as f32).log2().floor() as u32 + 1
        } else {
            1
        };

        let texture = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let bytes_per_pixel = match format {
            wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Rgba8Unorm => 4,
            _ => 4, // Default to 4
        };

        // Upload mip level 0
        ctxt.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * bytes_per_pixel),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Generate and upload remaining mip levels
        if generate_mipmaps && mip_level_count > 1 {
            // Color (sRGB) textures must be averaged in linear space; data
            // textures (normal/metallic-roughness/AO) are already linear.
            let srgb = matches!(format, wgpu::TextureFormat::Rgba8UnormSrgb);
            let mut current_data = data.to_vec();
            let mut current_width = width;
            let mut current_height = height;

            for mip_level in 1..mip_level_count {
                let new_width = (current_width / 2).max(1);
                let new_height = (current_height / 2).max(1);

                let new_data =
                    Self::downsample_rgba(&current_data, current_width, current_height, srgb);

                ctxt.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &new_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(new_width * bytes_per_pixel),
                        rows_per_image: Some(new_height),
                    },
                    wgpu::Extent3d {
                        width: new_width,
                        height: new_height,
                        depth_or_array_layers: 1,
                    },
                );

                current_data = new_data;
                current_width = new_width;
                current_height = new_height;
            }
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = ctxt.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("texture_sampler"),
            address_mode_u: sampling.wrap_u.into(),
            address_mode_v: sampling.wrap_v.into(),
            // Nothing here is a 3D texture, so w follows u rather than
            // offering a key that samples nothing.
            address_mode_w: sampling.wrap_u.into(),
            mag_filter: sampling.mag_filter,
            min_filter: sampling.min_filter,
            mipmap_filter: if generate_mipmaps {
                sampling.mipmap_filter
            } else {
                wgpu::MipmapFilterMode::Nearest
            },
            anisotropy_clamp: sampling.anisotropy,
            ..Default::default()
        });

        Arc::new(Texture {
            texture,
            view,
            sampler,
            size: (width, height),
            premultiplied: sampling.premultiply,
        })
    }

    /// RGB scaled by alpha, which is what a premultiplied blend expects.
    ///
    /// An sRGB texture is scaled in linear light: the sampler decodes the
    /// stored value before it is blended, so scaling the encoded byte would
    /// leave the fringe it is here to remove.
    fn premultiply_rgba(data: &[u8], srgb: bool) -> Vec<u8> {
        let mut out = data.to_vec();
        for texel in out.as_chunks_mut::<4>().0 {
            let alpha = texel[3] as f32 / 255.0;
            for channel in &mut texel[..3] {
                *channel = if srgb {
                    linear_to_srgb(srgb_to_linear(*channel) * alpha)
                } else {
                    ((*channel as f32) * alpha + 0.5) as u8
                };
            }
        }
        out
    }

    /// Downsamples an RGBA image by half using box filtering.
    ///
    /// When `srgb` is set, RGB channels are decoded to linear light before
    /// averaging and re-encoded afterward (alpha is always linear), so color
    /// mip chains don't darken — the gamma-correct behavior. Data textures pass
    /// `srgb = false` and are averaged directly.
    fn downsample_rgba(data: &[u8], width: u32, height: u32, srgb: bool) -> Vec<u8> {
        let new_width = (width / 2).max(1);
        let new_height = (height / 2).max(1);
        let mut new_data = vec![0u8; (new_width * new_height * 4) as usize];

        for y in 0..new_height {
            for x in 0..new_width {
                // Sample 2x2 block from source (or fewer pixels at edges)
                let src_x = (x * 2) as usize;
                let src_y = (y * 2) as usize;

                let mut r = 0f32;
                let mut g = 0f32;
                let mut b = 0f32;
                let mut a = 0f32;
                let mut count = 0f32;

                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = src_x + dx;
                        let sy = src_y + dy;
                        if sx < width as usize && sy < height as usize {
                            let idx = (sy * width as usize + sx) * 4;
                            if srgb {
                                r += srgb_to_linear(data[idx]);
                                g += srgb_to_linear(data[idx + 1]);
                                b += srgb_to_linear(data[idx + 2]);
                            } else {
                                r += data[idx] as f32 / 255.0;
                                g += data[idx + 1] as f32 / 255.0;
                                b += data[idx + 2] as f32 / 255.0;
                            }
                            a += data[idx + 3] as f32 / 255.0;
                            count += 1.0;
                        }
                    }
                }

                let inv = 1.0 / count;
                let dst_idx = ((y * new_width + x) * 4) as usize;
                if srgb {
                    new_data[dst_idx] = linear_to_srgb(r * inv);
                    new_data[dst_idx + 1] = linear_to_srgb(g * inv);
                    new_data[dst_idx + 2] = linear_to_srgb(b * inv);
                } else {
                    new_data[dst_idx] = ((r * inv) * 255.0 + 0.5) as u8;
                    new_data[dst_idx + 1] = ((g * inv) * 255.0 + 0.5) as u8;
                    new_data[dst_idx + 2] = ((b * inv) * 255.0 + 0.5) as u8;
                }
                new_data[dst_idx + 3] = ((a * inv) * 255.0 + 0.5) as u8;
            }
        }

        new_data
    }

    /// Creates a default white 1x1 texture.
    pub fn new_default() -> Arc<Texture> {
        let white_pixel: [u8; 4] = [255, 255, 255, 255];
        Self::new(
            1,
            1,
            &white_pixel,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::AddressMode::Repeat,
            wgpu::FilterMode::Linear,
            false,
        )
    }

    /// Creates a default flat normal map (1x1, pointing straight up in tangent space).
    ///
    /// The RGB value (128, 128, 255) represents a normal of (0, 0, 1) in tangent space.
    pub fn new_default_normal_map() -> Arc<Texture> {
        let normal_pixel: [u8; 4] = [128, 128, 255, 255];
        Self::new(
            1,
            1,
            &normal_pixel,
            wgpu::TextureFormat::Rgba8Unorm, // Normal maps use linear data, not sRGB
            wgpu::AddressMode::Repeat,
            wgpu::FilterMode::Linear,
            false,
        )
    }

    /// Creates a default metallic-roughness map (1x1, non-metallic with medium roughness).
    ///
    /// Follows glTF convention: B channel = metallic (0), G channel = roughness (0.5).
    pub fn new_default_metallic_roughness_map() -> Arc<Texture> {
        let mr_pixel: [u8; 4] = [0, 128, 0, 255]; // R=unused, G=roughness(0.5), B=metallic(0)
        Self::new(
            1,
            1,
            &mr_pixel,
            wgpu::TextureFormat::Rgba8Unorm, // Data texture, not sRGB
            wgpu::AddressMode::Repeat,
            wgpu::FilterMode::Linear,
            false,
        )
    }

    /// Creates a default ambient occlusion map (1x1, white = no occlusion).
    pub fn new_default_ao_map() -> Arc<Texture> {
        let ao_pixel: [u8; 4] = [255, 255, 255, 255];
        Self::new(
            1,
            1,
            &ao_pixel,
            wgpu::TextureFormat::Rgba8Unorm, // Data texture, not sRGB
            wgpu::AddressMode::Repeat,
            wgpu::FilterMode::Linear,
            false,
        )
    }

    /// Creates a default height map (1x1, mid-gray). Only used as a placeholder
    /// when no height map is set; parallax is gated by a flag, so its value is
    /// irrelevant.
    pub fn new_default_height_map() -> Arc<Texture> {
        let pixel: [u8; 4] = [128, 128, 128, 255];
        Self::new(
            1,
            1,
            &pixel,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::AddressMode::Repeat,
            wgpu::FilterMode::Linear,
            false,
        )
    }

    /// Creates a default emissive map (1x1, black = no emission).
    pub fn new_default_emissive_map() -> Arc<Texture> {
        let emissive_pixel: [u8; 4] = [0, 0, 0, 255];
        Self::new(
            1,
            1,
            &emissive_pixel,
            wgpu::TextureFormat::Rgba8UnormSrgb, // Emissive is color data, use sRGB
            wgpu::AddressMode::Repeat,
            wgpu::FilterMode::Linear,
            false,
        )
    }
}

/// The texture manager.
///
/// It keeps a cache of already-loaded textures, and can load new textures.
pub struct TextureManager {
    default_texture: Arc<Texture>,
    textures: HashMap<String, Arc<Texture>>,
    generate_mipmaps: bool,
}

impl Default for TextureManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TextureManager {
    /// Creates a new texture manager.
    pub fn new() -> TextureManager {
        let default_texture = Texture::new_default();

        TextureManager {
            textures: HashMap::new(),
            default_texture,
            generate_mipmaps: false,
        }
    }

    /// Mutably applies a function to the texture manager.
    pub fn get_global_manager<T, F: FnMut(&mut TextureManager) -> T>(mut f: F) -> T {
        crate::window::WINDOW_CACHE
            .with(|manager| f(&mut *manager.borrow_mut().texture_manager.as_mut().unwrap()))
    }

    /// Gets the default, completely white, texture.
    pub fn get_default(&self) -> Arc<Texture> {
        self.default_texture.clone()
    }

    /// Get a texture with the specified name. Returns `None` if the texture is not registered.
    pub fn get(&mut self, name: &str) -> Option<Arc<Texture>> {
        self.textures.get(name).cloned()
    }

    /// Get a texture (and its size) with the specified name. Returns `None` if the texture is not registered.
    pub fn get_with_size(&mut self, name: &str) -> Option<(Arc<Texture>, (u32, u32))> {
        self.textures.get(name).map(|t| (t.clone(), t.size))
    }

    /// Allocates a new texture that is not yet configured.
    ///
    /// If a texture with same name exists, nothing is created and the old texture is returned.
    pub fn add_empty(&mut self, name: &str) -> Arc<Texture> {
        match self.textures.entry(name.to_string()) {
            Entry::Occupied(entry) => entry.into_mut().clone(),
            Entry::Vacant(entry) => entry.insert(Texture::new_default()).clone(),
        }
    }

    /// Allocates a new texture read from a `DynamicImage` object, sampled with
    /// smooth (bilinear) filtering.
    ///
    /// If a texture with same name exists, nothing is created and the old texture is returned.
    pub fn add_image(&mut self, image: DynamicImage, name: &str) -> Arc<Texture> {
        self.add_image_filtered(image, name, wgpu::FilterMode::Linear)
    }

    /// Like [`add_image`](Self::add_image) but with nearest-neighbor filtering, for
    /// pixel-art / sprite-sheet textures: crisp when magnified, with no bilinear
    /// bleed across atlas-cell boundaries.
    pub fn add_image_pixelated(&mut self, image: DynamicImage, name: &str) -> Arc<Texture> {
        self.add_image_filtered(image, name, wgpu::FilterMode::Nearest)
    }

    fn add_image_filtered(
        &mut self,
        image: DynamicImage,
        name: &str,
        filter: wgpu::FilterMode,
    ) -> Arc<Texture> {
        let generate_mipmaps = self.generate_mipmaps;
        self.textures
            .entry(name.to_string())
            .or_insert_with(|| {
                TextureManager::load_texture_from_image(image, generate_mipmaps, filter)
            })
            .clone()
    }

    /// Registers a texture sampled exactly as `sampling` asks: wrapping per
    /// axis, a filter per magnification, minification and mip level,
    /// anisotropy, the colour space and premultiplied alpha.
    ///
    /// The named calls above are the handful of combinations that came before
    /// it; this is the whole sampler, for a caller reading its settings out of
    /// a file. If a texture with the same name exists it is returned as it is,
    /// so a caller changing a setting has to change the name too.
    pub fn add_image_sampled(
        &mut self,
        image: DynamicImage,
        name: &str,
        sampling: TextureSampling,
    ) -> Arc<Texture> {
        self.textures
            .entry(name.to_string())
            .or_insert_with(|| {
                let (width, height) = image.dimensions();
                let rgba = image.to_rgba8();
                Texture::sampled(width, height, rgba.as_raw(), sampling)
            })
            .clone()
    }

    /// Allocates a new texture and tries to decode it from bytes array
    /// Panics if unable to do so
    /// If a texture with same name exists, nothing is created and the old texture is returned.
    pub fn add_image_from_memory(&mut self, image_data: &[u8], name: &str) -> Arc<Texture> {
        self.add_image(
            image::load_from_memory(image_data).expect("Invalid data"),
            name,
        )
    }

    /// Like [`add_image_from_memory`](Self::add_image_from_memory) but with
    /// nearest-neighbor filtering, for pixel-art / sprite-sheet textures (see
    /// [`add_image_pixelated`](Self::add_image_pixelated)).
    pub fn add_image_from_memory_pixelated(
        &mut self,
        image_data: &[u8],
        name: &str,
    ) -> Arc<Texture> {
        self.add_image_pixelated(
            image::load_from_memory(image_data).expect("Invalid data"),
            name,
        )
    }

    /// Registers a texture from a [`DynamicImage`], choosing the color space and
    /// using glTF-style `Repeat` wrapping.
    ///
    /// Color/emissive textures are authored in sRGB (`srgb = true`); data textures
    /// — normal, metallic-roughness, occlusion — are linear (`srgb = false`). This
    /// is what the glTF loader needs, since [`add_image`](Self::add_image) always
    /// assumes sRGB. If a texture with the same name exists it is returned as-is.
    pub fn add_image_with_color_space(
        &mut self,
        image: DynamicImage,
        name: &str,
        srgb: bool,
    ) -> Arc<Texture> {
        let generate_mipmaps = self.generate_mipmaps;
        self.textures
            .entry(name.to_string())
            .or_insert_with(|| {
                let (width, height) = image.dimensions();
                let rgba_image = image.to_rgba8();
                let format = if srgb {
                    wgpu::TextureFormat::Rgba8UnormSrgb
                } else {
                    wgpu::TextureFormat::Rgba8Unorm
                };
                Texture::new(
                    width,
                    height,
                    rgba_image.as_raw(),
                    format,
                    wgpu::AddressMode::Repeat,
                    wgpu::FilterMode::Linear,
                    generate_mipmaps,
                )
            })
            .clone()
    }

    /// Loads a texture from a DynamicImage.
    fn load_texture_from_image(
        image: DynamicImage,
        generate_mipmaps: bool,
        filter: wgpu::FilterMode,
    ) -> Arc<Texture> {
        let (width, height) = image.dimensions();

        // Convert to RGBA8
        let rgba_image = image.to_rgba8();
        let pixels = rgba_image.as_raw();

        Texture::new(
            width,
            height,
            pixels,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::AddressMode::ClampToEdge,
            filter,
            generate_mipmaps,
        )
    }

    /// Allocates a new texture read from a file.
    fn load_texture_from_file(
        path: &Path,
        generate_mipmaps: bool,
        filter: wgpu::FilterMode,
    ) -> Arc<Texture> {
        let image = image::open(path)
            .unwrap_or_else(|e| panic!("Unable to load texture from file {:?}: {:?}", path, e));
        TextureManager::load_texture_from_image(image, generate_mipmaps, filter)
    }

    /// Allocates a new texture read from a file. If a texture with same name exists, nothing is
    /// created and the old texture is returned.
    pub fn add(&mut self, path: &Path, name: &str) -> Arc<Texture> {
        self.add_filtered(path, name, wgpu::FilterMode::Linear)
    }

    /// Like [`add`](Self::add) but samples with nearest-neighbor filtering, for
    /// pixel-art / sprite-sheet textures (see
    /// [`add_image_pixelated`](Self::add_image_pixelated)).
    pub fn add_pixelated(&mut self, path: &Path, name: &str) -> Arc<Texture> {
        self.add_filtered(path, name, wgpu::FilterMode::Nearest)
    }

    fn add_filtered(&mut self, path: &Path, name: &str, filter: wgpu::FilterMode) -> Arc<Texture> {
        let generate_mipmaps = self.generate_mipmaps;
        self.textures
            .entry(name.to_string())
            .or_insert_with(|| {
                TextureManager::load_texture_from_file(path, generate_mipmaps, filter)
            })
            .clone()
    }

    /// Changes whether textures will have mipmaps generated when they are
    /// loaded; does not affect already loaded textures.
    /// Mipmap generation is disabled by default.
    pub fn set_generate_mipmaps(&mut self, enabled: bool) {
        self.generate_mipmaps = enabled;
    }
}

#[cfg(test)]
mod tests {
    use super::{Texture, TextureSampling, TextureWrapping};

    /// wgpu aborts the process on a sampler it considers invalid, so a
    /// settings file asking for anisotropy on a pixel-art texture has to lose
    /// the anisotropy rather than the run.
    #[test]
    fn anisotropy_needs_every_filter_linear() {
        let pixelated = TextureSampling {
            mag_filter: wgpu::FilterMode::Nearest,
            anisotropy: 16,
            ..Default::default()
        };
        assert_eq!(pixelated.sane().anisotropy, 1);

        let smooth = TextureSampling {
            anisotropy: 16,
            ..Default::default()
        };
        assert_eq!(smooth.sane().anisotropy, 16);
        assert_eq!(
            TextureSampling {
                anisotropy: 64,
                ..Default::default()
            }
            .sane()
            .anisotropy,
            16,
            "more than the device offers is clamped, not refused"
        );
    }

    #[test]
    fn the_default_sampling_is_what_add_image_always_did() {
        let sampling = TextureSampling::default();
        assert_eq!(sampling.wrap_u, TextureWrapping::ClampToEdge);
        assert_eq!(sampling.mag_filter, wgpu::FilterMode::Linear);
        assert!(sampling.srgb && !sampling.mipmaps && !sampling.premultiply);
    }

    /// A transparent texel keeps no colour to bleed into its neighbours, and
    /// an opaque one is left exactly as it was authored.
    #[test]
    fn premultiplying_scales_colour_by_alpha() {
        let straight = [255u8, 128, 0, 0, 255, 128, 0, 255];
        let linear = Texture::premultiply_rgba(&straight, false);
        assert_eq!(&linear[..4], &[0, 0, 0, 0]);
        assert_eq!(&linear[4..], &[255, 128, 0, 255]);
    }

    /// Halved alpha in linear light is not a halved byte: scaling the stored
    /// value instead would leave the fringe premultiplication removes.
    #[test]
    fn an_srgb_texture_is_scaled_in_linear_light() {
        let straight = [255u8, 255, 255, 128];
        let encoded = Texture::premultiply_rgba(&straight, true);
        assert_eq!(&encoded[..3], &[188, 188, 188]);
        assert_eq!(encoded[3], 128, "alpha is not itself scaled");
    }
}
