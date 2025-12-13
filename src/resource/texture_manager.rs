//! A resource manager to load textures.

use image::{self, DynamicImage, GenericImageView};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::context::Context;

/// Wrapping parameters for a texture.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
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
}

impl Texture {
    /// Creates a new texture with the given data.
    pub fn new(
        width: u32,
        height: u32,
        data: &[u8],
        format: wgpu::TextureFormat,
        address_mode: wgpu::AddressMode,
        generate_mipmaps: bool,
    ) -> Arc<Texture> {
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
            let mut current_data = data.to_vec();
            let mut current_width = width;
            let mut current_height = height;

            for mip_level in 1..mip_level_count {
                let new_width = (current_width / 2).max(1);
                let new_height = (current_height / 2).max(1);

                let new_data = Self::downsample_rgba(&current_data, current_width, current_height);

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
            address_mode_u: address_mode,
            address_mode_v: address_mode,
            address_mode_w: address_mode,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: if generate_mipmaps {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            },
            ..Default::default()
        });

        Arc::new(Texture {
            texture,
            view,
            sampler,
            size: (width, height),
        })
    }

    /// Downsamples an RGBA image by half using box filtering.
    fn downsample_rgba(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let new_width = (width / 2).max(1);
        let new_height = (height / 2).max(1);
        let mut new_data = vec![0u8; (new_width * new_height * 4) as usize];

        for y in 0..new_height {
            for x in 0..new_width {
                // Sample 2x2 block from source (or fewer pixels at edges)
                let src_x = (x * 2) as usize;
                let src_y = (y * 2) as usize;

                let mut r = 0u32;
                let mut g = 0u32;
                let mut b = 0u32;
                let mut a = 0u32;
                let mut count = 0u32;

                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = src_x + dx;
                        let sy = src_y + dy;
                        if sx < width as usize && sy < height as usize {
                            let idx = (sy * width as usize + sx) * 4;
                            r += data[idx] as u32;
                            g += data[idx + 1] as u32;
                            b += data[idx + 2] as u32;
                            a += data[idx + 3] as u32;
                            count += 1;
                        }
                    }
                }

                let dst_idx = ((y * new_width + x) * 4) as usize;
                new_data[dst_idx] = (r / count) as u8;
                new_data[dst_idx + 1] = (g / count) as u8;
                new_data[dst_idx + 2] = (b / count) as u8;
                new_data[dst_idx + 3] = (a / count) as u8;
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

    /// Allocates a new texture read from a `DynamicImage` object.
    ///
    /// If a texture with same name exists, nothing is created and the old texture is returned.
    pub fn add_image(&mut self, image: DynamicImage, name: &str) -> Arc<Texture> {
        let generate_mipmaps = self.generate_mipmaps;
        self.textures
            .entry(name.to_string())
            .or_insert_with(|| TextureManager::load_texture_from_image(image, generate_mipmaps))
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

    /// Loads a texture from a DynamicImage.
    fn load_texture_from_image(image: DynamicImage, generate_mipmaps: bool) -> Arc<Texture> {
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
            generate_mipmaps,
        )
    }

    /// Allocates a new texture read from a file.
    fn load_texture_from_file(path: &Path, generate_mipmaps: bool) -> Arc<Texture> {
        let image = image::open(path)
            .unwrap_or_else(|e| panic!("Unable to load texture from file {:?}: {:?}", path, e));
        TextureManager::load_texture_from_image(image, generate_mipmaps)
    }

    /// Allocates a new texture read from a file. If a texture with same name exists, nothing is
    /// created and the old texture is returned.
    pub fn add(&mut self, path: &Path, name: &str) -> Arc<Texture> {
        let generate_mipmaps = self.generate_mipmaps;
        self.textures
            .entry(name.to_string())
            .or_insert_with(|| TextureManager::load_texture_from_file(path, generate_mipmaps))
            .clone()
    }

    /// Changes whether textures will have mipmaps generated when they are
    /// loaded; does not affect already loaded textures.
    /// Mipmap generation is disabled by default.
    pub fn set_generate_mipmaps(&mut self, enabled: bool) {
        self.generate_mipmaps = enabled;
    }
}
