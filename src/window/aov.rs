//! Auxiliary render outputs (AOVs): depth, surface normals and segmentation.
//!
//! These methods render the current 3D scene a second time, with a dedicated
//! material that writes a geometric quantity instead of shaded color, then read
//! the result back to the CPU. They are meant primarily for the headless
//! [`OffscreenSurface`](crate::window::OffscreenSurface) used by
//! robotics/embodied-AI pipelines, but are available on any [`Window`].
//!
//! Each AOV is rendered into a single-sampled texture (so read-back is exact)
//! using the shared scene graph and camera; see [`AovRenderer`].

use crate::builtin::{AovKind, AovRenderer};
use crate::camera::Camera3d;
use crate::context::Context;
use crate::light::LightCollection;
use crate::scene::SceneNode3d;
use image::{ImageBuffer, Luma, Rgb};

use super::Window;

impl Window {
    /// Renders the scene and returns per-pixel **linear, eye-space depth** in
    /// world units.
    ///
    /// The returned buffer is row-major with a top-left origin (matching
    /// [`snap_image`](Self::snap_image)). Each value is the positive distance,
    /// along the camera's view direction, from the camera to the closest
    /// surface at that pixel. Background pixels (no geometry) read back as `0.0`.
    ///
    /// Use [`depth_to_luma8`](Self::depth_to_luma8) to turn this into a
    /// normalized 8-bit grayscale image for visualization or saving as PNG.
    pub fn snap_depth_raw(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut dyn Camera3d,
    ) -> Vec<f32> {
        self.render_aov::<f32>(AovKind::Depth, scene, camera, 1)
    }

    /// Renders the scene and returns its depth as a normalized 8-bit grayscale
    /// image.
    ///
    /// Depth is linearly remapped from `[min, max]` (the smallest and largest
    /// finite, non-background depths in the frame) to `[0, 255]`, with the
    /// nearest surface brightest. Background pixels are black. This is a
    /// convenience wrapper around [`snap_depth_raw`](Self::snap_depth_raw).
    pub fn snap_depth(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut dyn Camera3d,
    ) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let (w, h) = self.canvas.size();
        let depth = self.snap_depth_raw(scene, camera);
        Self::depth_to_luma8(&depth, w, h)
    }

    /// Renders the scene and returns its **world-space surface normals**.
    ///
    /// Each pixel stores the unit normal encoded from `[-1, 1]` into `[0, 255]`
    /// per channel (`r = (nx + 1) / 2 * 255`, etc.), the same convention used by
    /// the [`NormalsMaterial`](crate::builtin::NormalsMaterial) preview.
    /// Background pixels are black.
    pub fn snap_normals(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut dyn Camera3d,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        self.snap_normals_kind(AovKind::Normals, scene, camera)
    }

    /// Like [`snap_normals`](Self::snap_normals) but in **camera (eye) space**:
    /// normals are expressed relative to the camera orientation.
    pub fn snap_camera_normals(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut dyn Camera3d,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        self.snap_normals_kind(AovKind::CameraNormals, scene, camera)
    }

    fn snap_normals_kind(
        &mut self,
        kind: AovKind,
        scene: &mut SceneNode3d,
        camera: &mut dyn Camera3d,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (w, h) = self.canvas.size();
        let data = self.render_aov::<f32>(kind, scene, camera, 4);
        let mut img = ImageBuffer::new(w, h);
        for (i, px) in img.pixels_mut().enumerate() {
            let base = i * 4;
            let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            *px = Rgb([
                to_u8(data[base]),
                to_u8(data[base + 1]),
                to_u8(data[base + 2]),
            ]);
        }
        img
    }

    /// Renders the scene and returns the per-pixel **segmentation / object id**.
    ///
    /// Each value is the integer id of the object covering that pixel (see
    /// [`Object3d::set_segmentation_id`](crate::scene::Object3d::set_segmentation_id)).
    /// Background pixels read back as `0`. The buffer is row-major, top-left
    /// origin.
    pub fn snap_segmentation(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut dyn Camera3d,
    ) -> Vec<u32> {
        self.render_aov::<u32>(AovKind::Segmentation, scene, camera, 1)
    }

    /// Renders the scene and returns a **colorized segmentation** image.
    ///
    /// Each distinct object id is mapped to a deterministic, well-spread RGB
    /// color (id `0`, the background, is always black). Convenient for saving a
    /// human-readable PNG.
    pub fn snap_segmentation_colored(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut dyn Camera3d,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (w, h) = self.canvas.size();
        let ids = self.snap_segmentation(scene, camera);
        let mut img = ImageBuffer::new(w, h);
        for (i, px) in img.pixels_mut().enumerate() {
            *px = Rgb(colorize_id(ids[i]));
        }
        img
    }

    /// Linearly normalizes a raw linear-depth buffer into an 8-bit grayscale
    /// image (nearest surface brightest, background black).
    pub fn depth_to_luma8(
        depth: &[f32],
        width: u32,
        height: u32,
    ) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        // Find the finite, non-background depth range.
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &d in depth {
            if d.is_finite() && d > 0.0 {
                min = min.min(d);
                max = max.max(d);
            }
        }

        let range = max - min;
        let mut img = ImageBuffer::new(width, height);
        for (i, px) in img.pixels_mut().enumerate() {
            let d = depth[i];
            let v = if d.is_finite() && d > 0.0 && range > 0.0 {
                // Nearer = brighter.
                let t = (d - min) / range;
                ((1.0 - t) * 255.0).round() as u8
            } else if d.is_finite() && d > 0.0 {
                255
            } else {
                0
            };
            *px = Luma([v]);
        }
        img
    }

    /// Shared AOV render + read-back path.
    ///
    /// Runs the scene's `prepare` phase (to propagate world transforms), then
    /// dispatches the [`AovRenderer`] for `kind` into freshly created
    /// single-sampled color/depth textures, and copies the color texture back to
    /// the CPU as `channels` elements of type `T` per pixel.
    fn render_aov<T: bytemuck::Pod + Default>(
        &mut self,
        kind: AovKind,
        scene: &mut SceneNode3d,
        camera: &mut dyn Camera3d,
        channels: usize,
    ) -> Vec<T> {
        let w = self.width().max(1);
        let h = self.height().max(1);
        let ctxt = Context::get();

        // Make sure the camera matrices match the target size, then propagate
        // world transforms so the AOV renderer can read them per object.
        camera.update(&self.canvas);
        let mut lights = LightCollection::with_ambient(self.ambient_intensity);
        scene.data_mut().prepare(0, camera, &mut lights, w, h);

        // Create the single-sampled color and depth targets for this AOV.
        let color = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("aov_color_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: kind.format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());

        let depth = ctxt.create_texture(&wgpu::TextureDescriptor {
            label: Some("aov_depth_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Context::depth_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

        if self.aov_renderer.is_none() {
            self.aov_renderer = Some(AovRenderer::new());
        }

        let mut encoder = ctxt.create_command_encoder(Some("aov_encoder"));
        self.aov_renderer.as_mut().unwrap().render(
            kind,
            scene,
            camera,
            &mut encoder,
            &color_view,
            &depth_view,
        );
        ctxt.submit(std::iter::once(encoder.finish()));

        read_texture::<T>(&color, w, h, channels)
    }
}

/// Reads back a color texture into a CPU buffer of `T` elements.
///
/// Handles wgpu's 256-byte row alignment and removes the padding. The texture
/// must have `COPY_SRC` usage and store `channels` elements of type `T` per
/// pixel (matching the AOV format). The result is row-major with a top-left
/// origin.
fn read_texture<T: bytemuck::Pod + Default>(
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    channels: usize,
) -> Vec<T> {
    let ctxt = Context::get();

    let elem_size = std::mem::size_of::<T>();
    let bytes_per_pixel = elem_size * channels;
    let unpadded_bytes_per_row = width as usize * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let buffer_size = padded_bytes_per_row * height as usize;

    let staging = ctxt.create_buffer(&wgpu::BufferDescriptor {
        label: Some("aov_staging_buffer"),
        size: buffer_size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctxt.create_command_encoder(Some("aov_readback_encoder"));
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row as u32),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    ctxt.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    let _ = ctxt.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range();
    let mut data: Vec<T> = vec![T::default(); width as usize * height as usize * channels];
    // wgpu uses a top-left origin, which matches the row-major layout we return.
    for row in 0..height as usize {
        let src = row * padded_bytes_per_row;
        let dst = row * width as usize * channels;
        let row_bytes = &mapped[src..src + unpadded_bytes_per_row];
        let row_elems: &[T] = bytemuck::cast_slice(row_bytes);
        data[dst..dst + width as usize * channels].copy_from_slice(row_elems);
    }
    drop(mapped);
    staging.unmap();

    data
}

/// Maps a segmentation id to a deterministic, well-spread RGB color.
///
/// Id `0` (background) is always black. Other ids are hashed and converted via
/// an HSV sweep with the golden-ratio hue increment, so consecutive ids get
/// visually distinct colors.
fn colorize_id(id: u32) -> [u8; 3] {
    if id == 0 {
        return [0, 0, 0];
    }

    // Golden-ratio hue stepping for maximally distinct successive hues.
    let hue = (id as f32 * 0.618_034).fract();
    hsv_to_rgb(hue, 0.65, 0.95)
}

/// Converts an HSV color (all components in `[0, 1]`) to 8-bit RGB.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    ]
}
