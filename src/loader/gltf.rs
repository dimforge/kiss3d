//! glTF / GLB loader.
//!
//! Loads a glTF 2.0 asset (`.gltf` + external buffers/images, or self-contained
//! `.glb`) into a kiss3d [`GltfModel`]: a scene-graph subtree mirroring the glTF
//! node hierarchy, plus an [`AnimationPlayer`] holding every animation in the
//! file. Meshes become regular scene nodes; PBR materials map onto kiss3d's
//! metallic-roughness [`Object3d`] surface; skinned meshes carry a [`Skin3d`]
//! that drives GPU vertex skinning.
//!
//! Use [`SceneNode3d::add_gltf`](crate::scene::SceneNode3d::add_gltf) rather than
//! calling [`load`] directly in most cases.

use crate::color::Color;
use crate::resource::{
    GpuMesh3d, MaterialManager3d, MorphTargets, SkinVertexData, Texture, TextureManager,
};
use crate::scene::{
    AnimationChannel, AnimationClip, AnimationPlayer, GltfModel, Interpolation, Object3d,
    SceneNode3d, Skin3d,
};
use glamx::{Mat4, Pose3, Quat, Vec2, Vec3};
use image::DynamicImage;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

/// One renderable primitive of a glTF mesh: a shareable GPU mesh plus the index
/// of the material that shades it.
struct PrimInfo {
    mesh: Rc<RefCell<GpuMesh3d>>,
    material_index: Option<usize>,
}

/// Loads a glTF/GLB file into a [`GltfModel`].
///
/// The returned model's `root` is an unrooted group node holding the file's
/// default scene; attach it to your scene graph (or use
/// [`SceneNode3d::add_gltf`](crate::scene::SceneNode3d::add_gltf), which does that
/// for you). `player` is stopped initially — call
/// [`AnimationPlayer::play`](crate::scene::AnimationPlayer::play) to start one.
pub fn load(path: &Path) -> Result<GltfModel, gltf::Error> {
    let (doc, buffers, images) = gltf::import(path)?;
    let prefix = path.to_string_lossy().into_owned();

    // Decode every image once into an `image::DynamicImage`; textures are then
    // created on demand per (image, color-space) pair.
    let images: Vec<DynamicImage> = images.iter().map(image_data_to_dynamic).collect();
    let mut tex_cache: HashMap<(usize, bool), Arc<Texture>> = HashMap::new();

    // Build GPU meshes for every primitive of every mesh, keyed by mesh index.
    let mut meshes: Vec<Vec<PrimInfo>> = Vec::with_capacity(doc.meshes().count());
    for mesh in doc.meshes() {
        let mut prims = Vec::new();
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                log::warn!(
                    "gltf: skipping non-triangle primitive (mode {:?}) in mesh {:?}",
                    prim.mode(),
                    mesh.name().unwrap_or("")
                );
                continue;
            }
            if let Some(gpu) = build_primitive(&prim, &buffers) {
                prims.push(PrimInfo {
                    mesh: Rc::new(RefCell::new(gpu)),
                    material_index: prim.material().index(),
                });
            }
        }
        meshes.push(prims);
    }

    // Create one scene node per glTF node (no parenting yet), with its local TRS.
    let mut node_map: Vec<SceneNode3d> = doc
        .nodes()
        .map(|node| {
            let (t, r, s) = node.transform().decomposed();
            let pose = Pose3::from_parts(Vec3::from_array(t), Quat::from_array(r));
            SceneNode3d::new(Vec3::from_array(s), pose, None)
        })
        .collect();

    let material = MaterialManager3d::get_global_manager(|mm| mm.get_default());
    let default_tex = TextureManager::get_global_manager(|tm| tm.get_default());

    // Attach mesh primitives (as child object nodes) and skins to each node.
    for node in doc.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        let skin_data = node.skin().map(|skin| build_skin_parts(&skin, &buffers, &node_map));

        for prim in &meshes[mesh.index()] {
            let mut object = Object3d::new(
                prim.mesh.clone(),
                Color::new(1.0, 1.0, 1.0, 1.0),
                default_tex.clone(),
                material.clone(),
            );
            if let Some(mat_index) = prim.material_index {
                if let Some(gltf_mat) = doc.materials().nth(mat_index) {
                    apply_material(&mut object, &gltf_mat, &images, &mut tex_cache, &prefix);
                }
            }
            // A skinned mesh only deforms if it actually carries per-vertex
            // joints/weights; otherwise it stays a rigid child of the node.
            if prim.mesh.borrow().has_skin_vertices() {
                if let Some((joints, inverse_bind)) = &skin_data {
                    object.set_skin(Skin3d::new(joints.clone(), inverse_bind.clone()));
                }
            }
            // Seed the morph weights (one per target) from the node's or mesh's glTF
            // default weights, padded/truncated to the primitive's target count.
            let num_targets = prim.mesh.borrow().morph_target_count();
            if num_targets > 0 {
                let mut weights = vec![0.0f32; num_targets];
                if let Some(src) = node.weights().or_else(|| mesh.weights()) {
                    let n = src.len().min(num_targets);
                    weights[..n].copy_from_slice(&src[..n]);
                }
                object.data_mut().set_morph_weights(&weights);
            }
            node_map[node.index()].add_object(Vec3::ONE, Pose3::IDENTITY, object);
        }
    }

    // Link the node hierarchy.
    for node in doc.nodes() {
        let mut parent = node_map[node.index()].clone();
        for child in node.children() {
            parent.add_child(node_map[child.index()].clone());
        }
    }

    // Assemble the model root from the default (or first) scene's root nodes.
    let mut root = SceneNode3d::empty();
    let scene = doc.default_scene().or_else(|| doc.scenes().next());
    if let Some(scene) = scene {
        for node in scene.nodes() {
            root.add_child(node_map[node.index()].clone());
        }
    }

    // Build the animation clips.
    let clips = doc
        .animations()
        .map(|anim| build_clip(&anim, &buffers, &node_map))
        .collect();

    Ok(GltfModel {
        root,
        player: AnimationPlayer::new(clips),
    })
}

/// Reads one triangle primitive into a [`GpuMesh3d`], including optional skinning
/// attributes. Returns `None` if the primitive has no positions.
fn build_primitive(prim: &gltf::Primitive, buffers: &[gltf::buffer::Data]) -> Option<GpuMesh3d> {
    let reader = prim.reader(|b| Some(&buffers[b.index()]));

    let positions: Vec<Vec3> = reader.read_positions()?.map(Vec3::from_array).collect();
    let num_vertices = positions.len();

    let faces: Vec<[u32; 3]> = match reader.read_indices() {
        Some(indices) => indices
            .into_u32()
            .collect::<Vec<u32>>()
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect(),
        None => (0..positions.len() as u32)
            .collect::<Vec<u32>>()
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect(),
    };

    let normals: Option<Vec<Vec3>> = reader
        .read_normals()
        .map(|it| it.map(Vec3::from_array).collect());
    let uvs: Option<Vec<Vec2>> = reader
        .read_tex_coords(0)
        .map(|it| it.into_f32().map(Vec2::from_array).collect());

    let mut mesh = GpuMesh3d::new(positions, faces, normals, uvs, false);

    // Skinning attributes: present together on skinned primitives. JOINTS_0 is
    // widened from u8/u16 to u32 so a single vertex format covers every mesh.
    if let (Some(joints), Some(weights)) = (reader.read_joints(0), reader.read_weights(0)) {
        let joints: Vec<[u32; 4]> = joints
            .into_u16()
            .map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32])
            .collect();
        let weights: Vec<[f32; 4]> = weights.into_f32().collect();
        mesh.set_skin_vertices(SkinVertexData::new(joints, weights));
    }

    // Morph targets: per-target position (and optional normal) deltas, flattened to
    // `[target * num_vertices + vertex]`. Read from `read_morph_targets`, which
    // yields each target's optional position/normal/tangent iterators (tangents are
    // unused — kiss3d derives tangents from screen-space derivatives).
    let mut morph_positions: Vec<[f32; 4]> = Vec::new();
    let mut morph_normals: Vec<[f32; 4]> = Vec::new();
    let mut num_targets = 0usize;
    let mut any_normals = false;
    for (pos_iter, nrm_iter, _tan_iter) in reader.read_morph_targets() {
        if num_targets >= crate::builtin::deform::MAX_MORPH_TARGETS {
            log::warn!(
                "gltf: mesh has more than {} morph targets; extra targets ignored",
                crate::builtin::deform::MAX_MORPH_TARGETS
            );
            break;
        }
        num_targets += 1;
        match pos_iter {
            Some(it) => morph_positions.extend(it.map(|p| [p[0], p[1], p[2], 0.0])),
            None => morph_positions.extend(std::iter::repeat_n([0.0; 4], num_vertices)),
        }
        match nrm_iter {
            Some(it) => {
                any_normals = true;
                morph_normals.extend(it.map(|n| [n[0], n[1], n[2], 0.0]));
            }
            None => morph_normals.extend(std::iter::repeat_n([0.0; 4], num_vertices)),
        }
    }
    if num_targets > 0 {
        let normals = any_normals.then_some(morph_normals);
        mesh.set_morph_targets(MorphTargets::new(
            num_targets,
            num_vertices,
            morph_positions,
            normals,
        ));
    }

    Some(mesh)
}

/// Builds the (joint node weak handles, inverse bind matrices) for a glTF skin.
fn build_skin_parts(
    skin: &gltf::Skin,
    buffers: &[gltf::buffer::Data],
    node_map: &[SceneNode3d],
) -> (Vec<std::rc::Weak<RefCell<crate::scene::SceneNodeData3d>>>, Vec<Mat4>) {
    let joints: Vec<_> = skin
        .joints()
        .map(|j| node_map[j.index()].downgrade())
        .collect();

    let reader = skin.reader(|b| Some(&buffers[b.index()]));
    let inverse_bind: Vec<Mat4> = match reader.read_inverse_bind_matrices() {
        Some(it) => it.map(|m| Mat4::from_cols_array_2d(&m)).collect(),
        None => vec![Mat4::IDENTITY; joints.len()],
    };

    (joints, inverse_bind)
}

/// Maps a glTF material's metallic-roughness parameters and textures onto an
/// [`Object3d`].
fn apply_material(
    object: &mut Object3d,
    material: &gltf::Material,
    images: &[DynamicImage],
    cache: &mut HashMap<(usize, bool), Arc<Texture>>,
    prefix: &str,
) {
    let pbr = material.pbr_metallic_roughness();
    let bc = pbr.base_color_factor();
    object.set_color(Color::new(bc[0], bc[1], bc[2], bc[3]));
    object.set_metallic(pbr.metallic_factor());
    object.set_roughness(pbr.roughness_factor());
    let ef = material.emissive_factor();
    object.set_emissive(Color::new(ef[0], ef[1], ef[2], 1.0));

    // Color/emissive textures are sRGB; data textures (normal/MR/AO) are linear.
    if let Some(info) = pbr.base_color_texture() {
        object.set_texture(get_texture(
            images,
            cache,
            prefix,
            info.texture().source().index(),
            true,
        ));
    }
    if let Some(info) = pbr.metallic_roughness_texture() {
        object.set_metallic_roughness_map(get_texture(
            images,
            cache,
            prefix,
            info.texture().source().index(),
            false,
        ));
    }
    if let Some(nt) = material.normal_texture() {
        object.set_normal_map(get_texture(
            images,
            cache,
            prefix,
            nt.texture().source().index(),
            false,
        ));
    }
    if let Some(ot) = material.occlusion_texture() {
        object.set_ao_map(get_texture(
            images,
            cache,
            prefix,
            ot.texture().source().index(),
            false,
        ));
    }
    if let Some(info) = material.emissive_texture() {
        object.set_emissive_map(get_texture(
            images,
            cache,
            prefix,
            info.texture().source().index(),
            true,
        ));
    }
}

/// Returns the kiss3d texture for a glTF image source in the requested color
/// space, registering and caching it on first use.
fn get_texture(
    images: &[DynamicImage],
    cache: &mut HashMap<(usize, bool), Arc<Texture>>,
    prefix: &str,
    source: usize,
    srgb: bool,
) -> Arc<Texture> {
    if let Some(t) = cache.get(&(source, srgb)) {
        return t.clone();
    }
    // `get_global_manager` takes an `FnMut`, so the `DynamicImage` is handed off
    // through an `Option::take` rather than moved directly out of the closure.
    let mut image = Some(images[source].clone());
    let name = format!("{prefix}::img{source}::srgb{}", srgb as u8);
    let tex = TextureManager::get_global_manager(move |tm| {
        tm.add_image_with_color_space(image.take().unwrap(), &name, srgb)
    });
    cache.insert((source, srgb), tex.clone());
    tex
}

/// Builds an [`AnimationClip`] from a glTF animation, binding each channel to its
/// target node in `node_map`.
fn build_clip(
    anim: &gltf::Animation,
    buffers: &[gltf::buffer::Data],
    node_map: &[SceneNode3d],
) -> AnimationClip {
    let name = anim.name().unwrap_or("").to_string();
    let mut channels = Vec::new();

    for channel in anim.channels() {
        let target = node_map[channel.target().node().index()].clone();
        let interp = match channel.sampler().interpolation() {
            gltf::animation::Interpolation::Linear => Interpolation::Linear,
            gltf::animation::Interpolation::Step => Interpolation::Step,
            gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
        };

        let reader = channel.reader(|b| Some(&buffers[b.index()]));
        let Some(times) = reader.read_inputs() else {
            continue;
        };
        let times: Vec<f32> = times.collect();

        let Some(outputs) = reader.read_outputs() else {
            continue;
        };
        let ch = match outputs {
            gltf::animation::util::ReadOutputs::Translations(it) => {
                let values = it.map(Vec3::from_array).collect();
                AnimationChannel::translation(target, times, values, interp)
            }
            gltf::animation::util::ReadOutputs::Scales(it) => {
                let values = it.map(Vec3::from_array).collect();
                AnimationChannel::scale(target, times, values, interp)
            }
            gltf::animation::util::ReadOutputs::Rotations(rot) => {
                let values = rot.into_f32().map(Quat::from_array).collect();
                AnimationChannel::rotation(target, times, values, interp)
            }
            gltf::animation::util::ReadOutputs::MorphTargetWeights(weights) => {
                let values: Vec<f32> = weights.into_f32().collect();
                let num_keys = times.len();
                if num_keys == 0 {
                    continue;
                }
                // glTF packs `num_targets` weights per keyframe (×3 for cubic spline,
                // which stores in/value/out tangents).
                let per_key = values.len() / num_keys;
                let num_targets = if interp == Interpolation::CubicSpline {
                    per_key / 3
                } else {
                    per_key
                };
                if num_targets == 0 {
                    continue;
                }
                AnimationChannel::morph_weights(target, times, values, num_targets, interp)
            }
        };
        channels.push(ch);
    }

    AnimationClip::new(name, channels)
}

/// Converts a decoded glTF image into an `image::DynamicImage`.
fn image_data_to_dynamic(data: &gltf::image::Data) -> DynamicImage {
    use gltf::image::Format;
    let w = data.width;
    let h = data.height;
    let px = data.pixels.clone();
    match data.format {
        Format::R8 => DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w, h, px).expect("gltf: bad R8 image"),
        ),
        Format::R8G8 => DynamicImage::ImageLumaA8(
            image::ImageBuffer::from_raw(w, h, px).expect("gltf: bad R8G8 image"),
        ),
        Format::R8G8B8 => DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(w, h, px).expect("gltf: bad R8G8B8 image"),
        ),
        Format::R8G8B8A8 => DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(w, h, px).expect("gltf: bad R8G8B8A8 image"),
        ),
        // 16-bit and float formats are uncommon for textures; up/down-convert via
        // the matching image buffer, falling back to an 8-bit RGBA copy.
        Format::R16 => DynamicImage::ImageLuma16(
            image::ImageBuffer::from_raw(w, h, bytemuck::cast_slice(&px).to_vec())
                .expect("gltf: bad R16 image"),
        ),
        Format::R16G16 => DynamicImage::ImageLumaA16(
            image::ImageBuffer::from_raw(w, h, bytemuck::cast_slice(&px).to_vec())
                .expect("gltf: bad R16G16 image"),
        ),
        Format::R16G16B16 => DynamicImage::ImageRgb16(
            image::ImageBuffer::from_raw(w, h, bytemuck::cast_slice(&px).to_vec())
                .expect("gltf: bad R16G16B16 image"),
        ),
        Format::R16G16B16A16 => DynamicImage::ImageRgba16(
            image::ImageBuffer::from_raw(w, h, bytemuck::cast_slice(&px).to_vec())
                .expect("gltf: bad R16G16B16A16 image"),
        ),
        Format::R32G32B32FLOAT => DynamicImage::ImageRgb32F(
            image::ImageBuffer::from_raw(w, h, bytemuck::cast_slice(&px).to_vec())
                .expect("gltf: bad RGB32F image"),
        ),
        Format::R32G32B32A32FLOAT => DynamicImage::ImageRgba32F(
            image::ImageBuffer::from_raw(w, h, bytemuck::cast_slice(&px).to_vec())
                .expect("gltf: bad RGBA32F image"),
        ),
    }
}
