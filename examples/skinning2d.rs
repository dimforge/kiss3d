use kiss3d::builtin::{Bone2d, SkinVertex2d, SkinnedMesh2d};
use kiss3d::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

// Demonstrates 2D skeletal deformation with GPU skinning: a textured strip is bound
// to a chain of bones, and a traveling sine wave along the chain makes it ripple
// like a banner / tentacle. The deformation happens entirely on the GPU.
#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: 2D skeletal skinning").await;
    let mut camera = PanZoomCamera2d::new(Vec2::ZERO, 2.0);
    let mut scene = SceneNode2d::empty();

    let segments = 10usize; // bone chain length
    let seg_h = 40.0f32;
    let half_w = 40.0f32;

    // One vertex row per bone, two columns; each row is fully weighted to its bone.
    let mut verts = Vec::new();
    for row in 0..=segments {
        for &x in &[-half_w, half_w] {
            verts.push(SkinVertex2d {
                position: Vec2::new(x, row as f32 * seg_h),
                uv: Vec2::new(
                    (x + half_w) / (2.0 * half_w),
                    1.0 - row as f32 / segments as f32,
                ),
                joints: [row as u32, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            });
        }
    }
    // Two triangles per segment quad.
    let mut faces = Vec::new();
    for row in 0..segments as u32 {
        let b = row * 2;
        faces.push([b, b + 1, b + 3]);
        faces.push([b, b + 3, b + 2]);
    }
    // Bone chain: root at the base, each bone offset one segment up from its parent.
    let mut bones = vec![Bone2d {
        parent: None,
        local: Pose2::IDENTITY,
    }];
    for i in 1..=segments {
        bones.push(Bone2d {
            parent: Some(i - 1),
            local: Pose2::from_translation(Vec2::new(0.0, seg_h)),
        });
    }

    let mut skinned = SkinnedMesh2d::new(verts, faces, bones);
    skinned.set_transform(Pose2::from_translation(Vec2::new(0.0, -200.0)));
    // Embed on wasm (no filesystem); read from disk on native.
    #[cfg(not(target_arch = "wasm32"))]
    skinned
        .node()
        .set_texture_from_file(Path::new("./examples/media/kitten.png"), "kitten");
    #[cfg(target_arch = "wasm32")]
    skinned
        .node()
        .set_texture_from_memory(include_bytes!("./media/kitten.png"), "kitten");
    scene.add_child(skinned.node());

    let mut t = 0.0f32;
    while window.render_2d(&mut scene, &mut camera).await {
        t += 0.05;
        // Travel a bend wave up the chain (the root stays fixed).
        for i in 1..=segments {
            let angle = 0.35 * (t + i as f32 * 0.6).sin();
            skinned.set_bone_local(i, Pose2::new(Vec2::new(0.0, seg_h), angle));
        }
        skinned.update();
    }
}
