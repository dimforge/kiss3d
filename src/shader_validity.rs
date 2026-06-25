//! Shader-validity test (CI).
//!
//! Instantiates the engine's shaders on a real (headless) GPU device — building the
//! actual `wgpu` shader modules and pipelines through the normal object/material/
//! effect code paths — so CI catches any variant that fails to compile on the
//! backend. This is deliberately *not* a parallel naga check: it exercises exactly
//! what the engine does at runtime.
//!
//! The object PBR über-shader has `2^FEATURE_COUNT` specialized variants — far too
//! many to build as real pipelines within CI time (minutes / tens of thousands of
//! pipelines). The default test therefore uses **combinatorial** coverage (all-off,
//! all-on, every single feature, every pair of features, and each-feature-omitted),
//! which catches the single-feature and pairwise-interaction failure class, and then
//! renders real scenes/effects that instantiate every other shader. A literal,
//! exhaustive `2^N` sweep is provided as an `#[ignore]`d test for thorough local /
//! nightly runs (`cargo test -- --ignored`).
//!
//! CI runs on a GPU-less `ubuntu-latest` runner, so the workflow installs Mesa's
//! software Vulkan driver (lavapipe). If no adapter is found the test *skips* (so a
//! dev box without a usable GPU doesn't fail the suite) rather than failing.

#[cfg(test)]
mod tests {
    use crate::builtin::{Bone2d, LitParams, ObjectMaterial, SkinVertex2d, SkinnedMesh2d};
    use crate::camera::{CoordinateSystem2d, FixedView2d, OrbitCamera3d};
    use crate::light2d::{Light2d, Light2dManager};
    use crate::context::Context;
    use crate::light::Light;
    use crate::post_processing::{
        Cas, Crt, Fxaa, Grayscales, OculusStereo, PostProcessingEffect, SobelEdgeHighlight, Waves,
    };
    use crate::renderer::RayTracer;
    use crate::scene::{AlphaMode, SceneNode2d, SceneNode3d, SpriteSheet, Tilemap};
    use crate::window::OffscreenSurface;
    use glamx::{Pose2, Vec2, Vec3};

    use crate::color::Color;

    /// Is a usable GPU adapter present? (CI installs lavapipe.) When none is found we
    /// skip the test instead of failing.
    async fn adapter_available() -> bool {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .is_ok()
    }

    /// Combinatorial feature masks over `n` bits: all-off, all-on, every single bit
    /// (on and off), and every pair of bits. Covers single-feature and pairwise
    /// interactions without the full `2^n` blow-up.
    fn combinatorial_masks(n: u32) -> Vec<u32> {
        let full = (1u32 << n) - 1;
        let mut masks = vec![0u32, full];
        for i in 0..n {
            masks.push(1 << i); // single feature on
            masks.push(full & !(1 << i)); // every feature but one
            for j in (i + 1)..n {
                masks.push((1 << i) | (1 << j)); // pair of features on
            }
        }
        masks.sort_unstable();
        masks.dedup();
        masks
    }

    /// A scene touching a broad slice of the rasterizer's material features.
    fn demo_scene_3d() -> SceneNode3d {
        let mut scene = SceneNode3d::empty();
        scene
            .add_light(Light::point(80.0).with_casts_shadows(true))
            .set_position(Vec3::new(4.0, 6.0, 8.0));
        let mut ground = scene.add_cube(20.0, 0.2, 20.0);
        ground.set_color(Color::new(0.6, 0.6, 0.6, 1.0));
        ground.set_position(Vec3::new(0.0, -1.5, 0.0));
        let mut cc = scene.add_sphere(0.9);
        cc.set_metallic(1.0);
        cc.set_roughness(0.1);
        cc.set_clearcoat(1.0, 0.1);
        cc.set_position(Vec3::new(-2.5, 0.0, 0.0));
        let mut an = scene.add_sphere(0.9);
        an.set_anisotropy(0.8, 0.0);
        let mut tr = scene.add_cube(1.2, 1.2, 1.2);
        tr.set_transmission(0.7);
        tr.set_color(Color::new(0.2, 0.9, 0.3, 0.6));
        tr.set_alpha_mode(AlphaMode::Blend);
        tr.set_position(Vec3::new(2.5, 0.0, 0.0));
        scene
    }

    /// A 2D scene touching the filled / points / lines pipelines.
    fn demo_scene_2d() -> SceneNode2d {
        let mut s = SceneNode2d::empty();
        s.add_rectangle(80.0, 60.0)
            .set_color(Color::new(0.9, 0.4, 0.2, 1.0));
        s.add_circle(30.0)
            .set_color(Color::new(0.2, 0.7, 0.9, 1.0))
            .set_position(Vec2::new(60.0, 40.0));
        s.add_circle(20.0)
            .set_points_size(5.0, false)
            .set_position(Vec2::new(-60.0, -40.0));
        s.add_rectangle(50.0, 50.0)
            .set_lines_width(2.0, false)
            .set_position(Vec2::new(60.0, -40.0));
        // Non-default blend modes exercise the extra surface pipelines.
        s.add_rectangle(40.0, 40.0)
            .set_color(Color::new(0.8, 0.2, 0.2, 0.6))
            .set_blend(crate::scene::Blend2d::Additive)
            .set_position(Vec2::new(0.0, 70.0));
        s.add_rectangle(40.0, 40.0)
            .set_color(Color::new(0.2, 0.8, 0.2, 0.6))
            .set_blend(crate::scene::Blend2d::Multiply)
            .set_position(Vec2::new(20.0, 70.0));
        // Sprite quad + 9-slice mesh (object2d shader, more vertices).
        s.add_sprite(30.0, 30.0).set_position(Vec2::new(-60.0, 60.0));
        s.add_nine_slice(
            Vec2::new(60.0, 40.0),
            crate::scene::Border::uniform(8.0),
            crate::scene::Border::uniform(0.25),
        )
        .set_position(Vec2::new(-60.0, 0.0));
        // Lit material (uses the default flat normal map + global 2D lights).
        s.add_lit_sprite(40.0, 40.0)
            .set_lit_params(LitParams::default().with_specular(0.5, 24.0))
            .set_position(Vec2::new(80.0, 0.0));
        // Tilemap mesh (atlas-textured single mesh, a few tiles set).
        let mut tm = Tilemap::new(4, 4, Vec2::new(14.0, 14.0), SpriteSheet::new(2, 2));
        tm.set_tile(0, 0, 0);
        tm.set_tile(1, 1, 1);
        tm.set_tile(2, 3, 2);
        let mut tnode = tm.node();
        tnode.set_position(Vec2::new(-90.0, 70.0));
        s.add_child(tnode);
        s
    }

    /// A small 3-bone vertical strip skinned mesh.
    fn demo_skinned_mesh() -> SkinnedMesh2d {
        let mut verts = Vec::new();
        for row in 0..3u32 {
            for &x in &[-12.0f32, 12.0] {
                verts.push(SkinVertex2d {
                    position: Vec2::new(x, row as f32 * 30.0),
                    uv: Vec2::new((x + 12.0) / 24.0, row as f32 / 2.0),
                    joints: [row, 0, 0, 0],
                    weights: [1.0, 0.0, 0.0, 0.0],
                });
            }
        }
        // Two quads (rows 0-1 and 1-2).
        let faces = vec![
            [0, 1, 3],
            [0, 3, 2],
            [2, 3, 5],
            [2, 5, 4],
        ];
        let bones = vec![
            Bone2d {
                parent: None,
                local: Pose2::IDENTITY,
            },
            Bone2d {
                parent: Some(0),
                local: Pose2::from_translation(Vec2::new(0.0, 30.0)),
            },
            Bone2d {
                parent: Some(1),
                local: Pose2::from_translation(Vec2::new(0.0, 30.0)),
            },
        ];
        SkinnedMesh2d::new(verts, faces, bones)
    }

    #[test]
    fn all_shaders_instantiate() {
        crate::pollster::block_on(async {
            if !adapter_available().await {
                eprintln!("shader-validity test: no GPU adapter found, skipping");
                return;
            }
            let mut surface = OffscreenSurface::new(96, 96).await;
            let ctxt = Context::get();
            let scope = ctxt.device.push_error_scope(wgpu::ErrorFilter::Validation);

            // 1) Object material: combinatorial feature coverage as real pipelines.
            // The shaders are sample-count-independent (MSAA is fixed-function), so we
            // build every variant once at 1×, plus all-on / all-off at 4× to exercise
            // the multisampled pipeline path at least once.
            let mat = ObjectMaterial::new();
            let masks = combinatorial_masks(ObjectMaterial::FEATURE_COUNT);
            let full = (1u32 << ObjectMaterial::FEATURE_COUNT) - 1;
            let mut built = 0usize;
            for &bits in &masks {
                if mat.try_instantiate_variant_for_test(bits, 1) {
                    built += 1;
                }
            }
            for &bits in &[0u32, full] {
                if mat.try_instantiate_variant_for_test(bits, 4) {
                    built += 1;
                }
            }
            eprintln!("instantiated {built} object-material variants");

            // 2) Render real scenes that instantiate the rest of the shaders, with the
            // screen-space effects enabled (shadows, SSAO, SSR, DoF, bloom, skybox).
            surface.window_mut().set_shadows_enabled(true);
            surface.window_mut().set_ssao_enabled(true);
            surface.window_mut().set_ssr_enabled(true);
            surface.window_mut().set_dof_enabled(true);
            surface.set_bloom_enabled(true);
            let mut cam = OrbitCamera3d::new(Vec3::new(0.0, 2.0, 9.0), Vec3::ZERO);
            let mut scene = demo_scene_3d();
            for _ in 0..2 {
                surface.render_3d(&mut scene, &mut cam).await;
            }

            // 3) 2D scene (object2d / points2d / polyline2d / wireframe / sdf2d / lit2d).
            Light2dManager::get_global_manager(|m| {
                m.set_ambient(Color::new(0.1, 0.1, 0.12, 1.0));
                m.set_lights(&[
                    Light2d::point(Vec2::new(80.0, 30.0), Color::new(1.0, 0.9, 0.8, 1.0), 2.0, 200.0),
                    Light2d::spot(
                        Vec2::new(40.0, 60.0),
                        Vec2::new(0.0, -1.0),
                        Color::new(0.6, 0.8, 1.0, 1.0),
                        2.0,
                        180.0,
                        0.3,
                        0.6,
                    ),
                ]);
            });
            let mut cam2 = FixedView2d::new(CoordinateSystem2d::default(), false);
            let mut scene2 = demo_scene_2d();
            // Skinned 2D mesh: a 3-bone vertical strip, posed and rendered.
            let mut skinned = demo_skinned_mesh();
            skinned.set_bone_local(1, Pose2::new(Vec2::new(0.0, 60.0), 0.3));
            skinned.update();
            let mut snode = skinned.node();
            snode.set_position(Vec2::new(-40.0, -30.0));
            scene2.add_child(snode);
            surface.render_2d(&mut scene2, &mut cam2).await;

            // 4) Path tracer (rt_kernel / denoise / rt tonemap).
            let mut rt = RayTracer::new();
            let mut rt_scene = demo_scene_3d();
            surface.raytrace_3d(&mut rt_scene, &mut cam, &mut rt).await;

            // 5) Each post-processing effect.
            let mut effects: Vec<Box<dyn PostProcessingEffect>> = vec![
                Box::new(Fxaa::new()),
                Box::new(SobelEdgeHighlight::new(0.1)),
                Box::new(Cas::new(0.5)),
                Box::new(Grayscales::new()),
                Box::new(Waves::new()),
                Box::new(OculusStereo::new()),
                Box::new(Crt::new()),
            ];
            for eff in &mut effects {
                surface
                    .render(
                        Some(&mut scene),
                        None,
                        Some(&mut cam),
                        None,
                        None,
                        Some(eff.as_mut()),
                    )
                    .await;
            }

            // Any invalid shader/pipeline created above is captured here.
            let err = scope.pop().await;
            assert!(err.is_none(), "shader validation error: {:?}", err);
        });
    }

    /// Exhaustive: builds the real pipeline for ALL `2^FEATURE_COUNT` object-shader
    /// variants. Minutes-long (tens of thousands of pipelines), so `#[ignore]`d — run
    /// on demand with `cargo test -- --ignored`.
    #[test]
    #[ignore = "exhaustive 2^N object-shader pipeline build; minutes-long, run on demand"]
    fn all_object_shader_variants_exhaustive() {
        crate::pollster::block_on(async {
            if !adapter_available().await {
                eprintln!("shader-validity (exhaustive): no GPU adapter, skipping");
                return;
            }
            let _surface = OffscreenSurface::new(64, 64).await;
            let ctxt = Context::get();
            let scope = ctxt.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let mat = ObjectMaterial::new();
            let mut built = 0usize;
            for bits in 0..(1u32 << ObjectMaterial::FEATURE_COUNT) {
                if mat.try_instantiate_variant_for_test(bits, 1) {
                    built += 1;
                }
            }
            eprintln!("exhaustive: built {built} object-material variants");
            let err = scope.pop().await;
            assert!(err.is_none(), "shader validation error: {:?}", err);
        });
    }
}
