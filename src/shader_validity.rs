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
    use crate::builtin::ObjectMaterial;
    use crate::camera::{CoordinateSystem2d, FixedView2d, OrbitCamera3d};
    use crate::context::Context;
    use crate::light::Light;
    use crate::post_processing::{
        Cas, Fxaa, Grayscales, OculusStereo, PostProcessingEffect, SobelEdgeHighlight, Waves,
    };
    use crate::renderer::RayTracer;
    use crate::scene::{AlphaMode, SceneNode2d, SceneNode3d};
    use crate::window::OffscreenSurface;
    use glamx::{Vec2, Vec3};

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
        s
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

            // 3) 2D scene (object2d / points2d / polyline2d / wireframe).
            let mut cam2 = FixedView2d::new(CoordinateSystem2d::default(), false);
            let mut scene2 = demo_scene_2d();
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
