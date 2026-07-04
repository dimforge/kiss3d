//! Built-in geometries, shaders and effects.

// Shared WESL modules that are composed into several shaders. Each is a single
// `static` (one fixed address) referenced everywhere, so the source bytes are
// embedded in the binary exactly ONCE — unlike repeating `include_str!(...)` per
// site (or a `const`, which is inlined at each use), where the compiler/linker
// only *sometimes* merges the duplicates.

/// The shared small-utility WESL module (`common.wgsl`): `luminance`,
/// `unpack_mat2/3`, and the full-screen-vertex helpers. Mounted as `package::common`.
pub(crate) static COMMON_WESL: &str = include_str!("common.wgsl");
/// Shared equirectangular mapping + analytic env-BRDF (`package::pbr_env`).
pub(crate) static PBR_ENV_WESL: &str = include_str!("pbr_env.wgsl");
/// Box-downsample full-screen pass, reused by IBL/probe/SSR/DoF mip chains.
pub(crate) static ENV_DOWNSAMPLE_WESL: &str = include_str!("env_downsample.wgsl");
/// Shared tonemap operators (`package::tonemap_ops`), used by both HDR resolves.
pub(crate) static TONEMAP_OPS_WESL: &str = include_str!("tonemap_ops.wgsl");
/// Shadow-map depth pre-pass (plain + `@if(skinned)` deform variants).
pub(crate) static SHADOW_DEPTH_WESL: &str = include_str!("shadow_depth.wgsl");
/// Colored-transmittance shadow pass (plain + `@if(skinned)` deform variants).
pub(crate) static SHADOW_TRANSMITTANCE_WESL: &str = include_str!("shadow_transmittance.wgsl");

/// Compiles a single shader that imports `package::common`, composing it with the
/// shared [`COMMON_WESL`] module. `modpath` is the shader's own module path (any
/// unique `package::...` name except `package::common`).
pub(crate) fn compile_shader_with_common(modpath: &str, src: &str) -> String {
    compile_wesl(
        &[(modpath, src), ("package::common", COMMON_WESL)],
        modpath,
        &[],
    )
}

/// Composes a set of in-memory WESL modules into a single WGSL string via the
/// `wesl` compiler, resolving `import`s and conditional-compilation `@if` features.
///
/// `modules` is `(module path, source)` pairs (e.g. `("package::tonemap_ops", src)`);
/// `root` is the module path to compile (its entry points + everything they reach).
/// `features` toggles `@if(name)` flags. Dead-code elimination (WESL "strip") removes
/// unreferenced declarations and their bindings, so the output only contains what the
/// root actually uses — replacing brittle source concatenation with real module
/// imports. naga validates the result at `create_shader_module`.
pub(crate) fn compile_wesl(
    modules: &[(&str, &str)],
    root: &str,
    features: &[(&str, bool)],
) -> String {
    let mut resolver = wesl::VirtualResolver::new();
    for (path, src) in modules {
        resolver.add_module(
            path.parse().expect("invalid WESL module path"),
            (*src).into(),
        );
    }
    let mut comp = wesl::Wesl::new("").set_custom_resolver(resolver);
    comp.set_options(wesl::CompileOptions {
        // wesl's own validation needs the `eval` crate feature (off); naga validates
        // at `create_shader_module` regardless.
        validate: false,
        ..Default::default()
    });
    for (name, on) in features {
        comp.set_feature(name, *on);
    }
    comp.compile(&root.parse().expect("invalid WESL root module path"))
        .unwrap_or_else(|e| panic!("WESL compilation of {} failed: {}", root, e))
        .to_string()
}

pub use self::aov::{
    AovKind, AovRenderer, DEPTH_AOV_FORMAT, NORMALS_AOV_FORMAT, SEGMENTATION_AOV_FORMAT,
};
pub use self::normals_material::{NormalsMaterial, NORMAL_FRAGMENT_SRC, NORMAL_VERTEX_SRC};
pub use self::object_material::{ObjectMaterial, OBJECT_FRAGMENT_SRC, OBJECT_VERTEX_SRC};
pub use self::uvs_material::{UvsMaterial, UVS_FRAGMENT_SRC, UVS_VERTEX_SRC};

pub use self::lit_material2d::{LitMaterial2d, LitMaterial2dGpuData, LitParams};
pub use self::object_material2d::ObjectMaterial2d;
pub use self::shadow::{ShadowMapper, MAX_SHADOW_VIEWS};
pub use self::skinned_material2d::{Bone2d, SkinVertex2d, SkinnedMesh2d, MAX_JOINTS_2D};

mod aov;
pub(crate) mod clustered;
pub mod deform;
mod normals_material;
mod object_material;
mod shadow;
mod uvs_material;

mod lit_material2d;
mod object_material2d;
mod skinned_material2d;
