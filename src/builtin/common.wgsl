// Small shader utilities shared (via WESL imports) across many passes, so the
// renderer keeps a single definition of each instead of copy-pasting them.

// Rec. 709 relative luminance of a linear RGB color.
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Unpacks three vec4 instance columns (the storage/attribute layout pads each
// mat3 column to a vec4) into a mat3x3. Used by the 2D instanced pipelines.
fn unpack_mat3(col0: vec4<f32>, col1: vec4<f32>, col2: vec4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(col0.xyz, col1.xyz, col2.xyz);
}

// Unpacks two vec4 instance columns into a mat2x2.
fn unpack_mat2(col0: vec4<f32>, col1: vec4<f32>) -> mat2x2<f32> {
    return mat2x2<f32>(col0.xy, col1.xy);
}

// Clip-space XY of an oversized full-screen triangle for vertex index `vid` (0..3),
// covering the viewport in a single triangle with no vertex buffer. Emit the clip
// position as `vec4(fullscreen_triangle_xy(vid), z, 1.0)` (z = 0 for most passes,
// 1 for a far-plane skybox).
fn fullscreen_triangle_xy(vid: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return corners[vid];
}

// Texture UV for a clip-space XY in [-1, 1], with the wgpu Y-flip (V = 0 at the
// top of the framebuffer). Works for both the full-screen-triangle passes and the
// position-buffer quad passes.
fn fullscreen_uv_from_clip(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
}
