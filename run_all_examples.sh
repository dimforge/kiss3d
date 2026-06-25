#!/bin/bash
# Run all kiss3d examples sequentially.
# Close each window to proceed to the next example.

set -e

cd "$(dirname "$0")"

# All examples (extracted from examples/*.rs)
EXAMPLES=(
    cube
    primitives
    primitives_scale
    primitives2d
    blend_modes2d
    sprites2d
    post_processing2d
    lighting2d
    tilemap2d
    clustered_lights
    wireframe
    lines
    lines2d
    points
    points2d
    text
    group
    add_remove
    camera
    dda_raycast2d
    event
    mouse_events
    custom_mesh
    custom_mesh_shared
    custom_material
    procedural
    quad
    rectangle
    obj
    gltf
    texturing
    texturing_mipmaps
    stereo
    post_processing
    hdr_bloom
    tonemapping
    color_grading
    antialiasing
    material_pbr
    fog
    camera_modes
    skybox
    reflections
    depth_of_field
    transmission
    mirror
    mirror_sphere
    parallax
    shadows
    transparency
    aov
    raytracing
    raytracing_bsdf
    raytracing_denoise
    raytracing_offscreen
    raytracing_transparency
    instancing2d
    instancing3d
    polylines
    polyline_strip
    polylines2d
    screenshot
    offscreen
    recording
    window
    multi_windows
    ui
    inspector
)

echo "Running ${#EXAMPLES[@]} kiss3d examples..."
echo "Close each window to proceed to the next example."
echo ""

for example in "${EXAMPLES[@]}"; do
    echo "=== Running: $example ==="
    cargo run --release --example "$example" --features egui,rt_switcher
    echo ""
done

echo "All examples completed!"
