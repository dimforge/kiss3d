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
    multi_light
    wireframe
    lines
    lines2d
    points
    points2d
    text
    group
    add_remove
    camera
    event
    mouse_events
    custom_mesh
    custom_mesh_shared
    custom_material
    procedural
    quad
    rectangle
    obj
    texturing
    texturing_mipmaps
    decomp
    stereo
    post_processing
    instancing2d
    instancing3d
    polylines
    polyline_strip
    polylines2d
    screenshot
    recording
    window
    multi_windows
    ui
)

echo "Running ${#EXAMPLES[@]} kiss3d examples..."
echo "Close each window to proceed to the next example."
echo ""

for example in "${EXAMPLES[@]}"; do
    echo "=== Running: $example ==="
    cargo run --release --example "$example" --features egui,parry
    echo ""
done

echo "All examples completed!"
