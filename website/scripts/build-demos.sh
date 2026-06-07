#!/bin/bash

# Build all kiss3d examples to WASM for the website demos
# Usage: ./scripts/build-demos.sh [example_name]
# If example_name is provided, only that example is built

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"  # absolute path for xargs re-invocation
WEBSITE_DIR="$(dirname "$SCRIPT_DIR")"
KISS3D_DIR="$(dirname "$WEBSITE_DIR")"
DEMOS_DIR="$WEBSITE_DIR/static/demos"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# All examples to build (excluding ones that won't work in WASM or that are not interesting at all)
EXAMPLES=(
  # Basics
  cube
  group
  camera_modes
  window
  # 2D
  rectangle
  primitives2d
  lines2d
  points2d
  polylines2d
  instancing2d
  mouse_events
  dda_raycast2d
  # Geometry & meshes
  primitives
  primitives_scale
  quad
  lines
  points
  polylines
  polyline_strip
  wireframe
  custom_mesh
  custom_mesh_shared
  procedural
  instancing3d
  # Materials & textures
  material_pbr
  texturing
  texturing_mipmaps
  parallax
  custom_material
  # Lighting & shadows
  shadows
  clustered_lights
  skybox
  fog
  # Reflections & refraction
  reflections
  mirror
  mirror_sphere
  transmission
  transparency
  # Post-processing
  post_processing
  hdr_bloom
  tonemapping
  color_grading
  antialiasing
  depth_of_field
  # Ray tracing
  raytracing
  raytracing_bsdf
  raytracing_denoise
  raytracing_transparency
  # Loading & animation
  gltf
  # UI & tools
  ui
  inspector
  text
)
# Check for required tools
check_requirements() {
    local missing=()

    if ! command -v cargo &> /dev/null; then
        missing+=("cargo")
    fi

    if ! command -v wasm-bindgen &> /dev/null; then
        missing+=("wasm-bindgen (install with: cargo install wasm-bindgen-cli)")
    fi

    if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
        missing+=("wasm32-unknown-unknown target (install with: rustup target add wasm32-unknown-unknown)")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}Error: Missing required tools:${NC}"
        for tool in "${missing[@]}"; do
            echo "  - $tool"
        done
        exit 1
    fi
}

# Tunables (override via environment):
#   JOBS=N            parallel post-processing jobs   (default: CPU cores)
#   WASM_OPT_FLAGS=…  wasm-opt optimization flags      (default: -O3)
#   SKIP_WASM_OPT=1   skip wasm-opt entirely (fast iteration; larger .wasm)
#   FEATURES=…        cargo features                   (default: egui,rt_switcher)
detect_cores() {
    if command -v nproc &> /dev/null; then
        nproc
    elif command -v sysctl &> /dev/null; then
        sysctl -n hw.logicalcpu 2>/dev/null || echo 4
    else
        echo 4
    fi
}
JOBS="${JOBS:-$(detect_cores)}"
WASM_OPT_FLAGS="${WASM_OPT_FLAGS:--O3}"
FEATURES="${FEATURES:-egui,rt_switcher}"

# Compile every example in a SINGLE cargo invocation. Cargo compiles the kiss3d
# library once and parallelizes codegen of all the example crates across cores —
# far faster than re-launching cargo once per example in a serial loop.
cargo_build_all() {
    local args=(build
        --manifest-path "$KISS3D_DIR/Cargo.toml"
        --target wasm32-unknown-unknown
        --features "$FEATURES"
        --release)
    local e
    for e in "$@"; do
        args+=(--example "$e")
    done
    cargo "${args[@]}"
}

# Post-process one already-compiled example: wasm-bindgen + wasm-opt + index.html.
# These steps are independent per example and CPU-bound, so they are run in
# parallel across examples (see the xargs pool below).
postprocess_example() {
    local example=$1
    local demo_dir="$DEMOS_DIR/$example"
    local target_dir="$KISS3D_DIR/target/wasm32-unknown-unknown/release/examples"

    mkdir -p "$demo_dir/pkg"

    if [ ! -f "$target_dir/$example.wasm" ]; then
        echo -e "${RED}✗${NC} $example (no .wasm — cargo build failed?)"
        return 1
    fi

    # Generate JS bindings with wasm-bindgen
    if ! wasm-bindgen \
        "$target_dir/$example.wasm" \
        --out-dir "$demo_dir/pkg" \
        --out-name example \
        --target web \
        --no-typescript 2>&1; then
        echo -e "${RED}✗${NC} $example (wasm-bindgen failed)"
        return 1
    fi

    # Optimize with wasm-opt if available (skippable for fast iteration)
    if [ -z "$SKIP_WASM_OPT" ] && command -v wasm-opt &> /dev/null; then
        wasm-opt $WASM_OPT_FLAGS "$demo_dir/pkg/example_bg.wasm" -o "$demo_dir/pkg/example_bg.wasm" 2>/dev/null || true
    fi

    write_index_html "$demo_dir"

    echo -e "${GREEN}✓${NC} $example"
    return 0
}

write_index_html() {
    local demo_dir=$1
    # Create index.html for the demo
    cat > "$demo_dir/index.html" << 'HTMLEOF'
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>kiss3d Demo</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body {
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #1a1a2e;
    }
    canvas {
      width: 100% !important;
      height: 100% !important;
      display: block;
    }
    .loading {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      color: #ff7b29;
      font-family: system-ui, sans-serif;
      font-size: 14px;
      text-align: center;
    }
    .loading::after {
      content: '';
      display: block;
      width: 30px;
      height: 30px;
      margin: 10px auto;
      border: 3px solid #333;
      border-top-color: #ff7b29;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    .error {
      color: #ff6b6b;
    }
    .error::after {
      display: none;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="loading" id="loading">Loading WebAssembly...</div>
  <script type="module">
    import init from './pkg/example.js';

    init().then(() => {
      document.getElementById('loading').style.display = 'none';
    }).catch(err => {
      console.error('WASM Error:', err);
      const loading = document.getElementById('loading');
      loading.className = 'loading error';
      loading.textContent = 'Error: ' + err.message;
    });
  </script>
</body>
</html>
HTMLEOF
}

# Internal entry point used by the parallel xargs pool: post-process a single
# example that has already been compiled. Short-circuits before the main flow.
if [ "$1" = "__postprocess_one" ]; then
    postprocess_example "$2"
    exit $?
fi

# Main logic
check_requirements

cd "$KISS3D_DIR"

if [ -n "$1" ]; then
    # Build a single example (compile + post-process).
    echo -e "${BLUE}Building${NC} $1..."
    if cargo_build_all "$1"; then
        postprocess_example "$1"
    else
        echo -e "${RED}✗${NC} $1 (cargo build failed)"
        exit 1
    fi
else
    # Build all examples.
    echo -e "${BLUE}Compiling ${#EXAMPLES[@]} examples to WASM (single cargo build)...${NC}"
    if ! cargo_build_all "${EXAMPLES[@]}"; then
        echo -e "${RED}cargo build failed.${NC} Post-processing the examples that did compile..."
    fi
    echo ""

    echo -e "${BLUE}Post-processing with up to ${JOBS} parallel jobs...${NC}"
    echo ""

    # Run wasm-bindgen + wasm-opt for every example in parallel across cores by
    # re-invoking this script via the __postprocess_one entry point.
    printf '%s\n' "${EXAMPLES[@]}" \
        | xargs -P "$JOBS" -I {} bash "$SELF" __postprocess_one {} || true

    # Tally results from what actually landed on disk.
    success=0
    failed=0
    failed_examples=()
    for example in "${EXAMPLES[@]}"; do
        if [ -f "$DEMOS_DIR/$example/pkg/example_bg.wasm" ]; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
            failed_examples+=("$example")
        fi
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}Success:${NC} $success"
    if [ $failed -gt 0 ]; then
        echo -e "${RED}Failed:${NC} $failed"
        echo "Failed examples: ${failed_examples[*]}"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    [ $failed -eq 0 ] || exit 1
fi
