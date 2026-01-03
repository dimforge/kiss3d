#!/bin/bash

# Build all kiss3d examples to WASM for the website demos
# Usage: ./scripts/build-demos.sh [example_name]
# If example_name is provided, only that example is built

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
  window
  rectangle
  primitives2d
  lines2d
  points2d
  polylines2d
  instancing2d
  cube
  primitives
  primitives_scale
  quad
  lines
  points
  polylines
  polyline_strip
  wireframe
  multi_light
  custom_mesh
  custom_mesh_shared
  procedural
  group
  post_processing
  mouse_events
  instancing3d
  custom_material
  ui
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

build_example() {
    local example=$1
    local demo_dir="$DEMOS_DIR/$example"
    local target_dir="$KISS3D_DIR/target/wasm32-unknown-unknown/release/examples"

    echo -e "${YELLOW}Building${NC} $example..."

    # Create demo directory
    mkdir -p "$demo_dir/pkg"

    # Build with cargo
    if ! cargo build \
        --manifest-path "$KISS3D_DIR/Cargo.toml" \
        --example "$example" \
        --target wasm32-unknown-unknown \
        --features parry,egui \
        --release 2>&1; then
        echo -e "${RED}✗${NC} $example (cargo build failed)"
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

    # Optimize with wasm-opt if available
    if command -v wasm-opt &> /dev/null; then
        wasm-opt -O3 "$demo_dir/pkg/example_bg.wasm" -o "$demo_dir/pkg/example_bg.wasm" 2>/dev/null || true
    fi

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

    echo -e "${GREEN}✓${NC} $example"
    return 0
}

# Main logic
check_requirements

cd "$KISS3D_DIR"

if [ -n "$1" ]; then
    # Build single example
    build_example "$1"
else
    # Build all examples
    echo -e "${BLUE}Building ${#EXAMPLES[@]} examples to WASM...${NC}"
    echo ""

    success=0
    failed=0
    failed_examples=()

    for example in "${EXAMPLES[@]}"; do
        if build_example "$example"; then
            ((success++))
        else
            ((failed++))
            failed_examples+=("$example")
        fi
        echo ""
    done

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}Success:${NC} $success"
    if [ $failed -gt 0 ]; then
        echo -e "${RED}Failed:${NC} $failed"
        echo "Failed examples: ${failed_examples[*]}"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi
