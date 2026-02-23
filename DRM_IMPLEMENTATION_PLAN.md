# DRM Display Output Implementation Plan

**Project**: kiss3d DRM Support
**Status**: Phase 1 & 2 Complete, Phase 3 In Progress
**Last Updated**: 2024

---

## Overview

This document outlines the complete implementation plan for adding DRM (Direct Rendering Manager) display output support to kiss3d, enabling rendering directly to displays without a window manager (console-only systems like Raspberry Pi).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         kiss3d                              │
│                                                             │
│  ┌─────────────┐     ┌──────────────┐    ┌──────────────┐ │
│  │  DRMWindow  │────▶│  DrmCanvas   │────▶│  wgpu        │ │
│  └─────────────┘     └──────────────┘    └──────────────┘ │
│                              │                              │
│                              ▼                              │
│                      ┌──────────────┐                      │
│                      │ RenderMode   │                      │
│                      └──────────────┘                      │
│                       │           │                        │
│               ┌───────┘           └────────┐              │
│               ▼                             ▼              │
│       ┌──────────────┐            ┌──────────────┐       │
│       │  Offscreen   │            │   Display    │       │
│       │  (existing)  │            │DrmDisplayState│      │
│       └──────────────┘            └──────────────┘       │
│                                            │              │
└────────────────────────────────────────────┼──────────────┘
                                             │
                 ┌───────────────────────────┼───────────────────────┐
                 │        Linux Kernel       │                       │
                 │                           ▼                       │
                 │  ┌──────────┐      ┌──────────┐     ┌─────────┐ │
                 │  │   DRM    │◀────▶│   GBM    │◀───▶│  GPU    │ │
                 │  │(KMS API) │      │(Buffers) │     │ Driver  │ │
                 │  └──────────┘      └──────────┘     └─────────┘ │
                 │       │                                          │
                 │       ▼                                          │
                 │  ┌──────────┐                                   │
                 │  │ Display  │                                   │
                 │  │Hardware  │                                   │
                 │  └──────────┘                                   │
                 └──────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation ✅ **COMPLETE**

**Goal**: Establish display resource discovery infrastructure

**Components Implemented**:
- ✅ Enhanced error types (DrmError, GbmError, ModesetError, etc.)
- ✅ Helper structs (DisplayConfig, FormatInfo, DrmDisplayState)
- ✅ Display resource query functions
  - `query_display_resources()` - Main entry point
  - `find_connected_connector()` - Finds active display
  - `find_available_crtc()` - Selects display controller
  - `select_best_mode()` - Chooses optimal resolution
  - `choose_formats()` - Format compatibility selection
- ✅ Drop implementation for DrmDisplayState (resource cleanup)

**Testing**: `examples/drm_test.rs` Phase 1 section
- Opens DRM devices with fallback
- Queries connectors, encoders, CRTCs
- Enumerates display modes
- Validates query logic

**Files Modified**:
- `src/window/drm/drm_canvas.rs` - Core implementation
- `src/window/drm/card.rs` - DRM device wrapper

---

### Phase 2: GBM Integration ✅ **COMPLETE**

**Goal**: Initialize GBM for GPU buffer management

**Components Implemented**:
- ✅ RenderMode enum (Offscreen vs Display)
- ✅ DrmDisplayState struct with complete field set
- ✅ `new_with_display()` constructor
  - Opens DRM device
  - Queries display configuration
  - Creates GBM device
  - Creates GBM surface (buffer pool)
  - Initializes wgpu
  - Sets up offscreen buffers
  - Builds complete display state
- ✅ Updated `present()` method signature to return Result

**Buffer Strategy**: CPU Copy Approach
- Render to offscreen wgpu texture
- Copy to GBM buffer before display
- Simple, works on all hardware
- Future optimization: DMA-BUF zero-copy

**Testing**: `examples/drm_test.rs` Phase 2 section
- Creates GBM device
- Validates format support
- Allocates GBM surface
- Tests buffer locking/unlocking

**Files Modified**:
- `src/window/drm/drm_canvas.rs` - GBM integration
- `src/window/drm/drm_window.rs` - Updated present() call

---

### Phase 3: Display Output 🚧 **IN PROGRESS**

**Goal**: Actually display rendered frames on screen

#### Step 1: Initial Mode Setting
```rust
fn set_initial_mode(display: &mut DrmDisplayState) -> Result<(), DrmCanvasError> {
    // 1. Get a buffer from GBM surface
    let bo = display.gbm_surface.lock_front_buffer()?;
    
    // 2. Create DRM framebuffer
    let fb = create_framebuffer(&display.card, &bo, display.mode.size())?;
    
    // 3. Set CRTC to display mode
    display.card.set_crtc(
        display.crtc,
        Some(fb),
        (0, 0),                    // Position
        &[display.connector],       // Connectors
        Some(display.mode)          // Mode
    )?;
    
    // 4. Store state
    display.current_fb = Some(fb);
    display.front_buffer = Some(bo);
    
    Ok(())
}
```

**Implementation Tasks**:
- [ ] Add `create_framebuffer()` helper function
- [ ] Call `set_initial_mode()` from `new_with_display()`
- [ ] Handle permission errors (requires root or DRM master)
- [ ] Store initial buffer and framebuffer

**Testing**:
- Screen should show last rendered content
- No tearing or corruption
- Proper resolution and refresh rate

---

#### Step 2: Framebuffer Management
```rust
fn create_framebuffer(
    card: &Card,
    bo: &gbm::BufferObject,
    (width, height): (u16, u16)
) -> Result<framebuffer::Handle, DrmCanvasError> {
    let handle = bo.handle();
    let stride = bo.stride();
    let format = bo.format();
    
    // Convert GBM format to DRM fourcc
    let fourcc = gbm_format_to_drm_fourcc(format);
    
    // Create framebuffer
    card.add_framebuffer(&handle, width as u32, height as u32, stride, fourcc)
        .map_err(|e| DrmCanvasError::DrmError(format!("Failed to create FB: {}", e)))
}

fn get_or_create_framebuffer(
    display: &mut DrmDisplayState,
    bo: &gbm::BufferObject
) -> Result<framebuffer::Handle, DrmCanvasError> {
    let bo_ptr = bo as *const _ as usize;
    
    // Check cache
    if let Some(&fb) = display.framebuffer_cache.get(&bo_ptr) {
        return Ok(fb);
    }
    
    // Create new framebuffer
    let (w, h) = display.mode.size();
    let fb = create_framebuffer(&display.card, bo, (w, h))?;
    
    // Cache it
    display.framebuffer_cache.insert(bo_ptr, fb);
    
    Ok(fb)
}
```

**Implementation Tasks**:
- [ ] Implement `create_framebuffer()` helper
- [ ] Implement `get_or_create_framebuffer()` with caching
- [ ] Add format conversion helper
- [ ] Test framebuffer creation with different formats

---

#### Step 3: Frame Rendering and Copy
```rust
fn copy_wgpu_to_gbm(
    offscreen_texture: &wgpu::Texture,
    gbm_bo: &mut gbm::BufferObject,
    width: u32,
    height: u32
) -> Result<(), DrmCanvasError> {
    // 1. Read pixels from wgpu texture to CPU buffer
    let mut pixels = Vec::new();
    // (Use existing read_pixels logic from screenshot)
    
    // 2. Map GBM buffer for writing
    let mut mapping = gbm_bo.map(...)?;
    let buffer = mapping.as_mut();
    
    // 3. Copy with format conversion (BGRA -> XRGB)
    for y in 0..height {
        for x in 0..width {
            let src_idx = ((y * width + x) * 4) as usize;
            let dst_idx = ((y * width + x) * 4) as usize;
            
            // BGRA8 -> XRGB8888
            buffer[dst_idx + 0] = pixels[src_idx + 2]; // B -> B
            buffer[dst_idx + 1] = pixels[src_idx + 1]; // G -> G
            buffer[dst_idx + 2] = pixels[src_idx + 0]; // R -> R
            buffer[dst_idx + 3] = 255;                  // X (unused)
        }
    }
    
    // 4. Unmap (commits changes)
    drop(mapping);
    
    Ok(())
}
```

**Implementation Tasks**:
- [ ] Implement `copy_wgpu_to_gbm()` function
- [ ] Handle format conversion (BGRA8Unorm → XRGB8888)
- [ ] Optimize copy with SIMD if needed
- [ ] Add error handling for mapping failures

---

#### Step 4: Page Flipping
```rust
fn present_to_display(
    &mut self,
    display: &mut DrmDisplayState
) -> Result<(), DrmCanvasError> {
    // 1. Lock back buffer for rendering
    let bo = unsafe { display.gbm_surface.lock_front_buffer()? };
    
    // 2. Copy rendered frame to GBM buffer
    copy_wgpu_to_gbm(
        &self.offscreen_buffers.color_texture,
        &mut bo,
        self.surface_config.width,
        self.surface_config.height
    )?;
    
    // 3. Get or create framebuffer
    let fb = get_or_create_framebuffer(display, &bo)?;
    
    // 4. Queue page flip
    display.card.page_flip(
        display.crtc,
        fb,
        PageFlipFlags::EVENT,  // Request vsync event
        None                    // User data
    )?;
    
    // 5. Wait for vsync (blocking for now)
    wait_for_vblank(&display.card)?;
    
    // 6. Release old buffer
    if let Some(old_bo) = display.front_buffer.take() {
        drop(old_bo);
    }
    
    // 7. Update state
    display.front_buffer = Some(bo);
    display.current_fb = Some(fb);
    
    Ok(())
}
```

**Implementation Tasks**:
- [ ] Implement `present_to_display()` in DrmCanvas
- [ ] Integrate into `present()` method
- [ ] Handle page flip errors (EBUSY, EINVAL)
- [ ] Add buffer state tracking

---

#### Step 5: VSync/VBlank Handling
```rust
fn wait_for_vblank(card: &Card) -> Result<(), DrmCanvasError> {
    use std::os::unix::io::AsRawFd;
    use drm::control::Event;
    
    let fd = card.as_raw_fd();
    
    // Poll for DRM events
    let mut fds = [libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    }];
    
    // Wait up to 1 second for vblank
    let ret = unsafe { libc::poll(fds.as_mut_ptr(), 1, 1000) };
    
    if ret < 0 {
        return Err(DrmCanvasError::PageFlipError("Poll failed".into()));
    }
    
    if ret == 0 {
        return Err(DrmCanvasError::PageFlipError("VBlank timeout".into()));
    }
    
    // Read and process DRM event
    let events = card.receive_events()?;
    for event in events {
        match event {
            Event::PageFlip(_) => return Ok(()),
            _ => continue,
        }
    }
    
    Err(DrmCanvasError::PageFlipError("No page flip event".into()))
}
```

**Implementation Tasks**:
- [ ] Implement `wait_for_vblank()` with event polling
- [ ] Add timeout handling
- [ ] Parse DRM events correctly
- [ ] Add async variant for Phase 4

---

### Phase 4: Optimization 🔮 **FUTURE**

**Goal**: Improve performance and features

#### Planned Improvements:

1. **DMA-BUF Zero-Copy Path**
   - Export GBM buffer as DMA-BUF fd
   - Import into wgpu as external texture
   - Render directly to display buffer
   - Eliminates CPU copy overhead

2. **Async VSync Handling**
   - Non-blocking page flip
   - Event loop integration
   - Better frame pacing
   - Reduced latency

3. **Triple Buffering**
   - Maintain 3 buffers (front, back, queued)
   - Prevent blocking on buffer acquisition
   - Smoother frame rate
   - Better GPU utilization

4. **Format Optimization**
   - Direct XRGB8888 rendering in wgpu
   - Avoid format conversion
   - Potentially use GPU for conversion

5. **Multi-Display Support**
   - Handle multiple connectors
   - Independent rendering per display
   - Clone or extended modes

---

### Phase 5: Polish 🔮 **FUTURE**

**Goal**: Production-ready features

1. **Error Recovery**
   - Handle display disconnect
   - Recover from driver errors
   - Fallback to offscreen mode

2. **Dynamic Mode Changes**
   - Support resolution changes
   - Handle display hotplug
   - Mode preference API

3. **Performance Metrics**
   - Frame timing
   - Flip statistics
   - Buffer usage monitoring

4. **Documentation**
   - API documentation
   - Architecture guide
   - Performance tuning guide
   - Platform-specific notes

---

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_format_conversion() {
        // Test BGRA -> XRGB conversion
    }
    
    #[test]
    fn test_framebuffer_cache() {
        // Test FB creation and caching
    }
}
```

### Integration Tests
- `examples/drm_test.rs` - Combined Phase 1 & 2 validation
- `examples/drm_cube.rs` - Basic 3D rendering test
- Future: Color bars, animated scenes, stress tests

### Hardware Testing Matrix

| Platform | GPU Driver | Status | Notes |
|----------|------------|--------|-------|
| Raspberry Pi 4 | vc4/v3d | ✅ Phase 1&2 | Primary target |
| Raspberry Pi 5 | vc4/v3d | 🔄 Untested | Should work |
| x86 Linux (Intel) | i915 | 🔄 Untested | Should work |
| x86 Linux (AMD) | amdgpu | 🔄 Untested | Should work |
| x86 Linux (NVIDIA) | nouveau | ⚠️ Limited | Limited GBM support |
| VM (virtio-gpu) | virtio | 🔄 Untested | For CI/testing |

---

## Technical Challenges

### Challenge 1: Format Conversion
**Problem**: wgpu uses BGRA8Unorm, displays typically want XRGB8888
**Solution**: CPU-based conversion during copy (Phase 3), GPU conversion later (Phase 4)

### Challenge 2: Buffer Synchronization
**Problem**: Prevent rendering to buffer being scanned out
**Solution**: GBM buffer locking + proper state tracking

### Challenge 3: Permission Requirements
**Problem**: set_crtc requires DRM master (root or active VT)
**Solution**: Document requirements, provide helpful error messages

### Challenge 4: Driver Variations
**Problem**: Different GPU drivers have different capabilities
**Solution**: Format probing, graceful degradation, extensive testing

### Challenge 5: wgpu + GBM Integration
**Problem**: wgpu doesn't natively support GBM surfaces
**Solution**: Phase 2 uses offscreen + copy; Phase 4 adds DMA-BUF path

---

## API Design

### Public API
```rust
// Existing offscreen mode (unchanged)
let canvas = DrmCanvas::new(device_path, width, height).await?;

// New display output mode
let canvas = DrmCanvas::new_with_display(device_path).await?;

// Check mode
if canvas.is_display_mode() {
    println!("Rendering to display");
}

// Present (works for both modes)
canvas.present()?;
```

### High-Level API (DRMWindow)
```rust
// Offscreen rendering (existing)
let window = DRMWindow::new(device_path, width, height).await?;

// Display output (future enhancement)
let window = DRMWindow::new_with_display(device_path).await?;
```

---

## Error Handling Strategy

### Error Types
- `DrmError` - DRM API failures
- `GbmError` - GBM operations
- `ModesetError` - Display configuration
- `PageFlipError` - Page flip failures
- `IoError` - File operations

### Error Propagation
```rust
pub fn present(&mut self) -> Result<(), DrmCanvasError> {
    match &mut self.mode {
        RenderMode::Display(display) => {
            self.present_to_display(display)
                .map_err(|e| {
                    log::error!("Display output failed: {}", e);
                    // Could fallback to offscreen here
                    e
                })
        }
        _ => Ok(())
    }
}
```

---

## Dependencies

### Required Crates
```toml
[dependencies]
drm = { version = "0.14.1", optional = true }
gbm = { version = "0.18.0", optional = true }
wgpu = "27"

[features]
drm = ["dep:drm", "dep:gbm"]
```

### System Requirements
- Linux kernel 4.0+ with KMS
- GPU driver with DRM/KMS support
- GBM library (libgbm-dev)
- For Raspberry Pi: vc4-kms-v3d overlay enabled

---

## Performance Considerations

### Current Approach (Phase 3)
- **Copy overhead**: ~2-5ms for 1080p frame
- **Format conversion**: Minimal CPU usage
- **Frame rate**: Limited by copy + vsync

### Optimized Approach (Phase 4)
- **DMA-BUF**: Zero-copy, direct GPU rendering
- **Triple buffering**: Never block on buffer
- **Expected**: Full refresh rate, minimal CPU

### Benchmarks (Target)
| Resolution | Phase 3 | Phase 4 |
|------------|---------|---------|
| 1920x1080  | ~30ms   | ~16ms   |
| 1280x720   | ~15ms   | ~8ms    |
| 640x480    | ~5ms    | ~3ms    |

---

## Code Organization

```
src/window/drm/
├── mod.rs                    # Public exports
├── card.rs                   # DRM device wrapper
├── drm_canvas.rs            # Main implementation ⭐
│   ├── Error types
│   ├── Helper structs
│   ├── DrmCanvas impl
│   │   ├── new() (offscreen)
│   │   ├── new_with_display() ⭐ Phase 2
│   │   ├── present() ⭐ Phase 3
│   │   └── Query functions
│   └── DrmDisplayState impl
├── drm_canvas_wrapper.rs    # Canvas API compatibility
└── drm_window.rs            # High-level window API

examples/
├── drm_test.rs              # Combined Phase 1+2 test ⭐
├── drm_cube.rs              # 3D rendering example
└── (phase 3 examples...)
```

---

## Next Steps

### Immediate (Phase 3)
1. ✅ Implement `create_framebuffer()` helper
2. ✅ Implement `set_initial_mode()`
3. ✅ Implement `copy_wgpu_to_gbm()`
4. ✅ Implement `present_to_display()`
5. ✅ Implement `wait_for_vblank()`
6. ✅ Test on Raspberry Pi 4
7. ✅ Create Phase 3 test example

### Short-term (Phase 4)
1. Profile copy performance
2. Research DMA-BUF integration
3. Implement async vsync
4. Add triple buffering

### Long-term (Phase 5)
1. Production hardening
2. Multi-display support
3. Hot-plug handling
4. Documentation

---

## Resources

### Documentation
- [DRM KMS Documentation](https://www.kernel.org/doc/html/latest/gpu/drm-kms.html)
- [GBM API Reference](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/gbm/main/gbm.h)
- [wgpu Documentation](https://wgpu.rs/)

### Examples
- [drm-rs examples](https://github.com/Smithay/drm-rs/tree/master/examples)
- [gbm-rs examples](https://github.com/Smithay/gbm-rs/tree/master/examples)

### Related Projects
- [Smithay](https://github.com/Smithay/smithay) - Wayland compositor
- [winit](https://github.com/rust-windowing/winit) - Window handling
- [glutin](https://github.com/rust-windowing/glutin) - OpenGL context

---

## Changelog

### 2024-XX-XX - Phase 2 Complete
- Implemented GBM integration
- Added `new_with_display()` constructor
- Created unified test (`examples/drm_test.rs`)
- Validated on Raspberry Pi 4

### 2024-XX-XX - Phase 1 Complete
- Implemented display resource discovery
- Added error types and helper structs
- Created Phase 1 test
- Validated on Raspberry Pi 4

### 2024-XX-XX - Project Start
- Initial planning
- Architecture design
- Dependency evaluation

---

## Contributors

- Primary: [Your Name]
- Testing: Raspberry Pi 4 validation
- Review: [Future contributors]

---

## License

Same as kiss3d: BSD-3-Clause