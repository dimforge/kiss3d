# DRM Test - Quick Start

## One-Line Test Command

```bash
cargo run --example drm_test --features drm
```

## With sudo (if needed)

```bash
cargo build --example drm_test --features drm && sudo ./target/debug/examples/drm_test
```

## What This Tests

**Phase 1: Display Discovery**
- ✅ DRM device access with fallback
- ✅ Display detection
- ✅ Resolution enumeration
- ✅ CRTC and encoder discovery

**Phase 2: GBM Integration**
- ✅ GBM device initialization
- ✅ Format support validation
- ✅ GBM surface creation
- ✅ Buffer operations

## Expected Result

```
╔══════════════════════════════════════════════════════════════════╗
║                      🎉 ALL TESTS PASSED 🎉                      ║
╚══════════════════════════════════════════════════════════════════╝
```

## If It Fails

1. **Permission denied**: Add yourself to video group or use sudo
   ```bash
   sudo usermod -a -G video $USER
   ```

2. **Device not found**: Enable KMS in `/boot/firmware/config.txt`
   ```
   dtoverlay=vc4-kms-v3d
   ```

3. **No display found**: Check HDMI connection

## Full Documentation

See `TESTING_DRM_PHASE1.md` for complete testing guide.

## Ready for Phase 3?

Once this test passes, you're ready to implement Phase 3 (display output with modesetting and page flipping).