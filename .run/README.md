# IntelliJ/CLion Run Configurations

This directory contains run configurations for all Kiss3d examples and common development tasks.

## Generated Configurations

### Example Configurations (72 total)

For each of the 37 examples, two run configurations are provided:

1. **Native** - `<example>.run.xml`
   - Runs the example on the native platform
   - Command: `cargo run --package kiss3d --example <name>`
   - Useful for quick testing and debugging

2. **Web (WASM)** - `<example> (web).run.xml`
   - Builds the example for WebAssembly
   - Command: `cargo run --package kiss3d --example <name> --target wasm32-unknown-unknown`
   - Produces `.wasm` files in `target/wasm32-unknown-unknown/`

### Examples List

- add_remove
- camera
- cube
- custom_material
- custom_mesh
- custom_mesh_shared
- decomp
- depth_of_field
- event
- group
- instancing2d
- instancing3d
- lines
- mouse_events
- multi_windows
- obj
- persistent_point_cloud
- planar_lines
- points
- post_processing
- primitives
- primitives2d
- primitives_scale
- procedural
- quad
- rectangle
- scene_cycler
- screenshot
- stereo
- text
- texturing
- texturing_mipmaps
- ui
- window
- wireframe

### Utility Configurations

- **check.run.xml** - Run `cargo check` on the project
- **check (wasm).run.xml** - Run `cargo check` for WASM target
- **clippy.run.xml** - Run Clippy linter
- **test.run.xml** - Run all tests

## Usage

### In IntelliJ/CLion

1. Open the project in IntelliJ IDEA or CLion
2. The run configurations will automatically appear in the run configuration dropdown (top-right)
3. Select any configuration and click Run (▶️) or Debug (🐛)

### Running Examples

#### Native
- Select `<example>` from the dropdown
- Click Run to execute the example
- A window will open showing the 3D scene

#### WASM
- Select `<example> (web)` from the dropdown
- Click Run to run for WebAssembly
- The compiled `.wasm` file will be in `target/wasm32-unknown-unknown/release/examples/`
- Use a web server to serve the WASM file (see `examples/wasm/` for web setup)

## Configuration Format

Each configuration file follows the IntelliJ Cargo Command Run Configuration format:

```xml
<component name="ProjectRunConfigurationManager">
  <configuration default="false" name="<name>" type="CargoCommandRunConfiguration" factoryName="Cargo Command">
    <option name="command" value="<cargo command>" />
    <option name="workingDirectory" value="file://$PROJECT_DIR$" />
    <!-- ... additional options ... -->
  </configuration>
</component>
```

## Regenerating Configurations

If you need to regenerate these configurations (e.g., after adding new examples), you can delete all `.run.xml` files and run the generation script again, or manually create new configuration files following the pattern above.

## Version Control

These files are tracked in git so all team members have the same run configurations available.
