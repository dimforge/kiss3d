//! Data structure of a scene node.

use crate::camera::Camera3d;
use crate::color::Color;
use crate::light::LightCollection;
use crate::resource::vertex_index::VertexIndex;
use crate::resource::{
    AllocationType, BufferType, GPUVec, GpuData, GpuMesh3d, Material3d, RenderContext, Texture,
    TextureManager,
};
use glamx::{Mat3, Pose3, Vec2, Vec3};
use std::any::Any;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

/// Rendering properties and state for a scene object.
///
/// Contains material, texture, color, and rendering settings for a 3D object.
/// This data is used by the rendering pipeline to determine how the object should be drawn.
pub struct ObjectData3d {
    material: Rc<RefCell<Box<dyn Material3d + 'static>>>,
    texture: Arc<Texture>,
    color: Color,
    lines_color: Option<Color>,
    points_color: Option<Color>,
    wlines: f32,
    wpoints: f32,
    lines_use_perspective: bool,
    points_use_perspective: bool,
    draw_surface: bool,
    cull: bool,
    user_data: Box<dyn Any + 'static>,
    // PBR material properties
    metallic: f32,
    roughness: f32,
    emissive: Color,
    // PBR texture maps
    normal_map: Option<Arc<Texture>>,
    metallic_roughness_map: Option<Arc<Texture>>,
    ao_map: Option<Arc<Texture>>,
    emissive_map: Option<Arc<Texture>>,
}

impl ObjectData3d {
    /// Returns a reference to this object's texture.
    ///
    /// # Returns
    /// A reference-counted texture
    #[inline]
    pub fn texture(&self) -> &Arc<Texture> {
        &self.texture
    }

    /// Returns the base color of this object.
    ///
    /// # Returns
    /// RGBA color with components in range [0.0, 1.0]
    #[inline]
    pub fn color(&self) -> Color {
        self.color
    }

    /// Returns the line width used for wireframe rendering.
    ///
    /// # Returns
    /// Line width in pixels
    #[inline]
    pub fn lines_width(&self) -> f32 {
        self.wlines
    }

    /// Returns the color used for wireframe line rendering.
    ///
    /// # Returns
    /// `Some(color)` if a custom line color is set, `None` to use the object's base color
    #[inline]
    pub fn lines_color(&self) -> Option<Color> {
        self.lines_color
    }

    /// Returns the point size used for point cloud rendering.
    ///
    /// # Returns
    /// Point size in pixels
    #[inline]
    pub fn points_size(&self) -> f32 {
        self.wpoints
    }

    /// Returns the color used for point rendering.
    ///
    /// # Returns
    /// `Some(color)` if a custom point color is set, `None` to use the object's base color
    #[inline]
    pub fn points_color(&self) -> Option<Color> {
        self.points_color
    }

    /// Checks if wireframe lines use perspective projection.
    ///
    /// # Returns
    /// `true` if wireframe lines scale with distance (perspective), `false` for constant screen-space width
    #[inline]
    pub fn lines_use_perspective(&self) -> bool {
        self.lines_use_perspective
    }

    /// Checks if points use perspective projection.
    ///
    /// # Returns
    /// `true` if points scale with distance (perspective), `false` for constant screen-space size
    #[inline]
    pub fn points_use_perspective(&self) -> bool {
        self.points_use_perspective
    }

    /// Checks if surface rendering is enabled for this object.
    ///
    /// # Returns
    /// `true` if surfaces are rendered, `false` if only wireframe/points are rendered
    #[inline]
    pub fn surface_rendering_active(&self) -> bool {
        self.draw_surface
    }

    /// Checks if backface culling is enabled for this object.
    ///
    /// # Returns
    /// `true` if backface culling is enabled
    #[inline]
    pub fn backface_culling_enabled(&self) -> bool {
        self.cull
    }

    /// Returns a reference to user-defined data attached to this object.
    ///
    /// Use the `Any` trait's downcasting methods to recover the actual data type.
    ///
    /// # Returns
    /// A reference to the user data as `&dyn Any`
    #[inline]
    pub fn user_data(&self) -> &dyn Any {
        &*self.user_data
    }

    /// Returns the metallic factor of this object.
    ///
    /// # Returns
    /// Metallic factor in range [0.0, 1.0] where 0.0 is dielectric and 1.0 is metal
    #[inline]
    pub fn metallic(&self) -> f32 {
        self.metallic
    }

    /// Returns the roughness factor of this object.
    ///
    /// # Returns
    /// Roughness factor in range [0.0, 1.0] where 0.0 is smooth and 1.0 is rough
    #[inline]
    pub fn roughness(&self) -> f32 {
        self.roughness
    }

    /// Returns the emissive color of this object.
    ///
    /// # Returns
    /// RGBA emissive color with components typically in range [0.0, 1.0] or higher for HDR
    #[inline]
    pub fn emissive(&self) -> Color {
        self.emissive
    }

    /// Returns a reference to this object's normal map texture.
    ///
    /// # Returns
    /// `Some` if a normal map is set, `None` otherwise
    #[inline]
    pub fn normal_map(&self) -> Option<&Arc<Texture>> {
        self.normal_map.as_ref()
    }

    /// Returns a reference to this object's metallic-roughness map texture.
    ///
    /// The texture follows glTF convention: B channel = metallic, G channel = roughness.
    ///
    /// # Returns
    /// `Some` if a metallic-roughness map is set, `None` otherwise
    #[inline]
    pub fn metallic_roughness_map(&self) -> Option<&Arc<Texture>> {
        self.metallic_roughness_map.as_ref()
    }

    /// Returns a reference to this object's ambient occlusion map texture.
    ///
    /// # Returns
    /// `Some` if an AO map is set, `None` otherwise
    #[inline]
    pub fn ao_map(&self) -> Option<&Arc<Texture>> {
        self.ao_map.as_ref()
    }

    /// Returns a reference to this object's emissive map texture.
    ///
    /// # Returns
    /// `Some` if an emissive map is set, `None` otherwise
    #[inline]
    pub fn emissive_map(&self) -> Option<&Arc<Texture>> {
        self.emissive_map.as_ref()
    }
}

/// Data for a single instance in instanced rendering.
///
/// When rendering multiple copies of the same mesh with different transformations
/// and colors (instancing), each instance is defined by this data.
///
/// # Example
/// ```no_run
/// # use kiss3d::scene::InstanceData3d;
/// # use kiss3d::color::{Color, RED, LIME, YELLOW};
/// # use glamx::{Vec3, Mat3};
/// let instance = InstanceData3d {
///     position: Vec3::new(1.0, 0.0, 0.0),
///     deformation: Mat3::IDENTITY,
///     color: RED,
///     lines_color: Some(LIME),  // Green wireframe
///     lines_width: Some(2.0),  // 2px wireframe
///     points_color: Some(YELLOW),  // Yellow points
///     points_size: Some(5.0),  // 5px points
/// };
/// ```
pub struct InstanceData3d {
    /// The position offset for this instance.
    pub position: Vec3,
    /// The 3x3 deformation matrix (scale, rotation, shear) for this instance.
    pub deformation: Mat3,
    /// The RGBA color for this instance.
    pub color: Color,
    /// The RGBA wireframe color for this instance. None = use object's wireframe color.
    pub lines_color: Option<Color>,
    /// The wireframe line width in pixels for this instance. None = use object's wireframe width.
    pub lines_width: Option<f32>,
    /// The RGBA point color for this instance. None = use object's point color.
    pub points_color: Option<Color>,
    /// The point size in pixels for this instance. None = use object's point size.
    pub points_size: Option<f32>,
}

impl Default for InstanceData3d {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            deformation: Mat3::IDENTITY,
            color: crate::color::WHITE,
            lines_color: None,  // Use object's wireframe color
            lines_width: None,  // Use object's wireframe width
            points_color: None, // Use object's point color
            points_size: None,  // Use object's point size
        }
    }
}

/// Sentinel value for lines_width indicating "use object's value".
pub const LINES_WIDTH_USE_OBJECT: f32 = -1.0;
/// Sentinel value for lines_color indicating "use object's value" (alpha = 0).
pub const LINES_COLOR_USE_OBJECT: Color = Color::new(0.0, 0.0, 0.0, 0.0);
/// Sentinel value for points_size indicating "use object's value".
pub const POINTS_SIZE_USE_OBJECT: f32 = -1.0;
/// Sentinel value for points_color indicating "use object's value" (alpha = 0).
pub const POINTS_COLOR_USE_OBJECT: Color = Color::new(0.0, 0.0, 0.0, 0.0);

/// GPU buffer for instanced rendering data.
///
/// Contains GPU-allocated buffers for positions, deformations, colors,
/// wireframe settings, and point settings of all instances to be rendered.
pub struct InstancesBuffer3d {
    /// GPU buffer of instance positions.
    pub positions: GPUVec<Vec3>,
    /// GPU buffer of instance deformation matrices (stored as 3 column vectors).
    pub deformations: GPUVec<Vec3>,
    /// GPU buffer of instance colors.
    pub colors: GPUVec<[f32; 4]>,
    /// GPU buffer of instance wireframe colors. Alpha = 0 means use object's color.
    pub lines_colors: GPUVec<[f32; 4]>,
    /// GPU buffer of instance wireframe line widths. Negative means use object's width.
    pub lines_widths: GPUVec<f32>,
    /// GPU buffer of instance point colors. Alpha = 0 means use object's color.
    pub points_colors: GPUVec<[f32; 4]>,
    /// GPU buffer of instance point sizes. Negative means use object's size.
    pub points_sizes: GPUVec<f32>,
}

/// Helper function to convert Color to [f32; 4] for GPU buffers.
#[inline]
pub(crate) fn color_to_array(color: Color) -> [f32; 4] {
    [color.r, color.g, color.b, color.a]
}

impl Default for InstancesBuffer3d {
    fn default() -> Self {
        InstancesBuffer3d {
            positions: GPUVec::new(
                vec![Vec3::ZERO],
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            deformations: GPUVec::new(
                vec![Vec3::X, Vec3::Y, Vec3::Z],
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            colors: GPUVec::new(
                vec![[1.0; 4]],
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            lines_colors: GPUVec::new(
                vec![color_to_array(LINES_COLOR_USE_OBJECT)], // Use object's wireframe color by default
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            lines_widths: GPUVec::new(
                vec![LINES_WIDTH_USE_OBJECT], // Use object's wireframe width by default
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            points_colors: GPUVec::new(
                vec![color_to_array(POINTS_COLOR_USE_OBJECT)], // Use object's point color by default
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            points_sizes: GPUVec::new(
                vec![POINTS_SIZE_USE_OBJECT], // Use object's point size by default
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
        }
    }
}

impl InstancesBuffer3d {
    /// Checks if there are no instances.
    ///
    /// # Returns
    /// `true` if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of instances in the buffer.
    ///
    /// # Returns
    /// The number of instances
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Checks if any instance has a specific wireframe width set (not using object's default).
    ///
    /// # Returns
    /// `true` if at least one instance has a specific wireframe width (>= 0)
    pub fn any_instance_has_wireframe(&self) -> bool {
        if let Some(widths) = self.lines_widths.data() {
            widths.iter().any(|&w| w >= 0.0)
        } else {
            false
        }
    }

    /// Checks if all instances use the object's wireframe width (all have sentinel value).
    ///
    /// # Returns
    /// `true` if all instances use object's wireframe width
    pub fn all_use_object_wireframe(&self) -> bool {
        if let Some(widths) = self.lines_widths.data() {
            widths.iter().all(|&w| w < 0.0)
        } else {
            true
        }
    }
}

/// A renderable 3D object in the scene.
///
/// `Object` combines a mesh with rendering properties (material, texture, color).
/// It's the primary interface for manipulating an object's appearance and geometry.
pub struct Object3d {
    // TODO: should Mesh and Object be merged?
    // (thus removing the need of ObjectData at all.)
    data: ObjectData3d,
    instances: Rc<RefCell<InstancesBuffer3d>>,
    mesh: Rc<RefCell<GpuMesh3d>>,
    /// Per-object GPU data for the material (uniform buffers, etc.)
    gpu_data: Box<dyn GpuData>,
}

impl Object3d {
    #[doc(hidden)]
    pub fn new(
        mesh: Rc<RefCell<GpuMesh3d>>,
        color: Color,
        texture: Arc<Texture>,
        material: Rc<RefCell<Box<dyn Material3d + 'static>>>,
    ) -> Object3d {
        // Create per-object GPU data from the material
        let gpu_data = material.borrow().create_gpu_data();

        let user_data = ();
        let data = ObjectData3d {
            color,
            lines_color: None,
            points_color: None,
            texture,
            wlines: 0.0,
            wpoints: 0.0,
            lines_use_perspective: true,
            points_use_perspective: true,
            draw_surface: true,
            cull: true,
            material,
            user_data: Box::new(user_data),
            // PBR defaults (backward compatible with Blinn-Phong appearance)
            metallic: 0.0,
            roughness: 0.5,
            emissive: crate::color::BLACK,
            normal_map: None,
            metallic_roughness_map: None,
            ao_map: None,
            emissive_map: None,
        };
        let instances = Rc::new(RefCell::new(InstancesBuffer3d::default()));

        Object3d {
            data,
            instances,
            mesh,
            gpu_data,
        }
    }

    #[doc(hidden)]
    pub fn prepare(
        &mut self,
        transform: Pose3,
        scale: Vec3,
        pass: usize,
        camera: &mut dyn Camera3d,
        lights: &LightCollection,
        viewport_width: u32,
        viewport_height: u32,
    ) {
        self.data.material.borrow_mut().prepare(
            pass,
            transform,
            scale,
            camera,
            lights,
            &self.data,
            &mut *self.gpu_data,
            viewport_width,
            viewport_height,
        );
    }

    #[doc(hidden)]
    pub fn render(
        &mut self,
        transform: Pose3,
        scale: Vec3,
        pass: usize,
        camera: &mut dyn Camera3d,
        lights: &LightCollection,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext,
    ) {
        self.data.material.borrow_mut().render(
            pass,
            transform,
            scale,
            camera,
            lights,
            &self.data,
            &mut self.mesh.borrow_mut(),
            &mut self.instances.borrow_mut(),
            &mut *self.gpu_data,
            render_pass,
            context,
        );
    }

    /// Gets the data of this object.
    #[inline]
    pub fn data(&self) -> &ObjectData3d {
        &self.data
    }

    /// Gets the data of this object.
    #[inline]
    pub fn data_mut(&mut self) -> &mut ObjectData3d {
        &mut self.data
    }

    /// Gets the instances of this object.
    #[inline]
    pub fn instances(&self) -> &Rc<RefCell<InstancesBuffer3d>> {
        &self.instances
    }

    pub fn set_instances(&mut self, instances: &[InstanceData3d]) {
        let mut pos_data: Vec<_> = self
            .instances
            .borrow_mut()
            .positions
            .data_mut()
            .take()
            .unwrap_or_default();
        let mut col_data: Vec<_> = self
            .instances
            .borrow_mut()
            .colors
            .data_mut()
            .take()
            .unwrap_or_default();
        let mut def_data: Vec<_> = self
            .instances
            .borrow_mut()
            .deformations
            .data_mut()
            .take()
            .unwrap_or_default();
        let mut lines_col_data: Vec<_> = self
            .instances
            .borrow_mut()
            .lines_colors
            .data_mut()
            .take()
            .unwrap_or_default();
        let mut lines_width_data: Vec<_> = self
            .instances
            .borrow_mut()
            .lines_widths
            .data_mut()
            .take()
            .unwrap_or_default();
        let mut points_col_data: Vec<_> = self
            .instances
            .borrow_mut()
            .points_colors
            .data_mut()
            .take()
            .unwrap_or_default();
        let mut points_size_data: Vec<_> = self
            .instances
            .borrow_mut()
            .points_sizes
            .data_mut()
            .take()
            .unwrap_or_default();

        pos_data.clear();
        col_data.clear();
        def_data.clear();
        lines_col_data.clear();
        lines_width_data.clear();
        points_col_data.clear();
        points_size_data.clear();

        pos_data.extend(instances.iter().map(|i| i.position));
        col_data.extend(instances.iter().map(|i| color_to_array(i.color)));
        def_data.extend(instances.iter().flat_map(|i| {
            [
                i.deformation.x_axis,
                i.deformation.y_axis,
                i.deformation.z_axis,
            ]
        }));
        lines_col_data.extend(
            instances
                .iter()
                .map(|i| color_to_array(i.lines_color.unwrap_or(LINES_COLOR_USE_OBJECT))),
        );
        lines_width_data.extend(
            instances
                .iter()
                .map(|i| i.lines_width.unwrap_or(LINES_WIDTH_USE_OBJECT)),
        );
        points_col_data.extend(
            instances
                .iter()
                .map(|i| color_to_array(i.points_color.unwrap_or(POINTS_COLOR_USE_OBJECT))),
        );
        points_size_data.extend(
            instances
                .iter()
                .map(|i| i.points_size.unwrap_or(POINTS_SIZE_USE_OBJECT)),
        );

        *self.instances.borrow_mut().positions.data_mut() = Some(pos_data);
        *self.instances.borrow_mut().colors.data_mut() = Some(col_data);
        *self.instances.borrow_mut().deformations.data_mut() = Some(def_data);
        *self.instances.borrow_mut().lines_colors.data_mut() = Some(lines_col_data);
        *self.instances.borrow_mut().lines_widths.data_mut() = Some(lines_width_data);
        *self.instances.borrow_mut().points_colors.data_mut() = Some(points_col_data);
        *self.instances.borrow_mut().points_sizes.data_mut() = Some(points_size_data);
    }

    /// Enables or disables backface culling for this object.
    #[inline]
    pub fn enable_backface_culling(&mut self, active: bool) {
        self.data.cull = active;
    }

    /// Attaches user-defined data to this object.
    #[inline]
    pub fn set_user_data(&mut self, user_data: Box<dyn Any + 'static>) {
        self.data.user_data = user_data;
    }

    /// Gets the material of this object.
    #[inline]
    pub fn material(&self) -> Rc<RefCell<Box<dyn Material3d + 'static>>> {
        self.data.material.clone()
    }

    /// Sets the material of this object.
    #[inline]
    pub fn set_material(&mut self, material: Rc<RefCell<Box<dyn Material3d + 'static>>>) {
        // Create new GPU data for the new material
        self.gpu_data = material.borrow().create_gpu_data();
        self.data.material = material;
    }

    /// Sets the width of the lines drawn for this object.
    ///
    /// If `use_perspective` is true, the width is in world units and scales with distance.
    /// If `use_perspective` is false, the width is in screen pixels and stays constant.
    #[inline]
    pub fn set_lines_width(&mut self, width: f32, use_perspective: bool) {
        self.data.wlines = width;
        self.data.lines_use_perspective = use_perspective;
    }

    /// Returns the width of the lines drawn for this object.
    #[inline]
    pub fn lines_width(&self) -> f32 {
        self.data.wlines
    }

    /// Sets the color of the lines drawn for this object.
    #[inline]
    pub fn set_lines_color(&mut self, color: Option<Color>) {
        self.data.lines_color = color
    }

    /// Returns the color of the lines drawn for this object.
    #[inline]
    pub fn lines_color(&self) -> Option<Color> {
        self.data.lines_color
    }

    /// Sets the size of the points drawn for this object.
    ///
    /// If `use_perspective` is true, the size is in world units and scales with distance.
    /// If `use_perspective` is false, the size is in screen pixels and stays constant.
    #[inline]
    pub fn set_points_size(&mut self, size: f32, use_perspective: bool) {
        self.data.wpoints = size;
        self.data.points_use_perspective = use_perspective;
    }

    /// Returns the size of the points drawn for this object.
    #[inline]
    pub fn points_size(&self) -> f32 {
        self.data.wpoints
    }

    /// Sets the color of the points drawn for this object.
    #[inline]
    pub fn set_points_color(&mut self, color: Option<Color>) {
        self.data.points_color = color
    }

    /// Returns the color of the points drawn for this object.
    #[inline]
    pub fn points_color(&self) -> Option<Color> {
        self.data.points_color
    }

    /// Activate or deactivate the rendering of this object surface.
    #[inline]
    pub fn set_surface_rendering_activation(&mut self, active: bool) {
        self.data.draw_surface = active
    }

    /// Activate or deactivate the rendering of this object surface.
    #[inline]
    pub fn surface_rendering_activation(&self) -> bool {
        self.data.draw_surface
    }

    /// This object's mesh.
    #[inline]
    pub fn mesh(&self) -> &Rc<RefCell<GpuMesh3d>> {
        &self.mesh
    }

    /// Mutably access the object's vertices.
    #[inline(always)]
    pub fn modify_vertices<F: FnMut(&mut Vec<Vec3>)>(&mut self, f: &mut F) {
        let bmesh = self.mesh.borrow_mut();
        let _ = bmesh.coords().write().unwrap().data_mut().as_mut().map(f);
    }

    /// Access the object's vertices.
    #[inline(always)]
    pub fn read_vertices<F: FnMut(&[Vec3])>(&self, f: &mut F) {
        let bmesh = self.mesh.borrow();
        let _ = bmesh
            .coords()
            .read()
            .unwrap()
            .data()
            .as_ref()
            .map(|coords| f(&coords[..]));
    }

    /// Recomputes the normals of this object's mesh.
    #[inline]
    pub fn recompute_normals(&mut self) {
        self.mesh.borrow_mut().recompute_normals();
    }

    /// Mutably access the object's normals.
    #[inline(always)]
    pub fn modify_normals<F: FnMut(&mut Vec<Vec3>)>(&mut self, f: &mut F) {
        let bmesh = self.mesh.borrow_mut();
        let _ = bmesh.normals().write().unwrap().data_mut().as_mut().map(f);
    }

    /// Access the object's normals.
    #[inline(always)]
    pub fn read_normals<F: FnMut(&[Vec3])>(&self, f: &mut F) {
        let bmesh = self.mesh.borrow();
        let _ = bmesh
            .normals()
            .read()
            .unwrap()
            .data()
            .as_ref()
            .map(|normals| f(&normals[..]));
    }

    /// Mutably access the object's faces.
    #[inline(always)]
    pub fn modify_faces<F: FnMut(&mut Vec<[VertexIndex; 3]>)>(&mut self, f: &mut F) {
        let bmesh = self.mesh.borrow_mut();
        let _ = bmesh.faces().write().unwrap().data_mut().as_mut().map(f);
    }

    /// Access the object's faces.
    #[inline(always)]
    pub fn read_faces<F: FnMut(&[[VertexIndex; 3]])>(&self, f: &mut F) {
        let bmesh = self.mesh.borrow();
        let _ = bmesh
            .faces()
            .read()
            .unwrap()
            .data()
            .as_ref()
            .map(|faces| f(&faces[..]));
    }

    /// Mutably access the object's texture coordinates.
    #[inline(always)]
    pub fn modify_uvs<F: FnMut(&mut Vec<Vec2>)>(&mut self, f: &mut F) {
        let bmesh = self.mesh.borrow_mut();
        let _ = bmesh.uvs().write().unwrap().data_mut().as_mut().map(f);
    }

    /// Access the object's texture coordinates.
    #[inline(always)]
    pub fn read_uvs<F: FnMut(&[Vec2])>(&self, f: &mut F) {
        let bmesh = self.mesh.borrow();
        let _ = bmesh
            .uvs()
            .read()
            .unwrap()
            .data()
            .as_ref()
            .map(|uvs| f(&uvs[..]));
    }

    /// Sets the color of the object.
    ///
    /// Colors components must be on the range `[0.0, 1.0]`.
    #[inline]
    pub fn set_color(&mut self, color: Color) {
        self.data.color = color;
    }

    /// Sets the texture of the object.
    ///
    /// The texture is loaded from a file and registered by the global `TextureManager`.
    ///
    /// # Arguments
    ///   * `path` - relative path of the texture on the disk
    #[inline]
    pub fn set_texture_from_file(&mut self, path: &Path, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));

        self.set_texture(texture)
    }

    /// Sets the texture of the object.
    ///
    /// The texture must already have been registered as `name`.
    #[inline]
    pub fn set_texture_with_name(&mut self, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| {
            tm.get(name).unwrap_or_else(|| {
                panic!("Invalid attempt to use the unregistered texture: {}", name)
            })
        });

        self.set_texture(texture)
    }

    /// Sets the texture of the object.
    #[inline]
    pub fn set_texture(&mut self, texture: Arc<Texture>) {
        self.data.texture = texture
    }

    // === PBR Material Properties ===

    /// Sets the metallic factor of this object.
    ///
    /// # Arguments
    /// * `metallic` - Metallic factor clamped to [0.0, 1.0] where 0.0 is dielectric and 1.0 is metal
    #[inline]
    pub fn set_metallic(&mut self, metallic: f32) {
        self.data.metallic = metallic.clamp(0.0, 1.0);
    }

    /// Sets the roughness factor of this object.
    ///
    /// # Arguments
    /// * `roughness` - Roughness factor clamped to [0.0, 1.0] where 0.0 is smooth and 1.0 is rough
    #[inline]
    pub fn set_roughness(&mut self, roughness: f32) {
        self.data.roughness = roughness.clamp(0.0, 1.0);
    }

    /// Sets the emissive color of this object.
    ///
    /// Objects with emissive color appear to glow. Values above 1.0 can be used for HDR.
    ///
    /// # Arguments
    /// * `color` - RGBA emissive color
    #[inline]
    pub fn set_emissive(&mut self, color: Color) {
        self.data.emissive = color;
    }

    // === PBR Texture Maps ===

    /// Sets the normal map texture from a file.
    ///
    /// Normal maps add surface detail without additional geometry.
    ///
    /// # Arguments
    /// * `path` - Path to the normal map image file
    /// * `name` - Name to register the texture under
    #[inline]
    pub fn set_normal_map_from_file(&mut self, path: &Path, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));
        self.set_normal_map(texture);
    }

    /// Sets the normal map texture.
    #[inline]
    pub fn set_normal_map(&mut self, texture: Arc<Texture>) {
        self.data.normal_map = Some(texture);
    }

    /// Clears the normal map.
    #[inline]
    pub fn clear_normal_map(&mut self) {
        self.data.normal_map = None;
    }

    /// Sets the metallic-roughness map texture from a file.
    ///
    /// Follows glTF convention: B channel = metallic, G channel = roughness.
    ///
    /// # Arguments
    /// * `path` - Path to the metallic-roughness map image file
    /// * `name` - Name to register the texture under
    #[inline]
    pub fn set_metallic_roughness_map_from_file(&mut self, path: &Path, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));
        self.set_metallic_roughness_map(texture);
    }

    /// Sets the metallic-roughness map texture.
    ///
    /// Follows glTF convention: B channel = metallic, G channel = roughness.
    #[inline]
    pub fn set_metallic_roughness_map(&mut self, texture: Arc<Texture>) {
        self.data.metallic_roughness_map = Some(texture);
    }

    /// Clears the metallic-roughness map.
    #[inline]
    pub fn clear_metallic_roughness_map(&mut self) {
        self.data.metallic_roughness_map = None;
    }

    /// Sets the ambient occlusion map texture from a file.
    ///
    /// AO maps add subtle shadows in crevices and corners.
    ///
    /// # Arguments
    /// * `path` - Path to the AO map image file
    /// * `name` - Name to register the texture under
    #[inline]
    pub fn set_ao_map_from_file(&mut self, path: &Path, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));
        self.set_ao_map(texture);
    }

    /// Sets the ambient occlusion map texture.
    #[inline]
    pub fn set_ao_map(&mut self, texture: Arc<Texture>) {
        self.data.ao_map = Some(texture);
    }

    /// Clears the ambient occlusion map.
    #[inline]
    pub fn clear_ao_map(&mut self) {
        self.data.ao_map = None;
    }

    /// Sets the emissive map texture from a file.
    ///
    /// The emissive map is multiplied by the emissive color.
    ///
    /// # Arguments
    /// * `path` - Path to the emissive map image file
    /// * `name` - Name to register the texture under
    #[inline]
    pub fn set_emissive_map_from_file(&mut self, path: &Path, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));
        self.set_emissive_map(texture);
    }

    /// Sets the emissive map texture.
    #[inline]
    pub fn set_emissive_map(&mut self, texture: Arc<Texture>) {
        self.data.emissive_map = Some(texture);
    }

    /// Clears the emissive map.
    #[inline]
    pub fn clear_emissive_map(&mut self) {
        self.data.emissive_map = None;
    }
}
