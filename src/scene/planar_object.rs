//! Data structure of a scene node.

use crate::planar_camera::PlanarCamera;
use crate::resource::vertex_index::VertexIndex;
use crate::resource::{
    AllocationType, BufferType, GPUVec, GpuData, PlanarMaterial, PlanarMesh, PlanarRenderContext,
    Texture, TextureManager,
};
use na::{Isometry2, Matrix2, Point2, Point3, Vector2};
use std::any::Any;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

/// Set of data identifying a scene node.
pub struct PlanarObjectData {
    material: Rc<RefCell<Box<dyn PlanarMaterial + 'static>>>,
    texture: Arc<Texture>,
    color: Point3<f32>,
    lines_color: Option<Point3<f32>>,
    points_color: Option<Point3<f32>>,
    wlines: f32,
    wpoints: f32,
    lines_use_perspective: bool,
    points_use_perspective: bool,
    draw_surface: bool,
    cull: bool,
    user_data: Box<dyn Any + 'static>,
}

impl PlanarObjectData {
    /// The texture of this object.
    #[inline]
    pub fn texture(&self) -> &Arc<Texture> {
        &self.texture
    }

    /// The color of this object.
    #[inline]
    pub fn color(&self) -> &Point3<f32> {
        &self.color
    }

    /// The width of the lines draw for this object.
    #[inline]
    pub fn lines_width(&self) -> f32 {
        self.wlines
    }

    /// The color of the lines draw for this object.
    #[inline]
    pub fn lines_color(&self) -> Option<&Point3<f32>> {
        self.lines_color.as_ref()
    }

    /// The size of the points draw for this object.
    #[inline]
    pub fn points_size(&self) -> f32 {
        self.wpoints
    }

    /// The color of the points draw for this object.
    #[inline]
    pub fn points_color(&self) -> Option<&Point3<f32>> {
        self.points_color.as_ref()
    }

    /// Whether wireframe lines use perspective projection.
    #[inline]
    pub fn lines_use_perspective(&self) -> bool {
        self.lines_use_perspective
    }

    /// Whether points use perspective projection.
    #[inline]
    pub fn points_use_perspective(&self) -> bool {
        self.points_use_perspective
    }

    /// Whether this object has its surface rendered or not.
    #[inline]
    pub fn surface_rendering_active(&self) -> bool {
        self.draw_surface
    }

    /// Whether this object uses backface culling or not.
    #[inline]
    pub fn backface_culling_enabled(&self) -> bool {
        self.cull
    }

    /// An user-defined data.
    ///
    /// Use dynamic typing capabilities of the `Any` type to recover the actual data.
    #[inline]
    pub fn user_data(&self) -> &dyn Any {
        &*self.user_data
    }
}

/// Data for a single 2D instance when using instanced rendering.
///
/// # Example
/// ```no_run
/// # use kiss3d::scene::PlanarInstanceData;
/// # use nalgebra::{Point2, Matrix2};
/// let instance = PlanarInstanceData {
///     position: Point2::new(100.0, 50.0),
///     deformation: Matrix2::identity(),
///     color: [1.0, 0.0, 0.0, 1.0],  // Red
///     lines_color: Some([0.0, 1.0, 0.0, 1.0]),  // Green wireframe
///     lines_width: Some(2.0),  // 2px wireframe
///     points_color: Some([1.0, 1.0, 0.0, 1.0]),  // Yellow points
///     points_size: Some(5.0),  // 5px points
/// };
/// ```
pub struct PlanarInstanceData {
    /// The position offset for this instance.
    pub position: Point2<f32>,
    /// The 2x2 deformation matrix (scale, rotation, shear) for this instance.
    pub deformation: Matrix2<f32>,
    /// The RGBA color for this instance [r, g, b, a] in range [0.0, 1.0].
    pub color: [f32; 4],
    /// The RGBA wireframe color for this instance. None = use object's wireframe color.
    pub lines_color: Option<[f32; 4]>,
    /// The wireframe line width in pixels for this instance. None = use object's wireframe width.
    pub lines_width: Option<f32>,
    /// The RGBA point color for this instance. None = use object's point color.
    pub points_color: Option<[f32; 4]>,
    /// The point size in pixels for this instance. None = use object's point size.
    pub points_size: Option<f32>,
}

impl Default for PlanarInstanceData {
    fn default() -> Self {
        Self {
            position: Point2::origin(),
            deformation: Matrix2::identity(),
            color: [1.0; 4],
            lines_color: None,  // Use object's wireframe color
            lines_width: None,  // Use object's wireframe width
            points_color: None, // Use object's point color
            points_size: None,  // Use object's point size
        }
    }
}

/// Sentinel value for lines_width indicating "use object's value".
pub const PLANAR_LINES_WIDTH_USE_OBJECT: f32 = -1.0;
/// Sentinel value for lines_color indicating "use object's value" (alpha = 0).
pub const PLANAR_LINES_COLOR_USE_OBJECT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
/// Sentinel value for points_size indicating "use object's value".
pub const PLANAR_POINTS_SIZE_USE_OBJECT: f32 = -1.0;
/// Sentinel value for points_color indicating "use object's value" (alpha = 0).
pub const PLANAR_POINTS_COLOR_USE_OBJECT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

/// GPU buffer for 2D instanced rendering data.
///
/// Contains GPU-allocated buffers for positions, deformations, colors,
/// wireframe settings, and point settings of all 2D instances to be rendered.
pub struct PlanarInstancesBuffer {
    /// GPU buffer of instance positions.
    pub positions: GPUVec<Point2<f32>>,
    /// GPU buffer of instance deformation matrices (stored as 2 column vectors).
    pub deformations: GPUVec<Vector2<f32>>,
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

impl Default for PlanarInstancesBuffer {
    fn default() -> Self {
        PlanarInstancesBuffer {
            positions: GPUVec::new(
                vec![Point2::origin()],
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            deformations: GPUVec::new(
                vec![Vector2::x(), Vector2::y()],
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            colors: GPUVec::new(
                vec![[1.0; 4]],
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            lines_colors: GPUVec::new(
                vec![PLANAR_LINES_COLOR_USE_OBJECT], // Use object's wireframe color by default
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            lines_widths: GPUVec::new(
                vec![PLANAR_LINES_WIDTH_USE_OBJECT], // Use object's wireframe width by default
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            points_colors: GPUVec::new(
                vec![PLANAR_POINTS_COLOR_USE_OBJECT], // Use object's point color by default
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
            points_sizes: GPUVec::new(
                vec![PLANAR_POINTS_SIZE_USE_OBJECT], // Use object's point size by default
                BufferType::Array,
                AllocationType::StreamDraw,
            ),
        }
    }
}

impl PlanarInstancesBuffer {
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
}

/// A 2D object on the scene.
///
/// This is the only interface to manipulate the object position, color, vertices and texture.
pub struct PlanarObject {
    // FIXME: should PlanarMesh and PlanarObject be merged?
    // (thus removing the need of PlanarObjectData at all.)
    data: PlanarObjectData,
    instances: Rc<RefCell<PlanarInstancesBuffer>>,
    mesh: Rc<RefCell<PlanarMesh>>,
    /// Per-object GPU data for the material (uniform buffers, etc.)
    gpu_data: Box<dyn GpuData>,
}

impl PlanarObject {
    #[doc(hidden)]
    pub fn new(
        mesh: Rc<RefCell<PlanarMesh>>,
        r: f32,
        g: f32,
        b: f32,
        texture: Arc<Texture>,
        material: Rc<RefCell<Box<dyn PlanarMaterial + 'static>>>,
    ) -> PlanarObject {
        // Create per-object GPU data from the material
        let gpu_data = material.borrow().create_gpu_data();

        let user_data = ();
        let data = PlanarObjectData {
            color: Point3::new(r, g, b),
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
        };
        let instances = Rc::new(RefCell::new(PlanarInstancesBuffer::default()));

        PlanarObject {
            data,
            instances,
            mesh,
            gpu_data,
        }
    }

    #[doc(hidden)]
    pub fn render(
        &mut self,
        transform: &Isometry2<f32>,
        scale: &Vector2<f32>,
        camera: &mut dyn PlanarCamera,
        context: &mut PlanarRenderContext,
    ) {
        self.data.material.borrow_mut().render(
            transform,
            scale,
            camera,
            &self.data,
            &mut self.mesh.borrow_mut(),
            &mut self.instances.borrow_mut(),
            &mut *self.gpu_data,
            context,
        );
    }

    /// Gets the instances of this object.
    #[inline]
    pub fn instances(&self) -> &Rc<RefCell<PlanarInstancesBuffer>> {
        &self.instances
    }

    /// Sets the instances for this object.
    pub fn set_instances(&mut self, instances: &[PlanarInstanceData]) {
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
        col_data.extend(instances.iter().map(|i| i.color));
        def_data.extend(
            instances
                .iter()
                .flat_map(|i| i.deformation.column_iter().map(|c| c.into_owned())),
        );
        lines_col_data.extend(
            instances
                .iter()
                .map(|i| i.lines_color.unwrap_or(PLANAR_LINES_COLOR_USE_OBJECT)),
        );
        lines_width_data.extend(
            instances
                .iter()
                .map(|i| i.lines_width.unwrap_or(PLANAR_LINES_WIDTH_USE_OBJECT)),
        );
        points_col_data.extend(
            instances
                .iter()
                .map(|i| i.points_color.unwrap_or(PLANAR_POINTS_COLOR_USE_OBJECT)),
        );
        points_size_data.extend(
            instances
                .iter()
                .map(|i| i.points_size.unwrap_or(PLANAR_POINTS_SIZE_USE_OBJECT)),
        );

        *self.instances.borrow_mut().positions.data_mut() = Some(pos_data);
        *self.instances.borrow_mut().colors.data_mut() = Some(col_data);
        *self.instances.borrow_mut().deformations.data_mut() = Some(def_data);
        *self.instances.borrow_mut().lines_colors.data_mut() = Some(lines_col_data);
        *self.instances.borrow_mut().lines_widths.data_mut() = Some(lines_width_data);
        *self.instances.borrow_mut().points_colors.data_mut() = Some(points_col_data);
        *self.instances.borrow_mut().points_sizes.data_mut() = Some(points_size_data);
    }

    /// Gets the data of this object.
    #[inline]
    pub fn data(&self) -> &PlanarObjectData {
        &self.data
    }

    /// Gets the data of this object.
    #[inline]
    pub fn data_mut(&mut self) -> &mut PlanarObjectData {
        &mut self.data
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
    pub fn material(&self) -> Rc<RefCell<Box<dyn PlanarMaterial + 'static>>> {
        self.data.material.clone()
    }

    /// Sets the material of this object.
    #[inline]
    pub fn set_material(&mut self, material: Rc<RefCell<Box<dyn PlanarMaterial + 'static>>>) {
        // Create new GPU data for the new material
        self.gpu_data = material.borrow().create_gpu_data();
        self.data.material = material;
    }

    /// Sets the width of the lines drawn for this object.
    ///
    /// If `use_perspective` is true, the width is in world units and scales with camera zoom.
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
    pub fn set_lines_color(&mut self, color: Option<Point3<f32>>) {
        self.data.lines_color = color
    }

    /// Returns the color of the lines drawn for this object.
    #[inline]
    pub fn lines_color(&self) -> Option<&Point3<f32>> {
        self.data.lines_color()
    }

    /// Sets the size of the points drawn for this object.
    ///
    /// If `use_perspective` is true, the size is in world units and scales with camera zoom.
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
    pub fn set_points_color(&mut self, color: Option<Point3<f32>>) {
        self.data.points_color = color
    }

    /// Returns the color of the points drawn for this object.
    #[inline]
    pub fn points_color(&self) -> Option<&Point3<f32>> {
        self.data.points_color()
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
    pub fn mesh(&self) -> &Rc<RefCell<PlanarMesh>> {
        &self.mesh
    }

    /// Mutably access the object's vertices.
    #[inline(always)]
    pub fn modify_vertices<F: FnMut(&mut Vec<Point2<f32>>)>(&mut self, f: &mut F) {
        let bmesh = self.mesh.borrow_mut();
        let _ = bmesh.coords().write().unwrap().data_mut().as_mut().map(f);
    }

    /// Access the object's vertices.
    #[inline(always)]
    pub fn read_vertices<F: FnMut(&[Point2<f32>])>(&self, f: &mut F) {
        let bmesh = self.mesh.borrow();
        let _ = bmesh
            .coords()
            .read()
            .unwrap()
            .data()
            .as_ref()
            .map(|coords| f(&coords[..]));
    }

    /// Mutably access the object's faces.
    #[inline(always)]
    pub fn modify_faces<F: FnMut(&mut Vec<Point3<VertexIndex>>)>(&mut self, f: &mut F) {
        let bmesh = self.mesh.borrow_mut();
        let _ = bmesh.faces().write().unwrap().data_mut().as_mut().map(f);
    }

    /// Access the object's faces.
    #[inline(always)]
    pub fn read_faces<F: FnMut(&[Point3<VertexIndex>])>(&self, f: &mut F) {
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
    pub fn modify_uvs<F: FnMut(&mut Vec<Point2<f32>>)>(&mut self, f: &mut F) {
        let bmesh = self.mesh.borrow_mut();
        let _ = bmesh.uvs().write().unwrap().data_mut().as_mut().map(f);
    }

    /// Access the object's texture coordinates.
    #[inline(always)]
    pub fn read_uvs<F: FnMut(&[Point2<f32>])>(&self, f: &mut F) {
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
    pub fn set_color(&mut self, r: f32, g: f32, b: f32) {
        self.data.color.x = r;
        self.data.color.y = g;
        self.data.color.z = b;
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
}
