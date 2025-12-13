use na::{self, Isometry2, Point2, Point3, Translation2, UnitComplex, Vector2};

use crate::planar_camera::PlanarCamera;
use crate::prelude::PlanarInstanceData;
use crate::resource::vertex_index::VertexIndex;
use crate::resource::{
    PlanarMaterial, PlanarMaterialManager, PlanarMesh, PlanarMeshManager, PlanarRenderContext,
    Texture, TextureManager,
};
use crate::scene::PlanarObject;
use std::cell::{Ref, RefCell, RefMut};
use std::f32;
use std::mem;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

// XXX: once something like `fn foo(self: Rc<RefCell<PlanarSceneNode>>)` is allowed, this extra struct
// will not be needed any more.
/// The data contained by a `PlanarSceneNode`.
pub struct PlanarSceneNodeData {
    local_scale: Vector2<f32>,
    local_transform: Isometry2<f32>,
    world_scale: Vector2<f32>,
    world_transform: Isometry2<f32>,
    visible: bool,
    up_to_date: bool,
    children: Vec<PlanarSceneNode>,
    object: Option<PlanarObject>,
    // FIXME: use Weak pointers instead of the raw pointer.
    parent: Option<*const RefCell<PlanarSceneNodeData>>,
}

/// A node of the scene graph.
///
/// This may represent a group of other nodes, and/or contain an object that can be rendered.
#[derive(Clone)]
pub struct PlanarSceneNode {
    data: Rc<RefCell<PlanarSceneNodeData>>,
}

impl PlanarSceneNodeData {
    // XXX: Because `node.borrow_mut().parent = Some(self.data.downgrade())`
    // causes a weird compiler error:
    //
    // ```
    // error: mismatched types: expected `&std::cell::RefCell<scene::scene_node::PlanarSceneNodeData>`
    // but found
    // `std::option::Option<std::rc::Weak<std::cell::RefCell<scene::scene_node::PlanarSceneNodeData>>>`
    // (expe cted &-ptr but found enum std::option::Option)
    // ```
    fn set_parent(&mut self, parent: *const RefCell<PlanarSceneNodeData>) {
        self.parent = Some(parent);
    }

    // XXX: this exists because of a similar bug as `set_parent`.
    fn remove_from_parent(&mut self, to_remove: &PlanarSceneNode) {
        let _ = self.parent.as_ref().map(|p| unsafe {
            let mut bp = (**p).borrow_mut();
            bp.remove(to_remove)
        });
    }

    fn remove(&mut self, o: &PlanarSceneNode) {
        if let Some(i) = self
            .children
            .iter()
            .rposition(|e| std::ptr::eq(&*o.data, &*e.data))
        {
            let _ = self.children.swap_remove(i);
        }
    }

    /// Whether this node contains an `PlanarObject`.
    #[inline]
    pub fn has_object(&self) -> bool {
        self.object.is_some()
    }

    /// Whether this node has no parent.
    #[inline]
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// Render the scene graph rooted by this node.
    pub fn render(&mut self, camera: &mut dyn PlanarCamera, context: &mut PlanarRenderContext) {
        if self.visible {
            self.do_render(&na::one(), &Vector2::repeat(1.0), camera, context)
        }
    }

    fn do_render(
        &mut self,
        transform: &Isometry2<f32>,
        scale: &Vector2<f32>,
        camera: &mut dyn PlanarCamera,
        context: &mut PlanarRenderContext,
    ) {
        if !self.up_to_date {
            self.up_to_date = true;
            self.world_transform = *transform * self.local_transform;
            self.world_scale = scale.component_mul(&self.local_scale);
        }

        if let Some(ref mut o) = self.object {
            o.render(&self.world_transform, &self.world_scale, camera, context)
        }

        for c in self.children.iter_mut() {
            let mut bc = c.data_mut();
            if bc.visible {
                bc.do_render(&self.world_transform, &self.world_scale, camera, context)
            }
        }
    }

    /// A reference to the object possibly contained by this node.
    #[inline]
    pub fn object(&self) -> Option<&PlanarObject> {
        self.object.as_ref()
    }

    /// A mutable reference to the object possibly contained by this node.
    #[inline]
    pub fn object_mut(&mut self) -> Option<&mut PlanarObject> {
        self.object.as_mut()
    }

    /// A reference to the object possibly contained by this node.
    ///
    /// # Failure
    /// Fails of this node does not contains an object.
    #[inline]
    pub fn get_object(&self) -> &PlanarObject {
        self.object()
            .expect("This scene node does not contain an PlanarObject.")
    }

    /// A mutable reference to the object possibly contained by this node.
    ///
    /// # Failure
    /// Fails of this node does not contains an object.
    #[inline]
    pub fn get_object_mut(&mut self) -> &mut PlanarObject {
        self.object_mut()
            .expect("This scene node does not contain an PlanarObject.")
    }

    ///////////////////~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HERE
    /* FIXME: the ~Any is kind of problematic here…
    /// Attaches user-defined data to the objects contained by this node and its children.
    #[inline]
    pub fn set_user_data(&mut self, user_data: ~Any) {
        self.apply_to_objects_mut(&mut |o| o.set_user_data(user_data))
    }
    */

    // FIXME: for all those set_stuff, would it be more per formant to add a special case for when
    // we are on a leaf? (to avoid the call to a closure required by the apply_to_*).
    /// Sets the material of the objects contained by this node and its children.
    #[inline]
    pub fn set_material(&mut self, material: Rc<RefCell<Box<dyn PlanarMaterial + 'static>>>) {
        self.apply_to_objects_mut(&mut |o| o.set_material(material.clone()))
    }

    /// Sets the material of the objects contained by this node and its children.
    ///
    /// The material must already have been registered as `name`.
    #[inline]
    pub fn set_material_with_name(&mut self, name: &str) {
        let material = PlanarMaterialManager::get_global_manager(|tm| {
            tm.get(name).unwrap_or_else(|| {
                panic!("Invalid attempt to use the unregistered material: {}", name)
            })
        });

        self.set_material(material)
    }

    /// Sets the width of the lines drawn for the objects contained by this node and its children.
    ///
    /// If `use_perspective` is true, width is in world units and scales with camera zoom.
    /// If `use_perspective` is false, width is in screen pixels and stays constant.
    #[inline]
    pub fn set_lines_width(&mut self, width: f32, use_perspective: bool) {
        self.apply_to_objects_mut(&mut |o| o.set_lines_width(width, use_perspective))
    }

    /// Sets the color of the lines drawn for the objects contained by this node and its children.
    #[inline]
    pub fn set_lines_color(&mut self, color: Option<Point3<f32>>) {
        self.apply_to_objects_mut(&mut |o| o.set_lines_color(color))
    }

    /// Sets the size of the points drawn for the objects contained by this node and its children.
    ///
    /// If `use_perspective` is true, size is in world units and scales with camera zoom.
    /// If `use_perspective` is false, size is in screen pixels and stays constant.
    #[inline]
    pub fn set_points_size(&mut self, size: f32, use_perspective: bool) {
        self.apply_to_objects_mut(&mut |o| o.set_points_size(size, use_perspective))
    }

    /// Sets the color of the points drawn for the objects contained by this node and its children.
    #[inline]
    pub fn set_points_color(&mut self, color: Option<Point3<f32>>) {
        self.apply_to_objects_mut(&mut |o| o.set_points_color(color))
    }

    /// Activates or deactivates the rendering of the surfaces of the objects contained by this node and its
    /// children.
    #[inline]
    pub fn set_surface_rendering_activation(&mut self, active: bool) {
        self.apply_to_objects_mut(&mut |o| o.set_surface_rendering_activation(active))
    }

    /// Activates or deactivates backface culling for the objects contained by this node and its
    /// children.
    #[inline]
    pub fn enable_backface_culling(&mut self, active: bool) {
        self.apply_to_objects_mut(&mut |o| o.enable_backface_culling(active))
    }

    /// Mutably accesses the vertices of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn modify_vertices<F: FnMut(&mut Vec<Point2<f32>>)>(&mut self, f: &mut F) {
        self.apply_to_objects_mut(&mut |o| o.modify_vertices(f))
    }

    /// Accesses the vertices of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn read_vertices<F: FnMut(&[Point2<f32>])>(&self, f: &mut F) {
        self.apply_to_objects(&mut |o| o.read_vertices(f))
    }

    /// Mutably accesses the faces of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn modify_faces<F: FnMut(&mut Vec<Point3<VertexIndex>>)>(&mut self, f: &mut F) {
        self.apply_to_objects_mut(&mut |o| o.modify_faces(f))
    }

    /// Accesses the faces of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn read_faces<F: FnMut(&[Point3<VertexIndex>])>(&self, f: &mut F) {
        self.apply_to_objects(&mut |o| o.read_faces(f))
    }

    /// Mutably accesses the texture coordinates of the objects contained by this node and its
    /// children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn modify_uvs<F: FnMut(&mut Vec<Point2<f32>>)>(&mut self, f: &mut F) {
        self.apply_to_objects_mut(&mut |o| o.modify_uvs(f))
    }

    /// Accesses the texture coordinates of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn read_uvs<F: FnMut(&[Point2<f32>])>(&self, f: &mut F) {
        self.apply_to_objects(&mut |o| o.read_uvs(f))
    }

    /// Get the visibility status of node.
    #[inline]
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Sets the visibility of this node.
    ///
    /// The node and its children are not rendered if it is not visible.
    #[inline]
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Sets the color of the objects contained by this node and its children.
    ///
    /// Colors components must be on the range `[0.0, 1.0]`.
    #[inline]
    pub fn set_color(&mut self, r: f32, g: f32, b: f32) {
        self.apply_to_objects_mut(&mut |o| o.set_color(r, g, b))
    }

    /// Sets the texture of the objects contained by this node and its children.
    ///
    /// The texture is loaded from a file and registered by the global `TextureManager`.
    ///
    /// # Arguments
    ///   * `path` - relative path of the texture on the disk
    ///   * `name` - &str identifier to store this texture under
    #[inline]
    pub fn set_texture_from_file(&mut self, path: &Path, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));

        self.set_texture(texture)
    }

    /// Sets the texture of the objects contained by this node and its children.
    ///
    /// The texture is loaded from a byte slice and registered by the global `TextureManager`.
    ///
    /// # Arguments
    ///   * `image_data` - slice of bytes containing encoded image
    ///   * `name` - &str identifier to store this texture under
    #[inline]
    pub fn set_texture_from_memory(&mut self, image_data: &[u8], name: &str) {
        let texture =
            TextureManager::get_global_manager(|tm| tm.add_image_from_memory(image_data, name));

        self.set_texture(texture)
    }

    /// Sets the texture of the objects contained by this node and its children.
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

    /// Sets the texture of the objects contained by this node and its children.
    pub fn set_texture(&mut self, texture: Arc<Texture>) {
        self.apply_to_objects_mut(&mut |o| o.set_texture(texture.clone()))
    }

    /// Applies a closure to each object contained by this node and its children.
    #[inline]
    pub fn apply_to_objects_mut<F: FnMut(&mut PlanarObject)>(&mut self, f: &mut F) {
        if let Some(ref mut o) = self.object {
            f(o)
        }

        for c in self.children.iter_mut() {
            c.data_mut().apply_to_objects_mut(f)
        }
    }

    /// Applies a closure to each object contained by this node and its children.
    #[inline]
    pub fn apply_to_objects<F: FnMut(&PlanarObject)>(&self, f: &mut F) {
        if let Some(ref o) = self.object {
            f(o)
        }

        for c in self.children.iter() {
            c.data().apply_to_objects(f)
        }
    }

    // FIXME: add folding?

    /// Sets the local scaling factors of the object.
    #[inline]
    pub fn set_local_scale(&mut self, sx: f32, sy: f32) {
        self.invalidate();
        self.local_scale = Vector2::new(sx, sy)
    }

    /// Returns the scaling factors of the object.
    #[inline]
    pub fn local_scale(&self) -> Vector2<f32> {
        self.local_scale
    }

    /// This node local transformation.
    #[inline]
    pub fn local_transformation(&self) -> Isometry2<f32> {
        self.local_transform
    }

    /// Inverse of this node local transformation.
    #[inline]
    pub fn inverse_local_transformation(&self) -> Isometry2<f32> {
        self.local_transform.inverse()
    }

    /// This node world transformation.
    ///
    /// This will force an update of the world transformation of its parents if they have been
    /// invalidated.
    #[inline]
    #[allow(mutable_transmutes)]
    pub fn world_transformation(&self) -> Isometry2<f32> {
        // NOTE: this is to have some kind of laziness without a `&mut self`.
        unsafe {
            let mself: &mut PlanarSceneNodeData = mem::transmute(self);
            mself.update();
        }
        self.world_transform
    }

    /// The inverse of this node world transformation.
    ///
    /// This will force an update of the world transformation of its parents if they have been
    /// invalidated.
    #[inline]
    #[allow(mutable_transmutes)]
    pub fn inverse_world_transformation(&self) -> Isometry2<f32> {
        // NOTE: this is to have some kind of laziness without a `&mut self`.
        unsafe {
            let mself: &mut PlanarSceneNodeData = mem::transmute(self);
            mself.update();
        }
        self.local_transform.inverse()
    }

    /// Appends a transformation to this node local transformation.
    #[inline]
    pub fn append_transformation(&mut self, t: &Isometry2<f32>) {
        self.invalidate();
        self.local_transform = t * self.local_transform
    }

    /// Prepends a transformation to this node local transformation.
    #[inline]
    pub fn prepend_to_local_transformation(&mut self, t: &Isometry2<f32>) {
        self.invalidate();
        self.local_transform *= t;
    }

    /// Set this node local transformation.
    #[inline]
    pub fn set_local_transformation(&mut self, t: Isometry2<f32>) {
        self.invalidate();
        self.local_transform = t
    }

    /// This node local translation.
    #[inline]
    pub fn local_translation(&self) -> Translation2<f32> {
        self.local_transform.translation
    }

    /// The inverse of this node local translation.
    #[inline]
    pub fn inverse_local_translation(&self) -> Translation2<f32> {
        self.local_transform.translation.inverse()
    }

    /// Appends a translation to this node local transformation.
    #[inline]
    pub fn append_translation(&mut self, t: &Translation2<f32>) {
        self.invalidate();
        self.local_transform = t * self.local_transform
    }

    /// Prepends a translation to this node local transformation.
    #[inline]
    pub fn prepend_to_local_translation(&mut self, t: &Translation2<f32>) {
        self.invalidate();
        self.local_transform *= t
    }

    /// Sets the local translation of this node.
    #[inline]
    pub fn set_local_translation(&mut self, t: Translation2<f32>) {
        self.invalidate();
        self.local_transform.translation = t
    }

    /// This node local rotation.
    #[inline]
    pub fn local_rotation(&self) -> UnitComplex<f32> {
        self.local_transform.rotation
    }

    /// The inverse of this node local rotation.
    #[inline]
    pub fn inverse_local_rotation(&self) -> UnitComplex<f32> {
        self.local_transform.rotation.inverse()
    }

    /// Appends a rotation to this node local transformation.
    #[inline]
    pub fn append_rotation(&mut self, r: &UnitComplex<f32>) {
        self.invalidate();
        self.local_transform = r * self.local_transform
    }

    /// Appends a rotation to this node local transformation.
    #[inline]
    pub fn append_rotation_wrt_center(&mut self, r: &UnitComplex<f32>) {
        self.invalidate();
        self.local_transform.append_rotation_wrt_center_mut(r)
    }

    /// Prepends a rotation to this node local transformation.
    #[inline]
    pub fn prepend_to_local_rotation(&mut self, r: &UnitComplex<f32>) {
        self.invalidate();
        self.local_transform *= r
    }

    /// Sets the local rotation of this node.
    #[inline]
    pub fn set_local_rotation(&mut self, r: UnitComplex<f32>) {
        self.invalidate();
        self.local_transform.rotation = r
    }

    fn invalidate(&mut self) {
        self.up_to_date = false;

        for c in self.children.iter_mut() {
            let mut dm = c.data_mut();

            if dm.up_to_date {
                dm.invalidate()
            }
        }
    }

    // FIXME: make this public?
    fn update(&mut self) {
        // NOTE: makin this test
        if !self.up_to_date {
            if let Some(ref mut p) = self.parent {
                unsafe {
                    let mut dp = (**p).borrow_mut();

                    dp.update();
                    self.world_transform = self.local_transform * dp.world_transform;
                    self.world_scale = self.local_scale.component_mul(&dp.local_scale);
                    self.up_to_date = true;
                    return;
                }
            }

            // no parent
            self.world_transform = self.local_transform;
            self.world_scale = self.local_scale;
            self.up_to_date = true;
        }
    }
}

impl PlanarSceneNode {
    /// Creates a new scene node that is not rooted.
    pub fn new(
        local_scale: Vector2<f32>,
        local_transform: Isometry2<f32>,
        object: Option<PlanarObject>,
    ) -> PlanarSceneNode {
        let data = PlanarSceneNodeData {
            local_scale,
            local_transform,
            world_transform: local_transform,
            world_scale: local_scale,
            visible: true,
            up_to_date: false,
            children: Vec::new(),
            object,
            parent: None,
        };

        PlanarSceneNode {
            data: Rc::new(RefCell::new(data)),
        }
    }

    /// Creates a new empty, not rooted, node with identity transformations.
    pub fn new_empty() -> PlanarSceneNode {
        PlanarSceneNode::new(Vector2::from_element(1.0), na::one(), None)
    }

    /// Removes this node from its parent.
    pub fn unlink(&mut self) {
        let self_self = self.clone();
        self.data_mut().remove_from_parent(&self_self);
        self.data_mut().parent = None
    }

    /// The data of this scene node.
    pub fn data(&self) -> Ref<'_, PlanarSceneNodeData> {
        self.data.borrow()
    }

    /// The data of this scene node.
    pub fn data_mut(&mut self) -> RefMut<'_, PlanarSceneNodeData> {
        self.data.borrow_mut()
    }

    /*
     *
     * Methods to add objects.
     *
     */
    /// Adds a node without object to this node children.
    pub fn add_group(&mut self) -> PlanarSceneNode {
        let node = PlanarSceneNode::new_empty();

        self.add_child(node.clone());

        node
    }

    /// Adds a node as a child of `parent`.
    ///
    /// # Failures:
    /// Fails if `node` already has a parent.
    pub fn add_child(&mut self, node: PlanarSceneNode) {
        assert!(
            node.data().is_root(),
            "The added node must not have a parent yet."
        );

        let mut node = node;
        node.data_mut().set_parent(&*self.data);
        self.data_mut().children.push(node)
    }

    /// Adds a node containing an object to this node children.
    pub fn add_object(
        &mut self,
        local_scale: Vector2<f32>,
        local_transform: Isometry2<f32>,
        object: PlanarObject,
    ) -> PlanarSceneNode {
        let node = PlanarSceneNode::new(local_scale, local_transform, Some(object));

        self.add_child(node.clone());

        node
    }

    /// Adds a rectangle as a children of this node. The rectangle is initially axis-aligned and centered
    /// at (0, 0).
    ///
    /// # Arguments
    /// * `wx` - the rectangle extent along the x axis
    /// * `wy` - the rectangle extent along the y axis
    pub fn add_rectangle(&mut self, wx: f32, wy: f32) -> PlanarSceneNode {
        let res = self.add_geom_with_name("rectangle", Vector2::new(wx, wy));

        res.expect("Unable to load the default rectangle geometry.")
    }

    /// Adds a circle as a children of this node. The circle is initially centered at (0, 0, 0).
    ///
    /// # Arguments
    /// * `r` - the circle radius
    pub fn add_circle(&mut self, r: f32) -> PlanarSceneNode {
        let res = self.add_geom_with_name("circle", Vector2::new(r * 2.0, r * 2.0));

        res.expect("Unable to load the default circle geometry.")
    }

    /// Adds a 2D capsule as a children of this node. The capsule is initially centered at (0, 0).
    ///
    /// # Arguments
    /// * `r` - the circle radius
    pub fn add_capsule(&mut self, r: f32, h: f32) -> PlanarSceneNode {
        let name = format!("capsule_{}_{}", r, h);

        let mesh = PlanarMeshManager::get_global_manager(|mm| {
            if let Some(geom) = mm.get(&name) {
                geom
            } else {
                // Create the capsule geometry.
                let mut capsule_vtx = vec![Point2::origin()];
                let mut capsule_ids = Vec::new();
                let nsamples = 50;

                for i in 0..=nsamples {
                    let ang = (i as f32) / (nsamples as f32) * f32::consts::PI;
                    capsule_vtx.push(Point2::new(ang.cos() * r, ang.sin() * r + h / 2.0));
                    capsule_ids.push(Point3::new(
                        0,
                        capsule_vtx.len() as VertexIndex - 2,
                        capsule_vtx.len() as VertexIndex - 1,
                    ));
                }

                for i in nsamples..=nsamples * 2 {
                    let ang = (i as f32) / (nsamples as f32) * f32::consts::PI;
                    capsule_vtx.push(Point2::new(ang.cos() * r, ang.sin() * r - h / 2.0));
                    capsule_ids.push(Point3::new(
                        0,
                        capsule_vtx.len() as VertexIndex - 2,
                        capsule_vtx.len() as VertexIndex - 1,
                    ));
                }

                capsule_ids.push(Point3::new(0, capsule_vtx.len() as VertexIndex - 1, 1));

                let capsule = PlanarMesh::new(capsule_vtx, capsule_ids, None, false);
                let mesh = Rc::new(RefCell::new(capsule));
                mm.add(mesh.clone(), &name);
                mesh
            }
        });

        self.add_mesh(mesh, Vector2::repeat(1.0))
    }

    /// Creates and adds a new object using the geometry registered as `geometry_name`.
    pub fn add_geom_with_name(
        &mut self,
        geometry_name: &str,
        scale: Vector2<f32>,
    ) -> Option<PlanarSceneNode> {
        PlanarMeshManager::get_global_manager(|mm| mm.get(geometry_name))
            .map(|g| self.add_mesh(g, scale))
    }

    /// Creates and adds a new object to this node children using a 2D mesh.
    pub fn add_mesh(
        &mut self,
        mesh: Rc<RefCell<PlanarMesh>>,
        scale: Vector2<f32>,
    ) -> PlanarSceneNode {
        let tex = TextureManager::get_global_manager(|tm| tm.get_default());
        let mat = PlanarMaterialManager::get_global_manager(|mm| mm.get_default());
        let object = PlanarObject::new(mesh, 1.0, 1.0, 1.0, tex, mat);

        self.add_object(scale, na::one(), object)
    }

    /// Creates and adds a new object to this node children using a convex polyline
    pub fn add_convex_polygon(
        &mut self,
        polygon: Vec<Point2<f32>>,
        scale: Vector2<f32>,
    ) -> PlanarSceneNode {
        let mut indices = Vec::new();

        for i in 1..polygon.len() - 1 {
            indices.push(Point3::new(0, i as VertexIndex, i as VertexIndex + 1));
        }

        let mesh = PlanarMesh::new(polygon, indices, None, false);
        let tex = TextureManager::get_global_manager(|tm| tm.get_default());
        let mat = PlanarMaterialManager::get_global_manager(|mm| mm.get_default());
        let object = PlanarObject::new(Rc::new(RefCell::new(mesh)), 1.0, 1.0, 1.0, tex, mat);

        self.add_object(scale, na::one(), object)
    }

    /// Applies a closure to each object contained by this node and its children.
    #[inline]
    pub fn apply_to_scene_nodes_mut<F: FnMut(&mut PlanarSceneNode)>(&mut self, f: &mut F) {
        f(self);

        for c in self.data_mut().children.iter_mut() {
            c.apply_to_scene_nodes_mut(f)
        }
    }

    /// Applies a closure to each object contained by this node and its children.
    #[inline]
    pub fn apply_to_scene_nodes<F: FnMut(&PlanarSceneNode)>(&self, f: &mut F) {
        f(self);

        for c in self.data().children.iter() {
            c.apply_to_scene_nodes(f)
        }
    }

    //
    //
    // fwd
    //
    //

    /// Render the scene graph rooted by this node.
    pub fn render(&mut self, camera: &mut dyn PlanarCamera, context: &mut PlanarRenderContext) {
        self.data_mut().render(camera, context)
    }

    /// Sets the material of the objects contained by this node and its children.
    #[inline]
    pub fn set_material(&mut self, material: Rc<RefCell<Box<dyn PlanarMaterial + 'static>>>) {
        self.data_mut().set_material(material)
    }

    /// Sets the material of the objects contained by this node and its children.
    #[inline]
    pub fn set_material_with_name(&mut self, name: &str) {
        self.data_mut().set_material_with_name(name)
    }

    /// Sets the width of the lines drawn for the objects contained by this node and its children.
    ///
    /// If `use_perspective` is true, width is in world units and scales with camera zoom.
    /// If `use_perspective` is false, width is in screen pixels and stays constant.
    #[inline]
    pub fn set_lines_width(&mut self, width: f32, use_perspective: bool) {
        self.data_mut().set_lines_width(width, use_perspective)
    }

    /// Sets the color of the lines drawn for the objects contained by this node and its children.
    #[inline]
    pub fn set_lines_color(&mut self, color: Option<Point3<f32>>) {
        self.data_mut().set_lines_color(color)
    }

    /// Sets the size of the points drawn for the objects contained by this node and its children.
    ///
    /// If `use_perspective` is true, size is in world units and scales with camera zoom.
    /// If `use_perspective` is false, size is in screen pixels and stays constant.
    #[inline]
    pub fn set_points_size(&mut self, size: f32, use_perspective: bool) {
        self.data_mut().set_points_size(size, use_perspective)
    }

    /// Sets the color of the points drawn for the objects contained by this node and its children.
    #[inline]
    pub fn set_points_color(&mut self, color: Option<Point3<f32>>) {
        self.data_mut().set_points_color(color)
    }

    /// Activates or deactivates the rendering of the surfaces of the objects contained by this node and its
    /// children.
    #[inline]
    pub fn set_surface_rendering_activation(&mut self, active: bool) {
        self.data_mut().set_surface_rendering_activation(active)
    }

    /// Activates or deactivates backface culling for the objects contained by this node and its
    /// children.
    #[inline]
    pub fn enable_backface_culling(&mut self, active: bool) {
        self.data_mut().enable_backface_culling(active)
    }

    /// Mutably accesses the vertices of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn modify_vertices<F: FnMut(&mut Vec<Point2<f32>>)>(&mut self, f: &mut F) {
        self.data_mut().modify_vertices(f)
    }

    /// Accesses the vertices of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn read_vertices<F: FnMut(&[Point2<f32>])>(&self, f: &mut F) {
        self.data().read_vertices(f)
    }

    /// Mutably accesses the faces of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn modify_faces<F: FnMut(&mut Vec<Point3<VertexIndex>>)>(&mut self, f: &mut F) {
        self.data_mut().modify_faces(f)
    }

    /// Accesses the faces of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn read_faces<F: FnMut(&[Point3<VertexIndex>])>(&self, f: &mut F) {
        self.data().read_faces(f)
    }

    /// Mutably accesses the texture coordinates of the objects contained by this node and its
    /// children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn modify_uvs<F: FnMut(&mut Vec<Point2<f32>>)>(&mut self, f: &mut F) {
        self.data_mut().modify_uvs(f)
    }

    /// Accesses the texture coordinates of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn read_uvs<F: FnMut(&[Point2<f32>])>(&self, f: &mut F) {
        self.data().read_uvs(f)
    }

    /// Get the visibility status of node.
    #[inline]
    pub fn is_visible(&self) -> bool {
        self.data().is_visible()
    }

    /// Sets the visibility of this node.
    ///
    /// The node and its children are not rendered if it is not visible.
    #[inline]
    pub fn set_visible(&mut self, visible: bool) {
        self.data_mut().set_visible(visible)
    }

    /// Sets the color of the objects contained by this node and its children.
    ///
    /// Colors components must be on the range `[0.0, 1.0]`.
    #[inline]
    pub fn set_color(&mut self, r: f32, g: f32, b: f32) {
        self.data_mut().set_color(r, g, b)
    }

    /// Sets the texture of the objects contained by this node and its children.
    ///
    /// The texture is loaded from a file and registered by the global `TextureManager`.
    ///
    /// # Arguments
    ///   * `path` - relative path of the texture on the disk
    ///   * `name` - &str to identify this texture in `TextureManager`
    #[inline]
    pub fn set_texture_from_file(&mut self, path: &Path, name: &str) {
        self.data_mut().set_texture_from_file(path, name)
    }

    /// Sets the texture of the objects contained by this node and its children.
    ///
    /// # Arguments
    ///   * `image_data` - slice of bytes containing encoded image
    ///   * `name` - &str to identify this texture in `TextureManager`
    pub fn set_texture_from_memory(&mut self, image_data: &[u8], name: &str) {
        self.data_mut().set_texture_from_memory(image_data, name)
    }

    /// Sets the texture of the objects contained by this node and its children.
    ///
    /// The texture must already have been registered as `name`.
    #[inline]
    pub fn set_texture_with_name(&mut self, name: &str) {
        self.data_mut().set_texture_with_name(name)
    }

    /// Sets the texture of the objects contained by this node and its children.
    pub fn set_texture(&mut self, texture: Arc<Texture>) {
        self.data_mut().set_texture(texture)
    }

    /// Sets the local scaling factors of the object.
    #[inline]
    pub fn set_local_scale(&mut self, sx: f32, sy: f32) {
        self.data_mut().set_local_scale(sx, sy)
    }

    /// Appends a transformation to this node local transformation.
    #[inline]
    pub fn append_transformation(&mut self, t: &Isometry2<f32>) {
        self.data_mut().append_transformation(t)
    }

    /// Prepends a transformation to this node local transformation.
    #[inline]
    pub fn prepend_to_local_transformation(&mut self, t: &Isometry2<f32>) {
        self.data_mut().prepend_to_local_transformation(t)
    }

    /// Set this node local transformation.
    #[inline]
    pub fn set_local_transformation(&mut self, t: Isometry2<f32>) {
        self.data_mut().set_local_transformation(t)
    }

    /// Appends a translation to this node local transformation.
    #[inline]
    pub fn append_translation(&mut self, t: &Translation2<f32>) {
        self.data_mut().append_translation(t)
    }

    /// Prepends a translation to this node local transformation.
    #[inline]
    pub fn prepend_to_local_translation(&mut self, t: &Translation2<f32>) {
        self.data_mut().prepend_to_local_translation(t)
    }

    /// Sets the local translation of this node.
    #[inline]
    pub fn set_local_translation(&mut self, t: Translation2<f32>) {
        self.data_mut().set_local_translation(t)
    }

    /// Appends a rotation to this node local transformation.
    #[inline]
    pub fn append_rotation(&mut self, r: &UnitComplex<f32>) {
        self.data_mut().append_rotation(r)
    }

    /// Appends a rotation to this node local transformation.
    #[inline]
    pub fn append_rotation_wrt_center(&mut self, r: &UnitComplex<f32>) {
        (*self.data_mut()).append_rotation_wrt_center(r)
    }

    /// Prepends a rotation to this node local transformation.
    #[inline]
    pub fn prepend_to_local_rotation(&mut self, r: &UnitComplex<f32>) {
        self.data_mut().prepend_to_local_rotation(r)
    }

    /// Sets the local rotation of this node.
    #[inline]
    pub fn set_local_rotation(&mut self, r: UnitComplex<f32>) {
        self.data_mut().set_local_rotation(r)
    }

    /// Sets the instances for rendering multiple duplicates of this scene node.
    ///
    /// This only duplicates this scene node, not any of its children.
    pub fn set_instances(&mut self, instances: &[PlanarInstanceData]) {
        self.data_mut().get_object_mut().set_instances(instances)
    }
}
