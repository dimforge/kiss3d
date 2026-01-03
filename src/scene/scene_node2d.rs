use crate::camera::Camera2d;
use crate::prelude::InstanceData2d;
use crate::resource::vertex_index::VertexIndex;
use crate::resource::{
    Material2d, MaterialManager2d, GpuMesh2d, MeshManager2d, RenderContext2d, Texture, TextureManager,
};
use crate::scene::Object2d;
use glamx::{Pose2, Rot2, Vec2};
use std::cell::{Ref, RefCell, RefMut};
use std::f32;
use std::path::Path;
use std::rc::{Rc, Weak};
use std::sync::Arc;
use crate::color::Color;

// XXX: once something like `fn foo(self: Rc<RefCell<SceneNode2d>>)` is allowed, this extra struct
// will not be needed any more.
/// The data contained by a `SceneNode2d`.
pub struct SceneNodeData2d {
    local_scale: Vec2,
    local_transform: Pose2,
    world_scale: Vec2,
    world_transform: Pose2,
    visible: bool,
    up_to_date: bool,
    children: Vec<SceneNode2d>,
    object: Option<Object2d>,
    parent: Option<Weak<RefCell<SceneNodeData2d>>>,
}

/// A node of the scene graph.
///
/// This may represent a group of other nodes, and/or contain an object that can be rendered.
#[derive(Clone)]
pub struct SceneNode2d {
    data: Rc<RefCell<SceneNodeData2d>>,
}

impl SceneNodeData2d {
    // XXX: Because `node.borrow_mut().parent = Some(self.data.downgrade())`
    // causes a weird compiler error:
    //
    // ```
    // error: mismatched types: expected `&std::cell::RefCell<scene::scene_node::SceneNodeData2d>`
    // but found
    // `std::option::Option<std::rc::Weak<std::cell::RefCell<scene::scene_node::SceneNodeData2d>>>`
    // (expe cted &-ptr but found enum std::option::Option)
    // ```
    fn set_parent(&mut self, parent: Weak<RefCell<SceneNodeData2d>>) {
        self.parent = Some(parent);
    }

    // TODO: this exists because of a similar bug as `set_parent`.
    fn remove_from_parent(&mut self, to_remove: &SceneNode2d) {
        let _ = self.parent.as_ref().map(|p| {
            if let Some(bp) = p.upgrade() {
                bp.borrow_mut().remove(to_remove);
            }
        });
    }

    fn remove(&mut self, o: &SceneNode2d) {
        if let Some(i) = self
            .children
            .iter()
            .rposition(|e| std::ptr::eq(&*o.data, &*e.data))
        {
            let _ = self.children.swap_remove(i);
        }
    }

    /// Whether this node contains an `Object2d`.
    #[inline]
    pub fn has_object(&self) -> bool {
        self.object.is_some()
    }

    /// Whether this node has no parent.
    #[inline]
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// Prepare the scene graph rooted by this node for rendering.
    pub fn prepare(&mut self, camera: &mut dyn Camera2d, context: &RenderContext2d) {
        if self.visible {
            self.do_prepare(Pose2::IDENTITY, Vec2::ONE, camera, context)
        }
    }

    fn do_prepare(
        &mut self,
        transform: Pose2,
        scale: Vec2,
        camera: &mut dyn Camera2d,
        context: &RenderContext2d,
    ) {
        if !self.up_to_date {
            self.up_to_date = true;
            self.world_transform = transform * self.local_transform;
            self.world_scale = scale * self.local_scale;
        }

        if let Some(ref mut o) = self.object {
            o.prepare(self.world_transform, self.world_scale, camera, context)
        }

        for c in self.children.iter_mut() {
            let mut bc = c.data_mut();
            if bc.visible {
                bc.do_prepare(self.world_transform, self.world_scale, camera, context)
            }
        }
    }

    /// Render the scene graph rooted by this node.
    pub fn render(
        &mut self,
        camera: &mut dyn Camera2d,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext2d,
    ) {
        if self.visible {
            self.do_render(Pose2::IDENTITY, Vec2::ONE, camera, render_pass, context)
        }
    }

    fn do_render(
        &mut self,
        transform: Pose2,
        scale: Vec2,
        camera: &mut dyn Camera2d,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext2d,
    ) {
        if !self.up_to_date {
            self.up_to_date = true;
            self.world_transform = transform * self.local_transform;
            self.world_scale = scale * self.local_scale;
        }

        if let Some(ref mut o) = self.object {
            o.render(
                self.world_transform,
                self.world_scale,
                camera,
                render_pass,
                context,
            )
        }

        for c in self.children.iter_mut() {
            let mut bc = c.data_mut();
            if bc.visible {
                bc.do_render(
                    self.world_transform,
                    self.world_scale,
                    camera,
                    render_pass,
                    context,
                )
            }
        }
    }

    /// A reference to the object possibly contained by this node.
    #[inline]
    pub fn object(&self) -> Option<&Object2d> {
        self.object.as_ref()
    }

    /// A mutable reference to the object possibly contained by this node.
    #[inline]
    pub fn object_mut(&mut self) -> Option<&mut Object2d> {
        self.object.as_mut()
    }

    /// A reference to the object possibly contained by this node.
    ///
    /// # Failure
    /// Fails of this node does not contains an object.
    #[inline]
    pub fn get_object(&self) -> &Object2d {
        self.object()
            .expect("This scene node does not contain an Object2d.")
    }

    /// A mutable reference to the object possibly contained by this node.
    ///
    /// # Failure
    /// Fails of this node does not contains an object.
    #[inline]
    pub fn get_object_mut(&mut self) -> &mut Object2d {
        self.object_mut()
            .expect("This scene node does not contain an Object2d.")
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

    // TODO: make this public?
    fn update(&mut self) {
        if !self.up_to_date {
            if let Some(ref mut p) = self.parent {
                if let Some(dp) = p.upgrade() {
                    let mut dp = dp.borrow_mut();
                    dp.update();
                    self.world_transform = self.local_transform * dp.world_transform;
                    self.world_scale = self.local_scale * dp.local_scale;
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

impl Default for SceneNode2d {
    fn default() -> Self {
        Self::empty()
    }
}

impl SceneNode2d {
    /// Creates a new scene node that is not rooted.
    pub fn new(local_scale: Vec2, local_transform: Pose2, object: Option<Object2d>) -> SceneNode2d {
        let data = SceneNodeData2d {
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

        SceneNode2d {
            data: Rc::new(RefCell::new(data)),
        }
    }

    /// Creates a new empty, not rooted, node with identity transformations.
    pub fn empty() -> SceneNode2d {
        SceneNode2d::new(Vec2::ONE, Pose2::IDENTITY, None)
    }

    // ==================
    // Primitive constructors
    // ==================

    /// Creates a new scene node with a rectangle mesh.
    ///
    /// The rectangle is initially axis-aligned and centered at (0, 0).
    ///
    /// # Arguments
    /// * `wx` - the rectangle extent along the x axis
    /// * `wy` - the rectangle extent along the y axis
    pub fn rectangle(wx: f32, wy: f32) -> SceneNode2d {
        Self::geom_with_name("rectangle", Vec2::new(wx, wy))
            .expect("Unable to load the default rectangle geometry.")
    }

    /// Creates a new scene node with a circle mesh.
    ///
    /// The circle is initially centered at (0, 0).
    ///
    /// # Arguments
    /// * `r` - the circle radius
    pub fn circle(r: f32) -> SceneNode2d {
        Self::geom_with_name("circle", Vec2::new(r * 2.0, r * 2.0))
            .expect("Unable to load the default circle geometry.")
    }

    /// Creates a new scene node with a circle mesh with custom subdivisions.
    ///
    /// The circle is initially centered at (0, 0).
    ///
    /// # Arguments
    /// * `r` - the circle radius
    /// * `nsubdiv` - number of subdivisions around the circumference
    pub fn circle_with_subdiv(r: f32, nsubdiv: u32) -> SceneNode2d {
        let mut circle_vtx = vec![Vec2::ZERO];
        let mut circle_ids = Vec::new();

        for i in 0..nsubdiv {
            let ang = (i as f32) / (nsubdiv as f32) * f32::consts::TAU;
            circle_vtx.push(Vec2::new(ang.cos(), ang.sin()) * r);
            circle_ids.push([
                0,
                circle_vtx.len() as VertexIndex - 2,
                circle_vtx.len() as VertexIndex - 1,
            ]);
        }
        circle_ids.push([0, circle_vtx.len() as VertexIndex - 1, 1]);

        let circle = GpuMesh2d::new(circle_vtx, circle_ids, None, false);
        Self::mesh(Rc::new(RefCell::new(circle)), Vec2::ONE)
    }

    /// Creates a new scene node with a 2D capsule mesh.
    ///
    /// The capsule is initially centered at (0, 0).
    ///
    /// # Arguments
    /// * `r` - the capsule caps radius
    /// * `h` - the capsule height
    pub fn capsule(r: f32, h: f32) -> SceneNode2d {
        let name = format!("capsule_{}_{}", r, h);

        let mesh = MeshManager2d::get_global_manager(|mm| {
            if let Some(geom) = mm.get(&name) {
                geom
            } else {
                // Create the capsule geometry.
                let mut capsule_vtx = vec![Vec2::ZERO];
                let mut capsule_ids = Vec::new();
                let nsamples = 50;

                for i in 0..=nsamples {
                    let ang = (i as f32) / (nsamples as f32) * f32::consts::PI;
                    capsule_vtx.push(Vec2::new(ang.cos() * r, ang.sin() * r + h / 2.0));
                    capsule_ids.push([
                        0,
                        capsule_vtx.len() as VertexIndex - 2,
                        capsule_vtx.len() as VertexIndex - 1,
                    ]);
                }

                for i in nsamples..=nsamples * 2 {
                    let ang = (i as f32) / (nsamples as f32) * f32::consts::PI;
                    capsule_vtx.push(Vec2::new(ang.cos() * r, ang.sin() * r - h / 2.0));
                    capsule_ids.push([
                        0,
                        capsule_vtx.len() as VertexIndex - 2,
                        capsule_vtx.len() as VertexIndex - 1,
                    ]);
                }

                capsule_ids.push([0, capsule_vtx.len() as VertexIndex - 1, 1]);

                let capsule = GpuMesh2d::new(capsule_vtx, capsule_ids, None, false);
                let mesh = Rc::new(RefCell::new(capsule));
                mm.add(mesh.clone(), &name);
                mesh
            }
        });

        Self::mesh(mesh, Vec2::ONE)
    }

    /// Creates a new scene node with a 2D capsule mesh with custom subdivisions.
    ///
    /// The capsule is initially centered at (0, 0).
    ///
    /// # Arguments
    /// * `r` - the capsule caps radius
    /// * `h` - the capsule height
    /// * `nsubdiv` - number of subdivisions for each semicircular cap
    pub fn capsule_with_subdiv(r: f32, h: f32, nsubdiv: u32) -> SceneNode2d {
        let mut capsule_vtx = vec![Vec2::ZERO];
        let mut capsule_ids = Vec::new();

        for i in 0..=nsubdiv {
            let ang = (i as f32) / (nsubdiv as f32) * f32::consts::PI;
            capsule_vtx.push(Vec2::new(ang.cos() * r, ang.sin() * r + h / 2.0));
            capsule_ids.push([
                0,
                capsule_vtx.len() as VertexIndex - 2,
                capsule_vtx.len() as VertexIndex - 1,
            ]);
        }

        for i in nsubdiv..=nsubdiv * 2 {
            let ang = (i as f32) / (nsubdiv as f32) * f32::consts::PI;
            capsule_vtx.push(Vec2::new(ang.cos() * r, ang.sin() * r - h / 2.0));
            capsule_ids.push([
                0,
                capsule_vtx.len() as VertexIndex - 2,
                capsule_vtx.len() as VertexIndex - 1,
            ]);
        }

        capsule_ids.push([0, capsule_vtx.len() as VertexIndex - 1, 1]);

        let capsule = GpuMesh2d::new(capsule_vtx, capsule_ids, None, false);
        Self::mesh(Rc::new(RefCell::new(capsule)), Vec2::ONE)
    }

    /// Creates a new scene node with a polyline.
    pub fn polyline(
        vertices: Vec<Vec2>,
        indices: Option<Vec<[u32; 2]>>,
        line_width: f32,
    ) -> SceneNode2d {
        let indices = if let Some(indices) = indices {
            indices
                .into_iter()
                .map(|idx| [idx[0], idx[0], idx[1]])
                .collect()
        } else {
            (0..vertices.len() - 1)
                .map(|i| [i as u32, i as u32, i as u32 + 1])
                .collect()
        };

        let mesh = GpuMesh2d::new(vertices, indices, None, false);
        let mut node = Self::mesh(Rc::new(RefCell::new(mesh)), Vec2::ONE);
        node.set_surface_rendering_activation(false);
        node.set_lines_width(line_width, true);
        node
    }

    /// Creates a new scene node using the geometry registered as `geometry_name`.
    pub fn geom_with_name(geometry_name: &str, scale: Vec2) -> Option<SceneNode2d> {
        MeshManager2d::get_global_manager(|mm| mm.get(geometry_name)).map(|g| Self::mesh(g, scale))
    }

    /// Creates a new scene node using a 2D mesh.
    pub fn mesh(mesh: Rc<RefCell<GpuMesh2d>>, scale: Vec2) -> SceneNode2d {
        let tex = TextureManager::get_global_manager(|tm| tm.get_default());
        let mat = MaterialManager2d::get_global_manager(|mm| mm.get_default());
        let object = Object2d::new(mesh, 1.0, 1.0, 1.0, tex, mat);

        SceneNode2d::new(scale, Pose2::IDENTITY, Some(object))
    }

    /// Creates a new scene node using a convex polyline.
    pub fn convex_polygon(polygon: Vec<Vec2>, scale: Vec2) -> SceneNode2d {
        let mut indices = Vec::new();

        for i in 1..polygon.len() - 1 {
            indices.push([0, i as VertexIndex, i as VertexIndex + 1]);
        }

        let mesh = GpuMesh2d::new(polygon, indices, None, false);
        let tex = TextureManager::get_global_manager(|tm| tm.get_default());
        let mat = MaterialManager2d::get_global_manager(|mm| mm.get_default());
        let object = Object2d::new(Rc::new(RefCell::new(mesh)), 1.0, 1.0, 1.0, tex, mat);

        SceneNode2d::new(scale, Pose2::IDENTITY, Some(object))
    }

    /// Removes this node from its parent.
    pub fn detach(&mut self) {
        let self_self = self.clone();
        self.data_mut().remove_from_parent(&self_self);
        self.data_mut().parent = None
    }

    /// The data of this scene node.
    pub fn data(&self) -> Ref<'_, SceneNodeData2d> {
        self.data.borrow()
    }

    /// The data of this scene node.
    pub fn data_mut(&mut self) -> RefMut<'_, SceneNodeData2d> {
        self.data.borrow_mut()
    }

    /*
     *
     * Methods to add objects.
     *
     */
    /// Adds a node without object to this node children.
    pub fn add_group(&mut self) -> SceneNode2d {
        let node = SceneNode2d::empty();

        self.add_child(node.clone());

        node
    }

    /// Adds a node as a child of `parent`.
    ///
    /// # Failures:
    /// Fails if `node` already has a parent.
    pub fn add_child(&mut self, node: SceneNode2d) {
        assert!(
            node.data().is_root(),
            "The added node must not have a parent yet."
        );

        let mut node = node;
        let self_weak_ptr = Rc::downgrade(&self.data);
        node.data_mut().set_parent(self_weak_ptr);
        self.data_mut().children.push(node)
    }

    /// Adds a node containing an object to this node children.
    pub fn add_object(
        &mut self,
        local_scale: Vec2,
        local_transform: Pose2,
        object: Object2d,
    ) -> SceneNode2d {
        let node = SceneNode2d::new(local_scale, local_transform, Some(object));

        self.add_child(node.clone());

        node
    }

    /// Adds a rectangle as a children of this node. The rectangle is initially axis-aligned and centered
    /// at (0, 0).
    ///
    /// # Arguments
    /// * `wx` - the rectangle extent along the x axis
    /// * `wy` - the rectangle extent along the y axis
    pub fn add_rectangle(&mut self, wx: f32, wy: f32) -> SceneNode2d {
        let node = Self::rectangle(wx, wy);
        self.add_child(node.clone());
        node
    }

    /// Adds a circle as a children of this node. The circle is initially centered at (0, 0, 0).
    ///
    /// # Arguments
    /// * `r` - the circle radius
    pub fn add_circle(&mut self, r: f32) -> SceneNode2d {
        let node = Self::circle(r);
        self.add_child(node.clone());
        node
    }

    /// Adds a circle with custom subdivisions as a child of this node.
    ///
    /// The circle is initially centered at (0, 0).
    ///
    /// # Arguments
    /// * `r` - the circle radius
    /// * `nsubdiv` - number of subdivisions around the circumference
    pub fn add_circle_with_subdiv(&mut self, r: f32, nsubdiv: u32) -> SceneNode2d {
        let node = Self::circle_with_subdiv(r, nsubdiv);
        self.add_child(node.clone());
        node
    }

    pub fn add_polyline(
        &mut self,
        vertices: Vec<Vec2>,
        indices: Option<Vec<[u32; 2]>>,
        line_width: f32,
    ) -> SceneNode2d {
        let node = Self::polyline(vertices, indices, line_width);
        self.add_child(node.clone());
        node
    }

    /// Adds a 2D capsule as a children of this node. The capsule is initially centered at (0, 0).
    ///
    /// # Arguments
    /// * `r` - the capsule caps radius
    /// * `h` - the capsule height
    pub fn add_capsule(&mut self, r: f32, h: f32) -> SceneNode2d {
        let node = Self::capsule(r, h);
        self.add_child(node.clone());
        node
    }

    /// Adds a 2D capsule with custom subdivisions as a child of this node.
    ///
    /// The capsule is initially centered at (0, 0).
    ///
    /// # Arguments
    /// * `r` - the capsule caps radius
    /// * `h` - the capsule height
    /// * `nsubdiv` - number of subdivisions for each semicircular cap
    pub fn add_capsule_with_subdiv(&mut self, r: f32, h: f32, nsubdiv: u32) -> SceneNode2d {
        let node = Self::capsule_with_subdiv(r, h, nsubdiv);
        self.add_child(node.clone());
        node
    }

    /// Creates and adds a new object using the geometry registered as `geometry_name`.
    pub fn add_geom_with_name(&mut self, geometry_name: &str, scale: Vec2) -> Option<SceneNode2d> {
        Self::geom_with_name(geometry_name, scale).inspect(|node| {
            self.add_child(node.clone());
        })
    }

    /// Creates and adds a new object to this node children using a 2D mesh.
    pub fn add_mesh(&mut self, mesh: Rc<RefCell<GpuMesh2d>>, scale: Vec2) -> SceneNode2d {
        let node = Self::mesh(mesh, scale);
        self.add_child(node.clone());
        node
    }

    /// Creates and adds a new object to this node children using a convex polyline
    pub fn add_convex_polygon(&mut self, polygon: Vec<Vec2>, scale: Vec2) -> SceneNode2d {
        let node = Self::convex_polygon(polygon, scale);
        self.add_child(node.clone());
        node
    }

    /// Applies a closure to each object contained by this node and its children.
    #[inline]
    pub fn apply_to_scene_nodes_mut<F: FnMut(&mut SceneNode2d)>(&mut self, f: &mut F) {
        f(self);

        for c in self.data_mut().children.iter_mut() {
            c.apply_to_scene_nodes_mut(f)
        }
    }

    /// Applies a closure to each object contained by this node and its children.
    #[inline]
    pub fn apply_to_scene_nodes<F: FnMut(&SceneNode2d)>(&self, f: &mut F) {
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

    /// Prepare the scene graph rooted by this node for rendering.
    pub fn prepare(&mut self, camera: &mut dyn Camera2d, context: &RenderContext2d) {
        self.data_mut().prepare(camera, context)
    }

    /// Render the scene graph rooted by this node.
    pub fn render(
        &mut self,
        camera: &mut dyn Camera2d,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext2d,
    ) {
        self.data_mut().render(camera, render_pass, context)
    }

    /// Sets the material of the objects contained by this node and its children.
    #[inline]
    pub fn set_material(&mut self, material: Rc<RefCell<Box<dyn Material2d + 'static>>>) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_material(material.clone()));
        self.clone()
    }

    /// Sets the material of the objects contained by this node and its children.
    ///
    /// The material must already have been registered as `name`.
    #[inline]
    pub fn set_material_with_name(&mut self, name: &str) -> Self {
        let material = MaterialManager2d::get_global_manager(|tm| {
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
    pub fn set_lines_width(&mut self, width: f32, use_perspective: bool) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_lines_width(width, use_perspective));
        self.clone()
    }

    /// Sets the color of the lines drawn for the objects contained by this node and its children.
    #[inline]
    pub fn set_lines_color(&mut self, color: Option<Color>) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_lines_color(color));
        self.clone()
    }

    /// Sets the size of the points drawn for the objects contained by this node and its children.
    ///
    /// If `use_perspective` is true, size is in world units and scales with camera zoom.
    /// If `use_perspective` is false, size is in screen pixels and stays constant.
    #[inline]
    pub fn set_points_size(&mut self, size: f32, use_perspective: bool) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_points_size(size, use_perspective));
        self.clone()
    }

    /// Sets the color of the points drawn for the objects contained by this node and its children.
    #[inline]
    pub fn set_points_color(&mut self, color: Option<Color>) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_points_color(color));
        self.clone()
    }

    /// Activates or deactivates the rendering of the surfaces of the objects contained by this node and its
    /// children.
    #[inline]
    pub fn set_surface_rendering_activation(&mut self, active: bool) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_surface_rendering_activation(active));
        self.clone()
    }

    /// Activates or deactivates backface culling for the objects contained by this node and its
    /// children.
    #[inline]
    pub fn enable_backface_culling(&mut self, active: bool) -> Self {
        self.apply_to_objects_mut(&mut |o| o.enable_backface_culling(active));
        self.clone()
    }

    /// Mutably accesses the vertices of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn modify_vertices<F: FnMut(&mut Vec<Vec2>)>(&mut self, f: &mut F) {
        self.apply_to_objects_mut(&mut |o| o.modify_vertices(f))
    }

    /// Accesses the vertices of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn read_vertices<F: FnMut(&[Vec2])>(&self, f: &mut F) {
        self.apply_to_objects(&mut |o| o.read_vertices(f))
    }

    /// Mutably accesses the faces of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn modify_faces<F: FnMut(&mut Vec<[VertexIndex; 3]>)>(&mut self, f: &mut F) {
        self.apply_to_objects_mut(&mut |o| o.modify_faces(f))
    }

    /// Accesses the faces of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn read_faces<F: FnMut(&[[VertexIndex; 3]])>(&self, f: &mut F) {
        self.apply_to_objects(&mut |o| o.read_faces(f))
    }

    /// Mutably accesses the texture coordinates of the objects contained by this node and its
    /// children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn modify_uvs<F: FnMut(&mut Vec<Vec2>)>(&mut self, f: &mut F) {
        self.apply_to_objects_mut(&mut |o| o.modify_uvs(f))
    }

    /// Accesses the texture coordinates of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn read_uvs<F: FnMut(&[Vec2])>(&self, f: &mut F) {
        self.apply_to_objects(&mut |o| o.read_uvs(f))
    }

    /// Get the visibility status of node.
    #[inline]
    pub fn is_visible(&self) -> bool {
        let data = self.data();
        data.visible
    }

    /// Sets the visibility of this node.
    ///
    /// The node and its children are not rendered if it is not visible.
    #[inline]
    pub fn set_visible(&mut self, visible: bool) -> Self {
        self.data_mut().visible = visible;
        self.clone()
    }

    /// Sets the color of the objects contained by this node and its children.
    ///
    /// Colors components must be on the range `[0.0, 1.0]`.
    #[inline]
    pub fn set_color(&mut self, color: Color) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_color(color));
        self.clone()
    }

    /// Sets the texture of the objects contained by this node and its children.
    ///
    /// The texture is loaded from a file and registered by the global `TextureManager`.
    ///
    /// # Arguments
    ///   * `path` - relative path of the texture on the disk
    ///   * `name` - &str identifier to store this texture under
    #[inline]
    pub fn set_texture_from_file(&mut self, path: &Path, name: &str) -> Self {
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
    pub fn set_texture_from_memory(&mut self, image_data: &[u8], name: &str) -> Self {
        let texture =
            TextureManager::get_global_manager(|tm| tm.add_image_from_memory(image_data, name));

        self.set_texture(texture)
    }

    /// Sets the texture of the objects contained by this node and its children.
    ///
    /// The texture must already have been registered as `name`.
    #[inline]
    pub fn set_texture_with_name(&mut self, name: &str) -> Self {
        let texture = TextureManager::get_global_manager(|tm| {
            tm.get(name).unwrap_or_else(|| {
                panic!("Invalid attempt to use the unregistered texture: {}", name)
            })
        });

        self.set_texture(texture)
    }

    /// Sets the texture of the objects contained by this node and its children.
    pub fn set_texture(&mut self, texture: Arc<Texture>) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_texture(texture.clone()));
        self.clone()
    }

    /// Applies a closure to each object contained by this node and its children.
    #[inline]
    pub fn apply_to_objects_mut<F: FnMut(&mut Object2d)>(&mut self, f: &mut F) {
        let mut data = self.data_mut();
        if let Some(ref mut o) = data.object {
            f(o)
        }

        for c in data.children.iter_mut() {
            c.apply_to_objects_mut(f)
        }
    }

    /// Applies a closure to each object contained by this node and its children.
    #[inline]
    pub fn apply_to_objects<F: FnMut(&Object2d)>(&self, f: &mut F) {
        let data = self.data();
        if let Some(ref o) = data.object {
            f(o)
        }

        for c in data.children.iter() {
            c.apply_to_objects(f)
        }
    }

    // TODO: add folding?

    /// Sets the local scaling factors of the object.
    #[inline]
    pub fn set_local_scale(&mut self, sx: f32, sy: f32) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_scale = Vec2::new(sx, sy);
        drop(data);
        self.clone()
    }

    /// Returns the scaling factors of the object.
    #[inline]
    pub fn local_scale(&self) -> Vec2 {
        let data = self.data();
        data.local_scale
    }

    /// This node local transformation.
    #[inline]
    pub fn local_transformation(&self) -> Pose2 {
        let data = self.data();
        data.local_transform
    }

    /// Inverse of this node local transformation.
    #[inline]
    pub fn inverse_local_transformation(&self) -> Pose2 {
        let data = self.data();
        data.local_transform.inverse()
    }

    /// This node’s world pose (translation and rotation).
    ///
    /// This will force an update of the world transformation of its parents if they have been
    /// invalidated.
    #[inline]
    pub fn world_pose(&self) -> Pose2 {
        let mut data = self.data.borrow_mut();
        data.update();
        data.world_transform
    }

    /// This node world scale.
    ///
    /// This will force an update of the world transformation of its parents if they have been
    /// invalidated.
    #[inline]
    pub fn world_scale(&self) -> Vec2 {
        let mut data = self.data.borrow_mut();
        data.update();
        data.world_scale
    }

    /// Appends a transformation to this node local transformation.
    #[inline]
    pub fn transform(&mut self, t: Pose2) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform = t * data.local_transform;
        drop(data);
        self.clone()
    }

    /// Prepends a transformation to this node local transformation.
    #[inline]
    pub fn prepend_transform(&mut self, t: Pose2) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform *= t;
        drop(data);
        self.clone()
    }

    /// Set this node local transformation.
    #[inline]
    pub fn set_pose(&mut self, t: Pose2) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform = t;
        drop(data);
        self.clone()
    }

    /// This node local translation.
    #[inline]
    pub fn position(&self) -> Vec2 {
        let data = self.data();
        data.local_transform.translation
    }

    /// Appends a translation to this node local transformation.
    #[inline]
    pub fn translate(&mut self, t: Vec2) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform = Pose2::from_translation(t) * data.local_transform;
        drop(data);
        self.clone()
    }

    /// Prepends a translation to this node local transformation.
    #[inline]
    pub fn prepend_translation(&mut self, t: Vec2) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform *= Pose2::from_translation(t);
        drop(data);
        self.clone()
    }

    /// Sets the local translation of this node.
    #[inline]
    pub fn set_position(&mut self, t: Vec2) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform.translation = t;
        drop(data);
        self.clone()
    }

    /// This node local rotation (in radians).
    #[inline]
    pub fn rotation(&self) -> Rot2 {
        let data = self.data();
        data.local_transform.rotation
    }

    /// Appends a rotation to this node local transformation.
    #[inline]
    pub fn append_rotation(&mut self, angle: f32) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform = Pose2::rotation(angle) * data.local_transform;
        drop(data);
        self.clone()
    }

    /// Appends a rotation to this node local transformation.
    #[inline]
    pub fn rotate(&mut self, angle: f32) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform.rotation = Rot2::from_angle(angle) * data.local_transform.rotation;
        drop(data);
        self.clone()
    }

    /// Prepends a rotation to this node local transformation.
    #[inline]
    pub fn prepend_rotation(&mut self, angle: f32) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform *= Pose2::rotation(angle);
        drop(data);
        self.clone()
    }

    /// Sets the local rotation of this node (in radians).
    #[inline]
    pub fn set_rotation(&mut self, angle: f32) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform.rotation = Rot2::from_angle(angle);
        drop(data);
        self.clone()
    }

    /// Sets the instances for rendering multiple duplicates of this scene node.
    ///
    /// This only duplicates this scene node, not any of its children.
    pub fn set_instances(&mut self, instances: &[InstanceData2d]) -> Self {
        self.data_mut().get_object_mut().set_instances(instances);
        self.clone()
    }
}
