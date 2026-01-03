use crate::camera::{Camera3d};
use crate::light::{CollectedLight, Light, LightCollection, LightType, MAX_LIGHTS};
use crate::procedural;
use crate::procedural::RenderMesh;
use crate::resource::vertex_index::VertexIndex;
use crate::resource::{
    GpuMesh3d, Material3d, MaterialManager3d, MeshManager3d, RenderContext, Texture, TextureManager,
};
use crate::scene::{InstanceData3d, Object3d};
use glamx::{Pose3, Quat, Vec2, Vec3};
#[cfg(feature = "parry")]
use parry3d::shape::TriMesh;
use std::cell::{Ref, RefCell, RefMut};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::rc::Weak;
use std::sync::Arc;
use crate::color::Color;

/// The data contained by a `SceneNode`.
pub struct SceneNodeData3d {
    local_scale: Vec3,
    local_transform: Pose3,
    world_scale: Vec3,
    world_transform: Pose3,
    visible: bool,
    up_to_date: bool,
    children: Vec<SceneNode3d>,
    object: Option<Object3d>,
    light: Option<Light>,
    parent: Option<Weak<RefCell<SceneNodeData3d>>>,
}

/// A node of the scene graph.
///
/// This may represent a group of other nodes, and/or contain an object that can be rendered.
#[derive(Clone)]
pub struct SceneNode3d {
    data: Rc<RefCell<SceneNodeData3d>>,
}

impl SceneNodeData3d {
    // XXX: Because `node.borrow_mut().parent = Some(self.data.downgrade())`
    // causes a weird compiler error:
    //
    // ```
    // error: mismatched types: expected `&std::cell::RefCell<scene::scene_node::SceneNodeData>`
    // but found
    // `std::option::Option<std::rc::Weak<std::cell::RefCell<scene::scene_node::SceneNodeData>>>`
    // (expe cted &-ptr but found enum std::option::Option)
    // ```
    fn set_parent(&mut self, parent: Weak<RefCell<SceneNodeData3d>>) {
        self.parent = Some(parent);
    }

    // TODO: this exists because of a similar bug as `set_parent`.
    fn remove_from_parent(&mut self, to_remove: &SceneNode3d) {
        let _ = self.parent.as_ref().map(|p| {
            if let Some(bp) = p.upgrade() {
                bp.borrow_mut().remove(to_remove);
            }
        });
    }

    fn remove(&mut self, o: &SceneNode3d) {
        if let Some(i) = self
            .children
            .iter()
            .rposition(|e| std::ptr::eq(&*o.data, &*e.data))
        {
            let _ = self.children.swap_remove(i);
        }
    }

    /// Whether this node contains an `Object`.
    #[inline]
    pub fn has_object(&self) -> bool {
        self.object.is_some()
    }

    /// Whether this node has no parent.
    #[inline]
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// Prepare uniforms for the scene graph rooted by this node.
    ///
    /// This is the first phase of two-phase rendering. It traverses the scene
    /// graph twice: first to collect all lights, then to call `Material::prepare()`
    /// for each object with the complete light collection.
    pub fn prepare(
        &mut self,
        pass: usize,
        camera: &mut dyn Camera3d,
        lights: &mut LightCollection,
        viewport_width: u32,
        viewport_height: u32,
    ) {
        // Pass 0: Collect all lights and update transforms
        self.do_propagate_transforms(Pose3::IDENTITY, Vec3::ONE);

        if self.visible {
            // Pass 1: Collect all lights and update transforms
            self.do_collect_lights(lights);

            // Pass 2: Prepare all objects with the complete light collection
            self.do_prepare_objects(pass, camera, lights, viewport_width, viewport_height);
        }
    }

    fn do_propagate_transforms(
        &mut self,
        transform: Pose3,
        scale: Vec3,
    ) {
        if !self.up_to_date {
            self.up_to_date = true;
            self.world_transform = transform * self.local_transform;
            self.world_scale = scale * self.local_scale;
        }

        // Recurse to children
        for c in self.children.iter_mut() {
            let mut bc = c.data_mut();
            bc.do_propagate_transforms(self.world_transform, self.world_scale);
        }
    }

    /// First pass: update transforms and collect all lights from the scene tree.
    fn do_collect_lights(
        &mut self,
        lights: &mut LightCollection,
    ) {
        // Collect light if present and enabled
        if let Some(ref light) = self.light {
            if light.enabled && lights.lights.len() < MAX_LIGHTS {
                let local_direction = match light.light_type {
                    LightType::Directional(direction) => direction.normalize_or(Vec3::NEG_Z),
                    _ => Vec3::NEG_Z,
                };

                lights.add(CollectedLight {
                    light_type: light.light_type.clone(),
                    color: Vec3::new(light.color.r, light.color.g, light.color.b),
                    intensity: light.intensity,
                    world_position: self.world_transform.translation,
                    world_direction: self.world_transform.rotation * local_direction,
                });
            }
        }



        // Recurse to children
        for c in self.children.iter_mut() {
            let mut bc = c.data_mut();
            if bc.visible {
                bc.do_collect_lights(lights);
            }
        }
    }

    /// Second pass: prepare all objects with the complete light collection.
    fn do_prepare_objects(
        &mut self,
        pass: usize,
        camera: &mut dyn Camera3d,
        lights: &LightCollection,
        viewport_width: u32,
        viewport_height: u32,
    ) {
        // Prepare this node's object
        if let Some(ref mut o) = self.object {
            o.prepare(
                self.world_transform,
                self.world_scale,
                pass,
                camera,
                lights,
                viewport_width,
                viewport_height,
            );
        }

        // Recurse to children
        for c in self.children.iter_mut() {
            let mut bc = c.data_mut();
            if bc.visible {
                bc.do_prepare_objects(pass, camera, lights, viewport_width, viewport_height);
            }
        }
    }

    /// Render the scene graph rooted by this node.
    pub fn render(
        &mut self,
        pass: usize,
        camera: &mut dyn Camera3d,
        lights: &LightCollection,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext,
    ) {
        if self.visible {
            self.do_render(
                pass,
                camera,
                lights,
                render_pass,
                context,
            )
        }
    }

    fn do_render(
        &mut self,
        pass: usize,
        camera: &mut dyn Camera3d,
        lights: &LightCollection,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext,
    ) {
        if let Some(ref mut o) = self.object {
            o.render(
                self.world_transform,
                self.world_scale,
                pass,
                camera,
                lights,
                render_pass,
                context,
            )
        }

        for c in self.children.iter_mut() {
            let mut bc = c.data_mut();
            if bc.visible {
                bc.do_render(
                    pass,
                    camera,
                    lights,
                    render_pass,
                    context,
                )
            }
        }
    }

    /// A reference to the object possibly contained by this node.
    #[inline]
    pub fn object(&self) -> Option<&Object3d> {
        self.object.as_ref()
    }

    /// A mutable reference to the object possibly contained by this node.
    #[inline]
    pub fn object_mut(&mut self) -> Option<&mut Object3d> {
        self.object.as_mut()
    }

    /// A reference to the object possibly contained by this node.
    ///
    /// # Failure
    /// Fails of this node does not contains an object.
    #[inline]
    pub fn get_object(&self) -> &Object3d {
        self.object()
            .expect("This scene node does not contain an Object.")
    }

    /// A mutable reference to the object possibly contained by this node.
    ///
    /// # Failure
    /// Fails of this node does not contains an object.
    // TODO: this method should return `Option`, whereas `object_mut` is the one
    //       that should return the naked ref.
    #[inline]
    pub fn get_object_mut(&mut self) -> &mut Object3d {
        self.object_mut()
            .expect("This scene node does not contain an Object.")
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

impl Default for SceneNode3d {
    fn default() -> SceneNode3d {
        SceneNode3d::empty()
    }
}

impl SceneNode3d {
    /// Creates a new unrooted scene node with the specified properties.
    ///
    /// # Arguments
    /// * `local_scale` - The initial scale factors along each axis
    /// * `local_transform` - The initial local transformation (rotation + translation)
    /// * `object` - Optional object to render (None for empty group nodes)
    ///
    /// # Returns
    /// A new `SceneNode` without a parent
    pub fn new(local_scale: Vec3, local_transform: Pose3, object: Option<Object3d>) -> SceneNode3d {
        let data = SceneNodeData3d {
            local_scale,
            local_transform,
            world_transform: local_transform,
            world_scale: local_scale,
            visible: true,
            up_to_date: false,
            children: Vec::new(),
            object,
            light: None,
            parent: None,
        };

        SceneNode3d {
            data: Rc::new(RefCell::new(data)),
        }
    }

    /// Creates a new empty scene node with identity transformations.
    ///
    /// The node has no parent, no object, unit scale, and identity transformation.
    ///
    /// # Returns
    /// A new empty `SceneNode`
    pub fn empty() -> SceneNode3d {
        SceneNode3d::new(Vec3::ONE, Pose3::IDENTITY, None)
    }

    // ==================
    // Primitive constructors
    // ==================

    /// Creates a new scene node with a cube mesh.
    ///
    /// The cube is initially axis-aligned and centered at (0, 0, 0).
    ///
    /// # Arguments
    /// * `wx` - the cube extent along the x axis
    /// * `wy` - the cube extent along the y axis
    /// * `wz` - the cube extent along the z axis
    pub fn cube(wx: f32, wy: f32, wz: f32) -> SceneNode3d {
        Self::geom_with_name("cube", Vec3::new(wx, wy, wz))
            .expect("Unable to load the default cube geometry.")
    }

    /// Creates a new scene node with a sphere mesh.
    ///
    /// The sphere is initially centered at (0, 0, 0).
    ///
    /// # Arguments
    /// * `r` - the sphere radius
    pub fn sphere(r: f32) -> SceneNode3d {
        Self::geom_with_name("sphere", Vec3::new(r * 2.0, r * 2.0, r * 2.0))
            .expect("Unable to load the default sphere geometry.")
    }

    /// Creates a new scene node with a sphere mesh with custom subdivisions.
    ///
    /// The sphere is initially centered at (0, 0, 0).
    ///
    /// # Arguments
    /// * `r` - the sphere radius
    /// * `ntheta_subdiv` - number of subdivisions around the sphere (longitude)
    /// * `nphi_subdiv` - number of subdivisions from top to bottom (latitude)
    pub fn sphere_with_subdiv(r: f32, ntheta_subdiv: u32, nphi_subdiv: u32) -> SceneNode3d {
        Self::render_mesh(
            procedural::sphere(r * 2.0, ntheta_subdiv, nphi_subdiv, true),
            Vec3::ONE,
        )
    }

    /// Creates a new scene node with a cone mesh.
    ///
    /// The cone is initially centered at (0, 0, 0) and points toward the positive `y` axis.
    ///
    /// # Arguments
    /// * `r` - the cone base radius
    /// * `h` - the cone height
    pub fn cone(r: f32, h: f32) -> SceneNode3d {
        Self::geom_with_name("cone", Vec3::new(r * 2.0, h, r * 2.0))
            .expect("Unable to load the default cone geometry.")
    }

    /// Creates a new scene node with a cone mesh with custom subdivisions.
    ///
    /// The cone is initially centered at (0, 0, 0) and points toward the positive `y` axis.
    ///
    /// # Arguments
    /// * `r` - the cone base radius
    /// * `h` - the cone height
    /// * `nsubdiv` - number of subdivisions around the base circle
    pub fn cone_with_subdiv(r: f32, h: f32, nsubdiv: u32) -> SceneNode3d {
        Self::render_mesh(procedural::cone(r * 2.0, h, nsubdiv), Vec3::ONE)
    }

    /// Creates a new scene node with a cylinder mesh.
    ///
    /// The cylinder is initially centered at (0, 0, 0) and has its principal axis
    /// aligned with the `y` axis.
    ///
    /// # Arguments
    /// * `r` - the cylinder base radius
    /// * `h` - the cylinder height
    pub fn cylinder(r: f32, h: f32) -> SceneNode3d {
        Self::geom_with_name("cylinder", Vec3::new(r * 2.0, h, r * 2.0))
            .expect("Unable to load the default cylinder geometry.")
    }

    /// Creates a new scene node with a cylinder mesh with custom subdivisions.
    ///
    /// The cylinder is initially centered at (0, 0, 0) and has its principal axis
    /// aligned with the `y` axis.
    ///
    /// # Arguments
    /// * `r` - the cylinder base radius
    /// * `h` - the cylinder height
    /// * `nsubdiv` - number of subdivisions around the circumference
    pub fn cylinder_with_subdiv(r: f32, h: f32, nsubdiv: u32) -> SceneNode3d {
        Self::render_mesh(procedural::cylinder(r * 2.0, h, nsubdiv), Vec3::ONE)
    }

    /// Creates a new scene node with a capsule mesh.
    ///
    /// The capsule is initially centered at (0, 0, 0) and has its principal axis
    /// aligned with the `y` axis.
    ///
    /// # Arguments
    /// * `r` - the capsule caps radius
    /// * `h` - the capsule height
    pub fn capsule(r: f32, h: f32) -> SceneNode3d {
        Self::render_mesh(procedural::capsule(r * 2.0, h, 50, 50), Vec3::ONE)
    }

    /// Creates a new scene node with a capsule mesh with custom subdivisions.
    ///
    /// The capsule is initially centered at (0, 0, 0) and has its principal axis
    /// aligned with the `y` axis.
    ///
    /// # Arguments
    /// * `r` - the capsule caps radius
    /// * `h` - the capsule height
    /// * `ntheta_subdiv` - number of subdivisions around the capsule (longitude)
    /// * `nphi_subdiv` - number of subdivisions along each hemisphere (latitude)
    pub fn capsule_with_subdiv(r: f32, h: f32, ntheta_subdiv: u32, nphi_subdiv: u32) -> SceneNode3d {
        Self::render_mesh(
            procedural::capsule(r * 2.0, h, ntheta_subdiv, nphi_subdiv),
            Vec3::ONE,
        )
    }

    /// Creates a new scene node with a double-sided quad mesh.
    ///
    /// The quad is initially centered at (0, 0, 0). The quad itself is composed of a
    /// user-defined number of triangles regularly spaced on a grid.
    ///
    /// # Arguments
    /// * `w` - the quad width.
    /// * `h` - the quad height.
    /// * `usubdivs` - number of horizontal subdivisions.
    /// * `vsubdivs` - number of vertical subdivisions.
    pub fn quad(w: f32, h: f32, usubdivs: usize, vsubdivs: usize) -> SceneNode3d {
        let mut node = Self::render_mesh(procedural::quad(w, h, usubdivs, vsubdivs), Vec3::ONE);
        node.enable_backface_culling(false);
        node
    }

    /// Creates a new scene node with a double-sided quad with the specified vertices.
    pub fn quad_with_vertices(vertices: &[Vec3], nhpoints: usize, nvpoints: usize) -> SceneNode3d {
        let geom = procedural::quad_with_vertices(vertices, nhpoints, nvpoints);
        let mut node = Self::render_mesh(geom, Vec3::ONE);
        node.enable_backface_culling(false);
        node
    }

    /// Creates a new scene node using the geometry registered as `geometry_name`.
    pub fn geom_with_name(geometry_name: &str, scale: Vec3) -> Option<SceneNode3d> {
        MeshManager3d::get_global_manager(|mm| mm.get(geometry_name)).map(|g| Self::mesh(g, scale))
    }

    /// Creates a new scene node using a mesh.
    pub fn mesh(mesh: Rc<RefCell<GpuMesh3d>>, scale: Vec3) -> SceneNode3d {
        let tex = TextureManager::get_global_manager(|tm| tm.get_default());
        let mat = MaterialManager3d::get_global_manager(|mm| mm.get_default());
        let object = Object3d::new(mesh, crate::color::WHITE, tex, mat);

        SceneNode3d::new(scale, Pose3::IDENTITY, Some(object))
    }

    /// Creates a new scene node using a mesh descriptor.
    pub fn render_mesh(mesh: RenderMesh, scale: Vec3) -> SceneNode3d {
        Self::mesh(
            Rc::new(RefCell::new(GpuMesh3d::from_render_mesh(mesh, false))),
            scale,
        )
    }

    /// Creates a new scene node using a parry3d TriMesh.
    ///
    /// Requires the `parry` feature.
    #[cfg(feature = "parry")]
    pub fn trimesh(mesh: TriMesh, scale: Vec3, flat_normals: bool) -> SceneNode3d {
        let mut render_mesh = RenderMesh::from(mesh);
        if flat_normals {
            render_mesh.replicate_vertices();
            render_mesh.recompute_normals();
        }

        Self::mesh(
            Rc::new(RefCell::new(GpuMesh3d::from_render_mesh(render_mesh, false))),
            scale,
        )
    }

    // ==================
    // Light constructors
    // ==================

    /// Creates a new scene node with a light.
    ///
    /// The light's position and direction are determined by the node's world transform.
    ///
    /// # Arguments
    /// * `light` - The light configuration
    pub fn new_light(light: Light) -> SceneNode3d {
        let mut node = SceneNode3d::empty();
        node.data_mut().light = Some(light);
        node
    }

    /// Creates a new scene node with a point light.
    ///
    /// # Arguments
    /// * `attenuation_radius` - Maximum distance the light affects
    pub fn new_point_light(attenuation_radius: f32) -> SceneNode3d {
        Self::new_light(Light::point(attenuation_radius))
    }

    /// Creates a new scene node with a directional light.
    ///
    /// The light direction is determined by the node's rotation (forward is -Z).
    ///
    /// # Arguments
    /// * `direction` - The light direction
    pub fn new_directional_light(direction: Vec3) -> SceneNode3d {
        Self::new_light(Light::directional(direction))
    }

    /// Creates a new scene node with a spot light.
    ///
    /// # Arguments
    /// * `inner_cone_angle` - Inner cone angle in radians (full intensity)
    /// * `outer_cone_angle` - Outer cone angle in radians (fades to zero)
    /// * `attenuation_radius` - Maximum distance the light affects
    pub fn new_spot_light(inner_cone_angle: f32, outer_cone_angle: f32, attenuation_radius: f32) -> SceneNode3d {
        Self::new_light(Light::spot(inner_cone_angle, outer_cone_angle, attenuation_radius))
    }

    /// Removes this node from its parent in the scene graph.
    ///
    /// This is an alias for [`Self::remove`].
    pub fn detach(&mut self) {
        self.remove();
    }

    /// Removes this node from its parent in the scene graph.
    ///
    /// After calling this, the node becomes unrooted and will no longer be rendered
    /// as part of the scene hierarchy.
    pub fn remove(&mut self) {
        let self_self = self.clone();
        self.data_mut().remove_from_parent(&self_self);
        self.data_mut().parent = None
    }

    /// Returns an immutable reference to this node's internal data.
    ///
    /// # Returns
    /// A `Ref` guard to the `SceneNodeData`
    pub fn data(&self) -> Ref<'_, SceneNodeData3d> {
        self.data.borrow()
    }

    /// Returns a mutable reference to this node's internal data.
    ///
    /// # Returns
    /// A `RefMut` guard to the `SceneNodeData`
    pub fn data_mut(&mut self) -> RefMut<'_, SceneNodeData3d> {
        self.data.borrow_mut()
    }

    /*
     *
     * Methods to add objects.
     *
     */
    /// Adds an empty group node as a child of this node.
    ///
    /// A group is a node without any renderable object, useful for organizing hierarchies.
    ///
    /// # Returns
    /// The newly created child `SceneNode`
    pub fn add_group(&mut self) -> SceneNode3d {
        let node = SceneNode3d::empty();

        self.add_child(node.clone());

        node
    }

    /// Adds an existing node as a child of this node.
    ///
    /// # Arguments
    /// * `node` - The node to add as a child
    ///
    /// # Panics
    /// Panics if the node already has a parent
    pub fn add_child(&mut self, node: SceneNode3d) {
        assert!(
            node.data().is_root(),
            "The added node must not have a parent yet."
        );

        let mut node = node;
        let self_weak_ptr = Rc::downgrade(&self.data);
        node.data_mut().set_parent(self_weak_ptr);
        self.data_mut().children.push(node)
    }

    /// Adds a new node with a renderable object as a child of this node.
    ///
    /// # Arguments
    /// * `local_scale` - Scale factors for the new node
    /// * `local_transform` - Local transformation for the new node
    /// * `object` - The object to render
    ///
    /// # Returns
    /// The newly created child `SceneNode`
    pub fn add_object(
        &mut self,
        local_scale: Vec3,
        local_transform: Pose3,
        object: Object3d,
    ) -> SceneNode3d {
        let node = SceneNode3d::new(local_scale, local_transform, Some(object));

        self.add_child(node.clone());

        node
    }

    /// Adds a cube as a children of this node. The cube is initially axis-aligned and centered
    /// at (0, 0, 0).
    ///
    /// # Arguments
    /// * `wx` - the cube extent along the x axis
    /// * `wy` - the cube extent along the y axis
    /// * `wz` - the cube extent along the z axis
    pub fn add_cube(&mut self, wx: f32, wy: f32, wz: f32) -> SceneNode3d {
        let node = Self::cube(wx, wy, wz);
        self.add_child(node.clone());
        node
    }

    /// Adds a sphere as a children of this node. The sphere is initially centered at (0, 0, 0).
    ///
    /// # Arguments
    /// * `r` - the sphere radius
    pub fn add_sphere(&mut self, r: f32) -> SceneNode3d {
        let node = Self::sphere(r);
        self.add_child(node.clone());
        node
    }

    /// Adds a sphere with custom subdivisions as a child of this node.
    ///
    /// The sphere is initially centered at (0, 0, 0).
    ///
    /// # Arguments
    /// * `r` - the sphere radius
    /// * `ntheta_subdiv` - number of subdivisions around the sphere (longitude)
    /// * `nphi_subdiv` - number of subdivisions from top to bottom (latitude)
    pub fn add_sphere_with_subdiv(
        &mut self,
        r: f32,
        ntheta_subdiv: u32,
        nphi_subdiv: u32,
    ) -> SceneNode3d {
        let node = Self::sphere_with_subdiv(r, ntheta_subdiv, nphi_subdiv);
        self.add_child(node.clone());
        node
    }

    /// Adds a cone to the scene. The cone is initially centered at (0, 0, 0) and points toward the
    /// positive `y` axis.
    ///
    /// # Arguments
    /// * `h` - the cone height
    /// * `r` - the cone base radius
    pub fn add_cone(&mut self, r: f32, h: f32) -> SceneNode3d {
        let node = Self::cone(r, h);
        self.add_child(node.clone());
        node
    }

    /// Adds a cone with custom subdivisions to the scene.
    ///
    /// The cone is initially centered at (0, 0, 0) and points toward the positive `y` axis.
    ///
    /// # Arguments
    /// * `r` - the cone base radius
    /// * `h` - the cone height
    /// * `nsubdiv` - number of subdivisions around the base circle
    pub fn add_cone_with_subdiv(&mut self, r: f32, h: f32, nsubdiv: u32) -> SceneNode3d {
        let node = Self::cone_with_subdiv(r, h, nsubdiv);
        self.add_child(node.clone());
        node
    }

    /// Adds a cylinder to this node children. The cylinder is initially centered at (0, 0, 0)
    /// and has its principal axis aligned with the `y` axis.
    ///
    /// # Arguments
    /// * `h` - the cylinder height
    /// * `r` - the cylinder base radius
    pub fn add_cylinder(&mut self, r: f32, h: f32) -> SceneNode3d {
        let node = Self::cylinder(r, h);
        self.add_child(node.clone());
        node
    }

    /// Adds a cylinder with custom subdivisions to this node children.
    ///
    /// The cylinder is initially centered at (0, 0, 0) and has its principal axis
    /// aligned with the `y` axis.
    ///
    /// # Arguments
    /// * `r` - the cylinder base radius
    /// * `h` - the cylinder height
    /// * `nsubdiv` - number of subdivisions around the circumference
    pub fn add_cylinder_with_subdiv(&mut self, r: f32, h: f32, nsubdiv: u32) -> SceneNode3d {
        let node = Self::cylinder_with_subdiv(r, h, nsubdiv);
        self.add_child(node.clone());
        node
    }

    /// Adds a capsule to this node children. The capsule is initially centered at (0, 0, 0) and
    /// has its principal axis aligned with the `y` axis.
    ///
    /// # Arguments
    /// * `h` - the capsule height
    /// * `r` - the capsule caps radius
    pub fn add_capsule(&mut self, r: f32, h: f32) -> SceneNode3d {
        let node = Self::capsule(r, h);
        self.add_child(node.clone());
        node
    }

    /// Adds a capsule with custom subdivisions to this node children.
    ///
    /// The capsule is initially centered at (0, 0, 0) and has its principal axis
    /// aligned with the `y` axis.
    ///
    /// # Arguments
    /// * `r` - the capsule caps radius
    /// * `h` - the capsule height
    /// * `ntheta_subdiv` - number of subdivisions around the capsule (longitude)
    /// * `nphi_subdiv` - number of subdivisions along each hemisphere (latitude)
    pub fn add_capsule_with_subdiv(
        &mut self,
        r: f32,
        h: f32,
        ntheta_subdiv: u32,
        nphi_subdiv: u32,
    ) -> SceneNode3d {
        let node = Self::capsule_with_subdiv(r, h, ntheta_subdiv, nphi_subdiv);
        self.add_child(node.clone());
        node
    }

    /// Adds a double-sided quad to this node children. The quad is initially centered at (0, 0,
    /// 0). The quad itself is composed of a user-defined number of triangles regularly spaced on a
    /// grid. This is the main way to draw height maps.
    ///
    /// # Arguments
    /// * `w` - the quad width.
    /// * `h` - the quad height.
    /// * `wsubdivs` - number of horizontal subdivisions. This correspond to the number of squares
    ///   which will be placed horizontally on each line. Must not be `0`.
    /// * `hsubdivs` - number of vertical subdivisions. This correspond to the number of squares
    ///   which will be placed vertically on each line. Must not be `0`.
    ///   update.
    pub fn add_quad(&mut self, w: f32, h: f32, usubdivs: usize, vsubdivs: usize) -> SceneNode3d {
        let node = Self::quad(w, h, usubdivs, vsubdivs);
        self.add_child(node.clone());
        node
    }

    /// Adds a double-sided quad with the specified vertices.
    pub fn add_quad_with_vertices(
        &mut self,
        vertices: &[Vec3],
        nhpoints: usize,
        nvpoints: usize,
    ) -> SceneNode3d {
        let node = Self::quad_with_vertices(vertices, nhpoints, nvpoints);
        self.add_child(node.clone());
        node
    }

    /// Creates and adds a new object using the geometry registered as `geometry_name`.
    pub fn add_geom_with_name(&mut self, geometry_name: &str, scale: Vec3) -> Option<SceneNode3d> {
        Self::geom_with_name(geometry_name, scale).inspect(|node| {
            self.add_child(node.clone());
        })
    }

    /// Creates and adds a new object to this node children using a mesh.
    pub fn add_mesh(&mut self, mesh: Rc<RefCell<GpuMesh3d>>, scale: Vec3) -> SceneNode3d {
        let node = Self::mesh(mesh, scale);
        self.add_child(node.clone());
        node
    }

    /// Creates and adds a new object using a mesh descriptor.
    pub fn add_render_mesh(&mut self, mesh: RenderMesh, scale: Vec3) -> SceneNode3d {
        let node = Self::render_mesh(mesh, scale);
        self.add_child(node.clone());
        node
    }

    /// Creates and adds a new object using a parry3d TriMesh.
    ///
    /// Requires the `parry` feature.
    #[cfg(feature = "parry")]
    pub fn add_trimesh(&mut self, mesh: TriMesh, scale: Vec3, flat_normals: bool) -> SceneNode3d {
        let node = Self::trimesh(mesh, scale, flat_normals);
        self.add_child(node.clone());
        node
    }

    /// Creates and adds multiple nodes created from an obj file.
    ///
    /// This will create a new node serving as a root of the scene described by the obj file. This
    /// newly created node is added to this node's children.
    pub fn add_obj(&mut self, path: &Path, mtl_dir: &Path, scale: Vec3) -> SceneNode3d {
        let tex = TextureManager::get_global_manager(|tm| tm.get_default());
        let mat = MaterialManager3d::get_global_manager(|mm| mm.get_default());

        // TODO: is there some error-handling stuff to do here instead of the `let _`.
        let result = MeshManager3d::load_obj(path, mtl_dir, path.to_str().unwrap()).map(|objs| {
            let mut root;

            let self_root = objs.len() == 1;
            let child_scale;

            if self_root {
                root = self.clone();
                child_scale = scale;
            } else {
                root = SceneNode3d::new(scale, Pose3::IDENTITY, None);
                self.add_child(root.clone());
                child_scale = Vec3::ONE;
            }

            for (_, mesh, mtl) in objs.into_iter() {
                let mut object = Object3d::new(mesh, crate::color::WHITE, tex.clone(), mat.clone());

                match mtl {
                    None => {}
                    Some(mtl) => {
                        object.set_color(Color::new(
                            mtl.diffuse[0],
                            mtl.diffuse[1],
                            mtl.diffuse[2],
                            1.0,
                        ));

                        for t in mtl.diffuse_texture.iter() {
                            let mut tpath = PathBuf::new();
                            tpath.push(mtl_dir);
                            tpath.push(&t[..]);
                            object.set_texture_from_file(&tpath, tpath.to_str().unwrap())
                        }

                        for t in mtl.ambient_texture.iter() {
                            let mut tpath = PathBuf::new();
                            tpath.push(mtl_dir);
                            tpath.push(&t[..]);
                            object.set_texture_from_file(&tpath, tpath.to_str().unwrap())
                        }
                    }
                }

                let _ = root.add_object(child_scale, Pose3::IDENTITY, object);
            }

            if self_root {
                root.data()
                    .children
                    .last()
                    .expect("There was nothing on this obj file.")
                    .clone()
            } else {
                root
            }
        });

        result.unwrap()
    }

    /// Applies a closure to each object contained by this node and its children.
    #[inline]
    pub fn apply_to_scene_nodes_mut<F: FnMut(&mut SceneNode3d)>(&mut self, f: &mut F) {
        f(self);

        for c in self.data_mut().children.iter_mut() {
            c.apply_to_scene_nodes_mut(f)
        }
    }

    /// Applies a closure to each object contained by this node and its children.
    #[inline]
    pub fn apply_to_scene_nodes<F: FnMut(&SceneNode3d)>(&self, f: &mut F) {
        f(self);

        for c in self.data().children.iter() {
            c.apply_to_scene_nodes(f)
        }
    }

    // TODO: for all those set_stuff, would it be more per formant to add a special case for when
    // we are on a leaf? (to avoid the call to a closure required by the apply_to_*).
    /// Sets the material for all objects in this node and its children.
    ///
    /// The material defines how the object is shaded (shader program and uniforms).
    ///
    /// # Arguments
    /// * `material` - The material to apply
    #[inline]
    pub fn set_material(&mut self, material: Rc<RefCell<Box<dyn Material3d + 'static>>>) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_material(material.clone()));
        self.clone()
    }

    /// Sets the material by name for all objects in this node and its children.
    ///
    /// The material must have been previously registered with the global material manager.
    ///
    /// # Arguments
    /// * `name` - The name of the registered material
    ///
    /// # Panics
    /// Panics if the material with the given name doesn't exist
    #[inline]
    pub fn set_material_with_name(&mut self, name: &str) -> Self {
        let material = MaterialManager3d::get_global_manager(|tm| {
            tm.get(name).unwrap_or_else(|| {
                panic!("Invalid attempt to use the unregistered material: {}", name)
            })
        });

        self.set_material(material)
    }

    /// Sets the line width for wireframe rendering of objects in this node and its children.
    ///
    /// # Arguments
    /// * `width` - The line width
    /// * `use_perspective` - If true, width is in world units and scales with distance.
    ///   If false, width is in screen pixels and stays constant.
    #[inline]
    pub fn set_lines_width(&mut self, width: f32, use_perspective: bool) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_lines_width(width, use_perspective));
        self.clone()
    }

    /// Sets the line color for wireframe rendering of objects in this node and its children.
    ///
    /// # Arguments
    /// * `color` - The RGBA color for lines, or `None` to use the object's default color
    #[inline]
    pub fn set_lines_color(&mut self, color: Option<Color>) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_lines_color(color));
        self.clone()
    }

    /// Sets the point size for point cloud rendering of objects in this node and its children.
    ///
    /// # Arguments
    /// * `size` - The point size
    /// * `use_perspective` - If true, size is in world units and scales with distance.
    ///   If false, size is in screen pixels and stays constant.
    #[inline]
    pub fn set_points_size(&mut self, size: f32, use_perspective: bool) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_points_size(size, use_perspective));
        self.clone()
    }

    /// Sets the point color for point cloud rendering of objects in this node and its children.
    ///
    /// # Arguments
    /// * `color` - The RGBA color for points, or `None` to use the object's default color
    #[inline]
    pub fn set_points_color(&mut self, color: Option<Color>) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_points_color(color));
        self.clone()
    }

    /// Enables or disables surface rendering for objects in this node and its children.
    ///
    /// When disabled, only wireframe and points are rendered.
    ///
    /// # Arguments
    /// * `active` - `true` to enable surface rendering, `false` to disable it
    #[inline]
    pub fn set_surface_rendering_activation(&mut self, active: bool) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_surface_rendering_activation(active));
        self.clone()
    }

    /// Enables or disables backface culling for objects in this node and its children.
    ///
    /// Backface culling improves performance by not rendering triangles facing away from the camera.
    ///
    /// # Arguments
    /// * `active` - `true` to enable backface culling, `false` to disable it
    #[inline]
    pub fn enable_backface_culling(&mut self, active: bool) -> Self {
        self.apply_to_objects_mut(&mut |o| o.enable_backface_culling(active));
        self.clone()
    }

    /// Mutably accesses the vertices of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn modify_vertices<F: FnMut(&mut Vec<Vec3>)>(&mut self, f: &mut F) {
        self.apply_to_objects_mut(&mut |o| o.modify_vertices(f))
    }

    /// Accesses the vertices of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn read_vertices<F: FnMut(&[Vec3])>(&self, f: &mut F) {
        self.apply_to_objects(&mut |o| o.read_vertices(f))
    }

    /// Recomputes the normals of the meshes of the objects contained by this node and its
    /// children.
    #[inline]
    pub fn recompute_normals(&mut self) {
        self.apply_to_objects_mut(&mut |o| o.recompute_normals())
    }

    /// Mutably accesses the normals of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn modify_normals<F: FnMut(&mut Vec<Vec3>)>(&mut self, f: &mut F) {
        self.apply_to_objects_mut(&mut |o| o.modify_normals(f))
    }

    /// Accesses the normals of the objects contained by this node and its children.
    ///
    /// The provided closure is called once per object.
    #[inline(always)]
    pub fn read_normals<F: FnMut(&[Vec3])>(&self, f: &mut F) {
        self.apply_to_objects(&mut |o| o.read_normals(f))
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
    pub fn set_color(&mut self, color: crate::color::Color) -> Self {
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

    // === PBR Material Properties ===

    /// Sets the metallic factor for all objects in this node and its children.
    ///
    /// # Arguments
    /// * `metallic` - Metallic factor [0.0, 1.0] where 0.0 is dielectric and 1.0 is metal
    #[inline]
    pub fn set_metallic(&mut self, metallic: f32) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_metallic(metallic));
        self.clone()
    }

    /// Sets the roughness factor for all objects in this node and its children.
    ///
    /// # Arguments
    /// * `roughness` - Roughness factor [0.0, 1.0] where 0.0 is smooth and 1.0 is rough
    #[inline]
    pub fn set_roughness(&mut self, roughness: f32) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_roughness(roughness));
        self.clone()
    }

    /// Sets the emissive color for all objects in this node and its children.
    ///
    /// # Arguments
    /// * `color` - RGBA emissive color
    #[inline]
    pub fn set_emissive(&mut self, color: crate::color::Color) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_emissive(color));
        self.clone()
    }

    // === PBR Texture Maps ===

    /// Sets the normal map for all objects in this node and its children.
    #[inline]
    pub fn set_normal_map(&mut self, texture: Arc<Texture>) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_normal_map(texture.clone()));
        self.clone()
    }

    /// Sets the normal map from a file for all objects in this node and its children.
    #[inline]
    pub fn set_normal_map_from_file(&mut self, path: &Path, name: &str) -> Self {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));
        self.set_normal_map(texture)
    }

    /// Clears the normal map for all objects in this node and its children.
    #[inline]
    pub fn clear_normal_map(&mut self) -> Self {
        self.apply_to_objects_mut(&mut |o| o.clear_normal_map());
        self.clone()
    }

    /// Sets the metallic-roughness map for all objects in this node and its children.
    #[inline]
    pub fn set_metallic_roughness_map(&mut self, texture: Arc<Texture>) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_metallic_roughness_map(texture.clone()));
        self.clone()
    }

    /// Sets the metallic-roughness map from a file for all objects in this node and its children.
    #[inline]
    pub fn set_metallic_roughness_map_from_file(&mut self, path: &Path, name: &str) -> Self {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));
        self.set_metallic_roughness_map(texture)
    }

    /// Clears the metallic-roughness map for all objects in this node and its children.
    #[inline]
    pub fn clear_metallic_roughness_map(&mut self) -> Self {
        self.apply_to_objects_mut(&mut |o| o.clear_metallic_roughness_map());
        self.clone()
    }

    /// Sets the ambient occlusion map for all objects in this node and its children.
    #[inline]
    pub fn set_ao_map(&mut self, texture: Arc<Texture>) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_ao_map(texture.clone()));
        self.clone()
    }

    /// Sets the ambient occlusion map from a file for all objects in this node and its children.
    #[inline]
    pub fn set_ao_map_from_file(&mut self, path: &Path, name: &str) -> Self {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));
        self.set_ao_map(texture)
    }

    /// Clears the ambient occlusion map for all objects in this node and its children.
    #[inline]
    pub fn clear_ao_map(&mut self) -> Self {
        self.apply_to_objects_mut(&mut |o| o.clear_ao_map());
        self.clone()
    }

    /// Sets the emissive map for all objects in this node and its children.
    #[inline]
    pub fn set_emissive_map(&mut self, texture: Arc<Texture>) -> Self {
        self.apply_to_objects_mut(&mut |o| o.set_emissive_map(texture.clone()));
        self.clone()
    }

    /// Sets the emissive map from a file for all objects in this node and its children.
    #[inline]
    pub fn set_emissive_map_from_file(&mut self, path: &Path, name: &str) -> Self {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));
        self.set_emissive_map(texture)
    }

    /// Clears the emissive map for all objects in this node and its children.
    #[inline]
    pub fn clear_emissive_map(&mut self) -> Self {
        self.apply_to_objects_mut(&mut |o| o.clear_emissive_map());
        self.clone()
    }

    /// Applies a closure to each object contained by this node and its children.
    #[inline]
    pub fn apply_to_objects_mut<F: FnMut(&mut Object3d)>(&mut self, f: &mut F) {
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
    pub fn apply_to_objects<F: FnMut(&Object3d)>(&self, f: &mut F) {
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
    pub fn set_local_scale(&mut self, sx: f32, sy: f32, sz: f32) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_scale = Vec3::new(sx, sy, sz);
        drop(data);
        self.clone()
    }

    /// Returns the scaling factors of the object.
    #[inline]
    pub fn local_scale(&self) -> Vec3 {
        let data = self.data();
        data.local_scale
    }

    /// Move and orient the object such that it is placed at the point `eye` and have its `z` axis
    /// oriented toward `at`.
    #[inline]
    pub fn reorient(&mut self, eye: Vec3, at: Vec3, up: Vec3) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform = Pose3::face_towards(eye, at, up);
        drop(data);
        self.clone()
    }

    /// This node local transformation.
    #[inline]
    pub fn local_transformation(&self) -> Pose3 {
        let data = self.data();
        data.local_transform
    }

    /// Inverse of this node local transformation.
    #[inline]
    pub fn inverse_local_transformation(&self) -> Pose3 {
        let data = self.data();
        data.local_transform.inverse()
    }

    /// This node world transformation.
    ///
    /// This will force an update of the world transformation of its parents if they have been
    /// invalidated.
    #[inline]
    pub fn world_pose(&self) -> Pose3 {
        let mut data = self.data.borrow_mut();
        data.update();
        data.world_transform
    }

    /// This node world scale.
    ///
    /// This will force an update of the world transformation of its parents if they have been
    /// invalidated.
    #[inline]
    pub fn world_scale(&self) -> Vec3 {
        let mut data = self.data.borrow_mut();
        data.update();
        data.world_scale
    }

    /// Appends a transformation to this node's local transformation.
    ///
    /// The transformation is applied before the current local transformation.
    ///
    /// # Arguments
    /// * `t` - The transformation to append (combines rotation and translation)
    #[inline]
    pub fn transform(&mut self, t: Pose3) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform = t * data.local_transform;
        drop(data);
        self.clone()
    }

    /// Prepends a transformation to this node's local transformation.
    ///
    /// The transformation is applied after the current local transformation.
    ///
    /// # Arguments
    /// * `t` - The transformation to prepend (combines rotation and translation)
    #[inline]
    pub fn prepend_transform(&mut self, t: Pose3) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform *= t;
        drop(data);
        self.clone()
    }

    /// Sets this node's local transformation, replacing the current one.
    ///
    /// # Arguments
    /// * `t` - The new local transformation (combines rotation and translation)
    #[inline]
    pub fn set_pose(&mut self, t: Pose3) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform = t;
        drop(data);
        self.clone()
    }

    /// Returns this node's local translation component.
    ///
    /// # Returns
    /// The translation relative to the parent node (or world origin if root)
    #[inline]
    pub fn position(&self) -> Vec3 {
        let data = self.data();
        data.local_transform.translation
    }

    /// Returns the inverse of this node's local translation.
    ///
    /// # Returns
    /// The inverse translation
    #[inline]
    pub fn inverse_position(&self) -> Vec3 {
        let data = self.data();
        -data.local_transform.translation
    }

    /// Appends a translation to this node's local transformation.
    ///
    /// The translation is applied before the current rotation and translation.
    ///
    /// # Arguments
    /// * `t` - The translation to append
    #[inline]
    pub fn translate(&mut self, t: Vec3) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform.translation += t;
        drop(data);
        self.clone()
    }

    /// Prepends a translation to this node's local transformation.
    ///
    /// The translation is applied after the current rotation and translation.
    ///
    /// # Arguments
    /// * `t` - The translation to prepend
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::scene::SceneNode3d;
    /// # use glamx::Vec3;
    /// let mut scene = SceneNode3d::empty();
    /// let mut cube = scene.add_cube(1.0, 1.0, 1.0);
    /// // Move the cube 0.1 units along the x-axis each frame
    /// cube.prepend_translation(Vec3::new(0.1, 0.0, 0.0));
    /// ```
    #[inline]
    pub fn prepend_translation(&mut self, t: Vec3) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform = data.local_transform.prepend_translation(t);
        drop(data);
        self.clone()
    }

    /// Sets this node's local translation, replacing the current one.
    ///
    /// # Arguments
    /// * `t` - The new local translation
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::scene::SceneNode3d;
    /// # use glamx::Vec3;
    /// let mut scene = SceneNode3d::empty();
    /// let mut cube = scene.add_cube(1.0, 1.0, 1.0);
    /// // Position the cube at (5, 0, -10)
    /// cube.set_position(Vec3::new(5.0, 0.0, -10.0));
    /// ```
    #[inline]
    pub fn set_position(&mut self, t: Vec3) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform.translation = t;
        drop(data);
        self.clone()
    }

    /// Returns this node's local rotation component.
    ///
    /// # Returns
    /// The rotation as a unit quaternion, relative to the parent node
    #[inline]
    pub fn rotation(&self) -> Quat {
        let data = self.data();
        data.local_transform.rotation
    }

    /// Returns the inverse of this node's local rotation.
    ///
    /// # Returns
    /// The inverse rotation
    #[inline]
    pub fn inverse_rotation(&self) -> Quat {
        let data = self.data();
        data.local_transform.rotation.conjugate()
    }

    /// Appends a rotation to this node's local transformation.
    ///
    /// The rotation is applied before the current transformation.
    ///
    /// # Arguments
    /// * `r` - The rotation to append (as a unit quaternion)
    #[inline]
    pub fn append_rotation(&mut self, r: Quat) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform = Pose3::from(r) * data.local_transform;
        drop(data);
        self.clone()
    }

    /// Appends a rotation to this node's local transformation, rotating around the object's center.
    ///
    /// Unlike [`append_rotation`](Self::append_rotation), this rotates the object in place
    /// rather than rotating it around the origin.
    ///
    /// # Arguments
    /// * `r` - The rotation to append (as a unit quaternion)
    #[inline]
    pub fn rotate(&mut self, r: Quat) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform.rotation = r * data.local_transform.rotation;
        drop(data);
        self.clone()
    }

    /// Prepends a rotation to this node's local transformation.
    ///
    /// The rotation is applied after the current transformation.
    ///
    /// # Arguments
    /// * `r` - The rotation to prepend (as a unit quaternion)
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::scene::SceneNode3d;
    /// # use glamx::{Quat, Vec3};
    /// let mut scene = SceneNode3d::empty();
    /// let mut cube = scene.add_cube(1.0, 1.0, 1.0);
    /// // Rotate the cube around the Y axis by 0.014 radians each frame
    /// let rot = Quat::from_axis_angle(Vec3::Y, 0.014);
    /// cube.prepend_rotation(rot);
    /// ```
    #[inline]
    pub fn prepend_rotation(&mut self, r: Quat) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform.rotation *= r;
        drop(data);
        self.clone()
    }

    /// Sets this node's local rotation, replacing the current one.
    ///
    /// # Arguments
    /// * `r` - The new local rotation (as a unit quaternion)
    #[inline]
    pub fn set_rotation(&mut self, r: Quat) -> Self {
        let mut data = self.data_mut();
        data.invalidate();
        data.local_transform.rotation = r;
        drop(data);
        self.clone()
    }

    /// Prepare uniforms for the scene graph rooted by this node.
    ///
    /// This is the first phase of two-phase rendering.
    pub fn prepare(
        &mut self,
        pass: usize,
        camera: &mut dyn Camera3d,
        lights: &mut LightCollection,
        viewport_width: u32,
        viewport_height: u32,
    ) {
        self.data_mut()
            .prepare(pass, camera, lights, viewport_width, viewport_height)
    }

    /// Render the scene graph rooted by this node.
    pub fn render(
        &mut self,
        pass: usize,
        camera: &mut dyn Camera3d,
        lights: &LightCollection,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &RenderContext,
    ) {
        self.data_mut()
            .render(pass, camera, lights, render_pass, context)
    }

    // ==================
    // Light methods
    // ==================

    /// Adds a light to this node as a child and returns the new node.
    ///
    /// The light's position and direction are determined by the node's world transform.
    ///
    /// # Arguments
    /// * `light_config` - The light configuration
    ///
    /// # Returns
    /// A new scene node containing the light
    pub fn add_light(&mut self, light_config: Light) -> SceneNode3d {
        let node = Self::new_light(light_config);
        self.add_child(node.clone());
        node
    }

    /// Adds a point light as a child of this node.
    ///
    /// # Arguments
    /// * `attenuation_radius` - Maximum distance the light affects
    ///
    /// # Returns
    /// A new scene node containing the point light
    pub fn add_point_light(&mut self, attenuation_radius: f32) -> SceneNode3d {
        let node = Self::new_point_light(attenuation_radius);
        self.add_child(node.clone());
        node
    }

    /// Adds a directional light as a child of this node.
    ///
    /// The light direction is determined by the node's rotation (forward is -Z).
    ///
    /// # Returns
    /// A new scene node containing the directional light
    pub fn add_directional_light(&mut self, direction: Vec3) -> SceneNode3d {
        let node = Self::new_directional_light(direction);
        self.add_child(node.clone());
        node
    }

    /// Adds a spot light as a child of this node.
    ///
    /// # Arguments
    /// * `inner_cone_angle` - Inner cone angle in radians (full intensity)
    /// * `outer_cone_angle` - Outer cone angle in radians (fades to zero)
    /// * `attenuation_radius` - Maximum distance the light affects
    ///
    /// # Returns
    /// A new scene node containing the spot light
    pub fn add_spot_light(
        &mut self,
        inner_cone_angle: f32,
        outer_cone_angle: f32,
        attenuation_radius: f32,
    ) -> SceneNode3d {
        let node = Self::new_spot_light(inner_cone_angle, outer_cone_angle, attenuation_radius);
        self.add_child(node.clone());
        node
    }

    /// Sets the light on this node.
    ///
    /// Pass `None` to remove the light.
    pub fn set_light(&mut self, light: Option<Light>) -> Self {
        self.data_mut().light = light;
        self.clone()
    }

    /// Returns a reference to the light on this node, if any.
    pub fn light(&self) -> Option<Light> {
        self.data().light.clone()
    }

    /// Modifies the light on this node.
    ///
    /// The closure is called only if the node has a light.
    pub fn modify_light<F: FnOnce(&mut Light)>(&mut self, f: F) {
        if let Some(ref mut light) = self.data_mut().light {
            f(light);
        }
    }

    /// Sets the instances for rendering multiple duplicates of this scene node.
    ///
    /// This only duplicates this scene node, not any of its children.
    pub fn set_instances(&mut self, instances: &[InstanceData3d]) -> Self {
        self.data_mut().get_object_mut().set_instances(instances);
        self.clone()
    }
}
