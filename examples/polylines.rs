//! N-Body gravitational simulation with polyline trails
//!
//! This example demonstrates the polyline rendering capability by simulating
//! a gravitational N-body system where each body leaves a colored trail.
//! Based on bevy_polyline's nbody example.
//!
//! Use the UI panel to toggle perspective mode for the polylines.

extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::camera::ArcBall;
use kiss3d::light::Light;
use kiss3d::renderer::Polyline;
use kiss3d::window::Window;
use na::{Point3, Vector3};

// Simulation parameters
const NUM_BODIES: usize = 128;
const TRAIL_LENGTH: usize = 256;
const G: f32 = 1.0; // Gravitational constant (scaled for visibility)
const EPSILON: f32 = 0.5; // Softening parameter to prevent singularities
const DT: f32 = 0.01; // Time step
const SUBSTEPS: usize = 2; // Physics substeps per frame

/// A body in the simulation
struct Body {
    position: Vector3<f32>,
    velocity: Vector3<f32>,
    mass: f32,
}

/// Trail storing position history as a ring buffer
struct Trail {
    positions: Vec<Point3<f32>>,
    head: usize,
    len: usize,
}

impl Trail {
    fn new() -> Self {
        Self {
            positions: vec![Point3::origin(); TRAIL_LENGTH],
            head: 0,
            len: 0,
        }
    }

    fn push(&mut self, pos: Point3<f32>) {
        self.positions[self.head] = pos;
        self.head = (self.head + 1) % TRAIL_LENGTH;
        if self.len < TRAIL_LENGTH {
            self.len += 1;
        }
    }

    /// Copy trail points into destination vector (avoids allocation)
    fn copy_to(&self, dest: &mut Vec<Point3<f32>>) {
        dest.clear();
        if self.len == 0 {
            return;
        }

        // Start from the oldest point
        let start = if self.len < TRAIL_LENGTH {
            0
        } else {
            self.head
        };

        for i in 0..self.len {
            let idx = (start + i) % TRAIL_LENGTH;
            dest.push(self.positions[idx]);
        }
    }
}

/// HSL to RGB conversion
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Point3<f32> {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let m = l - c / 2.0;

    let (r, g, b) = if h_prime < 1.0 {
        (c, x, 0.0)
    } else if h_prime < 2.0 {
        (x, c, 0.0)
    } else if h_prime < 3.0 {
        (0.0, c, x)
    } else if h_prime < 4.0 {
        (0.0, x, c)
    } else if h_prime < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    Point3::new(r + m, g + m, b + m)
}

/// Simple pseudo-random number generator (xorshift)
struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        (self.state as f32) / (u32::MAX as f32)
    }
}

/// Initialize bodies in a disk configuration with orbital velocities
fn init_bodies() -> Vec<Body> {
    use std::f32::consts::TAU;

    let mut rng = Rng::new(42);
    let mut bodies = Vec::with_capacity(NUM_BODIES);

    // Add a central massive body
    bodies.push(Body {
        position: Vector3::zeros(),
        velocity: Vector3::zeros(),
        mass: 100.0,
    });

    // Add orbiting bodies in a disk
    for _ in 1..NUM_BODIES {
        let angle = rng.next_f32() * TAU;
        let radius = 1.0 + rng.next_f32() * 8.0;
        let height = (rng.next_f32() - 0.5) * 1.0;

        let position = Vector3::new(radius * angle.cos(), height, radius * angle.sin());

        // Calculate orbital velocity for roughly circular orbit
        let orbital_speed = (G * 100.0 / radius).sqrt() * (0.9 + rng.next_f32() * 0.2);
        let tangent = Vector3::new(-angle.sin(), 0.0, angle.cos());
        let velocity = tangent * orbital_speed;

        let mass = 0.01 + rng.next_f32() * 0.1;

        bodies.push(Body {
            position,
            velocity,
            mass,
        });
    }

    bodies
}

/// Initialize polylines with colors and widths for each body
fn init_polylines() -> Vec<Polyline> {
    let mut rng = Rng::new(42);
    let mut polylines = Vec::with_capacity(NUM_BODIES);

    // Central body (yellow star)
    polylines.push(
        Polyline::new(Vec::with_capacity(TRAIL_LENGTH))
            .with_color(1.0, 1.0, 0.5)
            .with_width(3.0),
    );

    // Orbiting bodies - need to advance RNG to match init_bodies
    for i in 1..NUM_BODIES {
        // Skip the same random values used in init_bodies
        let _ = rng.next_f32(); // angle
        let _ = rng.next_f32(); // radius
        let _ = rng.next_f32(); // height
        let _ = rng.next_f32(); // orbital_speed variation
        let _ = rng.next_f32(); // mass

        // Use index to spread hue across spectrum
        let hue = (i as f32 / NUM_BODIES as f32) * 360.0;
        let color = hsl_to_rgb(hue, 1.0, 0.6);
        // Vary line width dramatically (0.5 to 10 pixels)
        let line_width = 0.5 + rng.next_f32() * 9.5;

        polylines.push(
            Polyline::new(Vec::with_capacity(TRAIL_LENGTH))
                .with_color(color.x, color.y, color.z)
                .with_width(line_width),
        );
    }

    polylines
}

/// Compute gravitational acceleration on body i from all other bodies
fn compute_acceleration(bodies: &[Body], i: usize) -> Vector3<f32> {
    let mut acceleration = Vector3::zeros();
    let pos_i = bodies[i].position;

    for (j, body_j) in bodies.iter().enumerate() {
        if i == j {
            continue;
        }

        let offset = body_j.position - pos_i;
        let dist_sq = offset.norm_squared() + EPSILON * EPSILON;
        let dist = dist_sq.sqrt();
        let force_mag = G * body_j.mass / dist_sq;
        acceleration += offset / dist * force_mag;
    }

    acceleration
}

/// Perform one physics step using velocity Verlet integration
fn physics_step(bodies: &mut [Body], dt: f32) {
    let n = bodies.len();

    // Store accelerations
    let accelerations: Vec<Vector3<f32>> = (0..n)
        .map(|i| compute_acceleration(bodies, i))
        .collect();

    // Update positions
    for (i, body) in bodies.iter_mut().enumerate() {
        body.position += body.velocity * dt + accelerations[i] * (0.5 * dt * dt);
    }

    // Compute new accelerations
    let new_accelerations: Vec<Vector3<f32>> = (0..n)
        .map(|i| compute_acceleration(bodies, i))
        .collect();

    // Update velocities
    for (i, body) in bodies.iter_mut().enumerate() {
        body.velocity += (accelerations[i] + new_accelerations[i]) * (0.5 * dt);
    }
}

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Kiss3d: N-Body Polyline Simulation").await;

    window.set_light(Light::StickToCamera);
    window.set_background_color(0.02, 0.02, 0.05);

    // Set up camera looking at the simulation (user can control it)
    let eye = Point3::new(0.0, 15.0, 25.0);
    let at = Point3::origin();
    let mut camera = ArcBall::new(eye, at);

    // Initialize simulation
    let mut bodies = init_bodies();
    let mut trails: Vec<Trail> = (0..NUM_BODIES).map(|_| Trail::new()).collect();

    // Create polylines once with pre-allocated capacity
    let mut polylines = init_polylines();

    // UI state
    let mut perspective_mode = false;

    while window.render_with_camera(&mut camera).await {
        // Draw UI
        #[cfg(feature = "egui")]
        window.draw_ui(|ctx| {
            egui::Window::new("Settings")
                .default_pos([10.0, 10.0])
                .show(ctx, |ui| {
                    ui.checkbox(&mut perspective_mode, "Perspective mode");
                    ui.label("When enabled, lines get thinner with distance");
                });
        });

        // Physics simulation with substeps
        let sub_dt = DT / SUBSTEPS as f32;
        for _ in 0..SUBSTEPS {
            physics_step(&mut bodies, sub_dt);
        }

        // Update trails with current positions
        for (i, body) in bodies.iter().enumerate() {
            trails[i].push(Point3::from(body.position));
        }

        // Draw trails as polylines (reusing pre-allocated polylines)
        for (i, trail) in trails.iter().enumerate() {
            // Copy trail points into polyline's vertex buffer
            trail.copy_to(&mut polylines[i].vertices);
            polylines[i].perspective = perspective_mode;

            if polylines[i].vertices.len() >= 2 {
                window.draw_polyline(&polylines[i]);
            }
        }
    }
}
