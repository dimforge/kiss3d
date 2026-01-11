use std::collections::HashSet;

use kiss3d::prelude::*;

const SCREEN_WIDTH: u32 = 1024;
const SCREEN_HEIGHT: u32 = 768;
const MAP_HEIGHT: usize = 24;
const MAP_WIDTH: usize = 24;

#[derive(Debug, Clone, Copy)]
struct DdaResult {
    hit: bool,
    distance: f32,
    map_x: i32,
    map_y: i32,
}

#[kiss3d::main]
async fn main() {
    // world setup
    let mut world_map = [0u8; MAP_WIDTH * MAP_HEIGHT];
    let mut player = Vec2::new(MAP_WIDTH as f32 / 2.0, MAP_HEIGHT as f32 / 2.0);

    // input handling
    let mut pressed_keys = HashSet::new();
    let mut mouse_down_left = false;
    let mut mouse_down_right = false;
    let mut mouse_pos = Vec2::ZERO;

    // timing
    let mut last_time = std::time::Instant::now();
    let mut frame_count = 0;
    let mut fps_timer = std::time::Instant::now();
    let mut fps = 0.0;

    // window, cam, scene setup
    let mut window = Window::new_with_size("Kiss3D Ray Casting", SCREEN_WIDTH, SCREEN_HEIGHT).await;
    let mut camera = FixedView2dTopLeft::default();
    let mut scene = SceneNode2d::empty();

    let window_sz = window.size();
    let cell_sz = Vec2::new(
        window_sz.x as f32 / MAP_WIDTH as f32,
        window_sz.y as f32 / MAP_HEIGHT as f32,
    );

    // use instancing for grid cells
    let mut grid_instances = Vec::with_capacity(MAP_WIDTH * MAP_HEIGHT);
    for y in 0..MAP_HEIGHT {
        for x in 0..MAP_WIDTH {
            grid_instances.push(InstanceData2d {
                position: Vec2::new(
                    x as f32 * cell_sz.x + cell_sz.x / 2.0f32,
                    y as f32 * cell_sz.y + cell_sz.y / 2.0f32,
                ),
                color: TRANSPARENT.into(),
                ..Default::default()
            });
        }
    }

    let mut grid = scene
        .add_rectangle(cell_sz.x, cell_sz.y)
        .set_instances(&grid_instances);

    // use instancing for dynamic circle drawing
    // [0] = player, [1] = mouse cursor, [2] = ray intersection
    let player_radius = (cell_sz.x.min(cell_sz.y) * 0.4f32).min(10.0f32);
    let mut circle_instances = vec![
        InstanceData2d {
            position: Vec2::ZERO,
            color: RED.into(),
            ..Default::default()
        },
        InstanceData2d {
            position: Vec2::ZERO,
            color: GREEN.into(),
            ..Default::default()
        },
        InstanceData2d {
            position: Vec2::ZERO,
            color: TRANSPARENT.into(),
            ..Default::default()
        },
    ];

    let mut circles = scene
        .add_circle(player_radius)
        .set_instances(&circle_instances);

    while window.render_2d(&mut scene, &mut camera).await {
        let now = std::time::Instant::now();
        let dt = now.duration_since(last_time).as_secs_f32();
        last_time = now;

        frame_count += 1;
        if fps_timer.elapsed().as_secs_f32() >= 1.0f32 {
            fps = frame_count as f32;
            frame_count = 0;
            fps_timer = now;
        }

        // handle events
        for event in window.events().iter() {
            match event.value {
                WindowEvent::CursorPos(x, y, _) => {
                    mouse_pos = Vec2::new(x as f32, y as f32);
                }
                WindowEvent::MouseButton(btn, action, _) => match btn {
                    MouseButton::Button1 => mouse_down_left = action == Action::Press,
                    MouseButton::Button2 => mouse_down_right = action == Action::Press,
                    _ => {}
                },
                WindowEvent::Key(key, action, _) => match action {
                    Action::Press => {
                        pressed_keys.insert(key);
                    }
                    Action::Release => {
                        pressed_keys.remove(&key);
                    }
                },
                _ => {}
            }
        }

        // handle keypresses
        let move_speed = dt * 10.0f32;

        if pressed_keys.contains(&Key::D) {
            player.x += move_speed;
        }
        if pressed_keys.contains(&Key::A) {
            player.x -= move_speed;
        }
        if pressed_keys.contains(&Key::W) {
            player.y -= move_speed;
        }
        if pressed_keys.contains(&Key::S) {
            player.y += move_speed;
        }

        if pressed_keys.contains(&Key::C) {
            world_map = [0u8; MAP_WIDTH * MAP_HEIGHT];
        }

        // convert mouse_pos to world coordinates
        let mouse_world = camera.unproject(mouse_pos, window_sz.as_vec2());
        let mouse_cell = Vec2::new(mouse_world.x / cell_sz.x, mouse_world.y / cell_sz.y);

        // place walls
        if mouse_down_left {
            let cell_x = mouse_cell.x as i32;
            let cell_y = mouse_cell.y as i32;

            if cell_x >= 0 && cell_x < MAP_WIDTH as i32 && cell_y >= 0 && cell_y < MAP_HEIGHT as i32
            {
                world_map[cell_y as usize * MAP_WIDTH + cell_x as usize] = 1;
            }
        }

        // DDA ray casting setup
        let ray_start = player;
        let ray_dir = (mouse_cell - player).normalize_or_zero();

        let dda_result = dda_ray_cast(
            ray_start,
            ray_dir,
            &world_map,
            MAP_WIDTH,
            MAP_HEIGHT,
            window.width() as f32,
        );

        let tile_found = dda_result.hit;
        let distance = dda_result.distance;
        let map_check_x = dda_result.map_x;
        let map_check_y = dda_result.map_y;

        window.set_background_color(BLACK);

        // update instanced grid
        for (i, instance) in grid_instances.iter_mut().enumerate() {
            let x = i % MAP_WIDTH;
            let y = i / MAP_WIDTH;

            instance.position = Vec2::new(
                x as f32 * cell_sz.x + cell_sz.x / 2.0f32,
                y as f32 * cell_sz.y + cell_sz.y / 2.0f32,
            );

            if world_map[y * MAP_WIDTH + x] == 1 {
                instance.color = BLUE.into();
            } else {
                instance.color = TRANSPARENT.into();
            }
        }

        grid.set_instances(&grid_instances);

        // instanced circles
        let player_screen = player * cell_sz;

        // player
        circle_instances[0].position = player_screen;

        // mouse
        circle_instances[1].position = mouse_world;

        // ray intersection circle
        if tile_found && mouse_down_right {
            let intersection = ray_start + ray_dir * distance;
            let intersection_screen = intersection * cell_sz;
            circle_instances[2].position = intersection_screen;
            circle_instances[2].color = YELLOW.into();
        } else {
            circle_instances[2].color = TRANSPARENT.into();
        }
        circles.set_instances(&circle_instances);

        // immediate mode grid lines
        for i in 0..=MAP_WIDTH {
            let x = i as f32 * cell_sz.x;
            window.draw_line_2d(
                Vec2::new(x, 0.0f32),
                Vec2::new(x, window_sz.y as f32),
                GRAY,
                1.0,
            );
        }

        for i in 0..=MAP_HEIGHT {
            let y = i as f32 * cell_sz.y;
            window.draw_line_2d(
                Vec2::new(0.0f32, y),
                Vec2::new(window_sz.x as f32, y),
                GRAY,
                1.0,
            );
        }

        // immediate mode ray line
        if mouse_down_right {
            if tile_found {
                window.draw_line_2d(player_screen, mouse_world, PINK, 2.0f32);
            } else {
                window.draw_line_2d(player_screen, mouse_pos, PINK, 2.0f32);
            }
        }

        // immediate mode text
        let fps_text = format!("FPS: {:.1}, dt: {:.3}ms", fps, dt);
        window.draw_text(
            &fps_text,
            Vec2::new(20.0f32, 20.0f32),
            30.0f32,
            &Font::default(),
            WHITE,
        );

        let debug_text = format!(
            "DDA: hit={}, dist={:.2}, cell=({},{})",
            tile_found, distance, map_check_x, map_check_y
        );
        window.draw_text(
            &debug_text,
            Vec2::new(20.0f32, 60.0f32),
            20.0f32,
            &Font::default(),
            WHITE,
        );
    }
}

fn dda_ray_cast(
    ray_start: Vec2,
    ray_dir: Vec2,
    world_map: &[u8],
    map_width: usize,
    map_height: usize,
    max_distance: f32,
) -> DdaResult {
    let (ray_dir_x, ray_dir_y) = if ray_dir.length_squared() > 0.0f32 {
        (ray_dir.x, ray_dir.y)
    } else {
        (0.0001, 0.0001)
    };

    let ray_unit_step_sz = Vec2::new(
        (1.0f32 + (ray_dir_y / ray_dir_x).powi(2)).sqrt(),
        (1.0f32 + (ray_dir_x / ray_dir_y).powi(2)).sqrt(),
    );

    let mut map_check_x = ray_start.x as i32;
    let mut map_check_y = ray_start.y as i32;

    let (step_x, mut ray_len_1d_x) = if ray_dir_x < 0.0f32 {
        (-1, (ray_start.x - map_check_x as f32) * ray_unit_step_sz.x)
    } else {
        (
            1,
            ((map_check_x + 1) as f32 - ray_start.x) * ray_unit_step_sz.x,
        )
    };

    let (step_y, mut ray_len_1d_y) = if ray_dir_y < 0.0f32 {
        (-1, (ray_start.y - map_check_y as f32) * ray_unit_step_sz.y)
    } else {
        (
            1,
            ((map_check_y + 1) as f32 - ray_start.y) * ray_unit_step_sz.y,
        )
    };

    // DDA loop
    let mut tile_found = false;
    let mut distance = 0.0f32;

    while !tile_found && distance < max_distance {
        if ray_len_1d_x < ray_len_1d_y {
            map_check_x += step_x;
            distance = ray_len_1d_x;
            ray_len_1d_x += ray_unit_step_sz.x;
        } else {
            map_check_y += step_y;
            distance = ray_len_1d_y;
            ray_len_1d_y += ray_unit_step_sz.y;
        }

        // Check bounds and collision
        if map_check_x >= 0
            && map_check_x < map_width as i32
            && map_check_y >= 0
            && map_check_y < map_height as i32
        {
            let index = map_check_y as usize * map_width + map_check_x as usize;
            if world_map[index] == 1 {
                tile_found = true;
            }
        }
    }

    DdaResult {
        hit: tile_found,
        distance,
        map_x: map_check_x,
        map_y: map_check_y,
    }
}
