import sys
import glob
import os
import time
import random
import math
import platform
import cv2
import numpy as np
from settings import Config

from navigation.global_route_planner import GlobalRoutePlanner
from navigation.global_route_planner_dao import GlobalRoutePlannerDAO
# --- 1. SETUP PATHS ---
if platform.system() == "windows":
    egg_path = r"C:\Adithyan\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.14-py3.7-win-amd64.egg"
    sys.path.append(egg_path)
else:
    print("Running on Linux/WSL")
try:
    import carla
except ImportError:
    print("ERROR: Could not load CARLA. Use 'pip install carla==0.9.14' in WSL.")
    sys.exit(1)

# --- 2. THE ENVIRONMENT WRAPPER ---
class CarlaEnv:
    def __init__(self,visualize = Config.VISUALIZE):
        print("Connecting to CARLA (127.0.0.1)...")
        self.client = carla.Client(Config.SIMULATOR_HOST, Config.SIMULATOR_PORT)
        self.client.set_timeout(Config.TIMEOUT)
        
        self.world = self.client.get_world()

        # --- NEW: OPTIMIZATION SETTINGS ---
        settings = self.world.get_settings()
        settings.no_rendering_mode = not visualize
        settings.synchronous_mode = Config.SYNC_MODE
        settings.fixed_delta_seconds = Config.FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)
        
        # Traffic Manager (Handles CPU-controlled cars)
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_global_distance_to_leading_vehicle(2.5)
        self.tm.set_hybrid_physics_mode(True) # Optimizes physics for distant cars

        # Configuration
        self.n_vehicles = Config.N_VEHICLES  # Number of other cars
        self.n_walkers = Config.N_WALKERS    # Number of pedestrians
        
        # Route Planner
        dao = GlobalRoutePlannerDAO(self.world.get_map(), sampling_resolution=2.0)
        self.route_planner = GlobalRoutePlanner(dao)
        self.route_planner.setup()
        
        # State Variables
        self.actor_list = []
        self.destination = None
        self.last_steer = 0.0
        self.data = {
            'rgb': None,
            'lidar': None,
            'gnss': None,
            'imu': None,
            'collision': [],
            'lane_invasion': []
        }
        
        self.stats = {
            "step_count": 0,
            "idle_steps" : 0,
            "total_reward": 0,
            "max_speed": 0,
            "lane_invasions": 0,
            "collision_count": 0,
            "total_progress_reward": 0, # How much it earned for moving toward goal
            "total_jerk_penalty": 0,     # How much it was punished for shaky steering
            "distance_to_goal": 0        # Final distance when episode ended
        }

    def reset(self):
        self.destroy_agents()
        # 2. Reset Data Container (DO THIS BEFORE SPAWNING SENSORS)
        self.data = {
            'rgb': None,
            'lidar': None,
            'gnss': None,
            'imu': None,
            'collision': [],
            'lane_invasion': []
        }
        self.last_steer = 0.0
        
        self.stats = {
            "step_count": 0,
            "idle_steps": 0,
            "total_reward": 0,
            "max_speed": 0,
            "lane_invasions": 0,
            "collision_count": 0,
            "total_progress_reward": 0, # How much it earned for moving toward goal
            "total_jerk_penalty": 0,     # How much it was punished for shaky steering
            "distance_to_goal": 0        # Final distance when episode ended
        }
        
        # 1. Randomize Weather (Visual Irregularities)
        # Allows AI to generalize across Sunny, Rainy, Foggy, and Night conditions
        weather_presets = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.WetNoon,
            carla.WeatherParameters.HardRainNoon,
            carla.WeatherParameters.ClearSunset,
            carla.WeatherParameters.WetCloudySunset,
            carla.WeatherParameters.MidRainSunset
        ]
        self.world.set_weather(weather_presets[0])

        # 1. Spawn Car
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()

        
        #Try spawning until successfull
        spawn_point = random.choice(spawn_points)
        self.vehicle = None
        while self.vehicle is None:
            spawn_point = random.choice(spawn_points) 
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        
        self.global_route = []
        while len(self.global_route) < 10:
            self.destination = random.choice(spawn_points).location
            dist = math.sqrt((self.destination.x - spawn_point.location.x)**2 + (self.destination.y - spawn_point.location.y)**2)
            
            # Keep picking a new destination until it is at least 50 meters away
            while dist < 50.0:
                self.destination = random.choice(spawn_points).location
                dist = math.sqrt((self.destination.x - spawn_point.location.x)**2 + (self.destination.y - spawn_point.location.y)**2)
            
            # Generate Global Route
            self.global_route = self.route_planner.trace_route(spawn_point.location, self.destination)
            
        self.current_route_index = 0
        
        self.actor_list.append(self.vehicle)
        # self.vehicle.set_autopilot(False) # Removed to prevent RPC timeouts during resets

        # 3. Spawn Traffic & Walkers (The Environment Noise)
        self._spawn_traffic(bp_lib, spawn_points)
        self._spawn_walkers(bp_lib)
        
        # 2. Get Spectator
        self.spectator = self.world.get_spectator()
        
        # 3. ATTACH SENSORS (Hidden, but active)
        # RGB Camera (Hood)
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '128')
        cam_bp.set_attribute('image_size_y', '128')
        cam_bp.set_attribute('fov', str(Config.CAM_FOV))
        cam_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera_sensor = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera_sensor)
        self.camera_sensor.listen(self._process_img)

        # LiDAR (Roof)
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(Config.LIDAR_RANGE))
        lidar_bp.set_attribute('channels', str(Config.LIDAR_CHANNELS))
        lidar_bp.set_attribute('points_per_second', str(Config.LIDAR_PPS))
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.4))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.actor_list.append(self.lidar_sensor)
        self.lidar_sensor.listen(self._process_lidar)

        # C. GNSS (GPS) IS NOT NECESSARY , WE USE CARLA XYZ TO REDUCE CALCULATION
        # gnss_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
        # self.gnss_sensor = self.world.spawn_actor(
        #     gnss_bp, carla.Transform(carla.Location(x=0, z=0)), attach_to=self.vehicle)
        # self.actor_list.append(self.gnss_sensor)
        # self.gnss_sensor.listen(lambda data: self._process_gnss(data))

        # D. IMU
        imu_bp = self.world.get_blueprint_library().find('sensor.other.imu')
        self.imu_sensor = self.world.spawn_actor(
            imu_bp, carla.Transform(carla.Location(x=0, z=0)), attach_to=self.vehicle)
        self.actor_list.append(self.imu_sensor)
        self.imu_sensor.listen(lambda data: self._process_imu(data))

        # E. Collision Sensor
        col_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self._process_collision(event))

        # F. Lane Invasion Sensor
        lane_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lane_sensor = self.world.spawn_actor(
            lane_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self._process_lane_invasion(event))

        print("Car and ALL Sensors (RGB, LiDAR, GPS, IMU, Col, Lane) spawned.")
        
        self.world.tick()
        
        # ... wait for sensors ...
        timeout_counter = 0
        while (self.data['rgb'] is None or self.data['lidar'] is None) and timeout_counter < 50:
            self.world.tick()
            time.sleep(0.05)
            timeout_counter += 1
            
        if timeout_counter >= 50:
            print("WARNING: Sensors timed out during spawn. Forcing reset.")
            return self.reset()
        
        nav_vector, distance_to_goal, angle_diff, cte, is_off_road, _, _ = self._get_navigation()
        self.last_distance = distance_to_goal  # <--- ADD THIS
        self.initial_distance = distance_to_goal
        
        imu = self.data['imu'] if self.data['imu'] else [0.0]*6
        
        # RETURN THE STATE
        return [self.data['rgb'], self.data['lidar'], nav_vector + imu]
    
    
    def _spawn_traffic(self, bp_lib, spawn_points):
        # Filter for safe cars (no trucks/bikes to reduce physics glitches)
        car_bps = [b for b in bp_lib.filter('vehicle.*') if int(b.get_attribute('number_of_wheels')) == 4]
        
        # Batch Command List
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        
        batch = []
        for _ in range(self.n_vehicles):
            bp = random.choice(car_bps)
            
            # AI Check: If blueprint has 'driver_id', set it (avoids warnings)
            if bp.has_attribute('driver_id'):
                bp.set_attribute('driver_id', 'Recommended')
            
            t = random.choice(spawn_points)
            # Create command to spawn and immediately set autopilot
            batch.append(SpawnActor(bp, t).then(SetAutopilot(FutureActor, True, self.tm.get_port())))

        # Execute all spawns in one go (Much faster than loops)
        results = self.client.apply_batch_sync(batch, True)
        
        # Track actors for cleanup
        for response in results:
            if not response.error:
                self.actor_list.append(self.world.get_actor(response.actor_id))
                
    
    def _spawn_walkers(self, bp_lib):
        # 1. Generate random locations on sidewalks
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        controller_bp = bp_lib.find('controller.ai.walker')
        
        spawn_points = []
        for i in range(self.n_walkers):
            # Find a point on the sidewalk (NavMesh)
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_points.append(carla.Transform(loc))

        # 2. Batch Spawn Walkers
        batch = []
        for t in spawn_points:
            bp = random.choice(walker_bps)
            if bp.has_attribute('is_invincible'): bp.set_attribute('is_invincible', 'false')
            batch.append(carla.command.SpawnActor(bp, t))
        
        results = self.client.apply_batch_sync(batch, True)
        walkers_list = []
        for r in results:
            if not r.error:
                walkers_list.append(r.actor_id)
                self.actor_list.append(self.world.get_actor(r.actor_id))

        # 3. Batch Spawn Controllers (The brains of the walkers)
        batch = []
        for w_id in walkers_list:
            batch.append(carla.command.SpawnActor(controller_bp, carla.Transform(), w_id))
        
        results = self.client.apply_batch_sync(batch, True)
        for r in results:
            if not r.error:
                self.actor_list.append(self.world.get_actor(r.actor_id))
                # Tell controller to start walking
                self.world.get_actor(r.actor_id).start()
                self.world.get_actor(r.actor_id).go_to_location(self.world.get_random_location_from_navigation())
                self.world.get_actor(r.actor_id).set_max_speed(1.4 + random.random()) # Random speed
    

    def _process_img(self, image):
        # image.raw_data is a flat sequence of BGRA bytes
        i = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))

        #reshape to 128x128x4 (The raw stable render size from CARLA)
        i2 = i.reshape((128, 128, 4))

        # CARLA returns BGRA. Extract BGR
        bgr = i2[:, :, :3]

        # SAFELY SHRINK TO 80x80 USING OPENCV
        bgr_shrunk = cv2.resize(bgr, (Config.IMG_WIDTH, Config.IMG_HEIGHT), interpolation=cv2.INTER_AREA)

        # Convert to Grayscale
        # Grayscale = 0.114*B + 0.587*G + 0.299*R
        grayscale = np.dot(bgr_shrunk, [0.114, 0.587, 0.299])

        # Normalize to 0.0 - 1.0
        normalized = grayscale / 255.0

        # Add channel dimension for CNN
        final_state_image = np.expand_dims(normalized, axis=-1).astype(np.float32)

        self.data['rgb'] = final_state_image


    def _process_lidar(self, data):
# 1. Access raw buffer (Float32 array of x, y, z)
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        
        # 2. Setup the Grid (80x80 Image)
        # We start with a black image (all zeros)
        lidar_img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), dtype=np.float32)
        
        # 3. Define Range
        # How far can the AI see? (e.g., 20 meters ahead, 10 meters side-to-side)
        lidar_range = Config.LIDAR_RANGE
        
        # 4. Filter Points & Map to Pixels
        # X axis (Forward) -> Image Y axis (Up/Down)
        # Y axis (Left/Right) -> Image X axis (Left/Right)
        
        # Scale: 80 pixels / (2 * lidar_range) -> Pixels per meter
        # We want the car to be at the bottom center of the image (pixel 40, 80)
        
        # Filter: Only keep points within range
        # x > 0 means in front of the car
        valid = (points[:, 0] > 0) & (points[:, 0] < lidar_range) & \
                (points[:, 1] > -lidar_range/2) & (points[:, 1] < lidar_range/2)
        
        valid_points = points[valid]
        
        # --- NEW: Extract forward obstacle distance for DENSE REWARD ---
        # Look specifically at points directly in the car's path (e.g. 1 meter left/right)
        in_path = (valid_points[:, 1] > -1.0) & (valid_points[:, 1] < 1.0)
        path_points = valid_points[in_path]
        
        if len(path_points) > 0:
            self.stats["forward_obstacle_dist"] = float(np.min(path_points[:, 0]))
        else:
            self.stats["forward_obstacle_dist"] = float(lidar_range)
        
        if len(valid_points) > 0:
            # Convert Real World Meters -> Image Pixels
            # X (Forward) maps to Y (0 is top, 80 is bottom). 
            # We want close points at bottom (80), far points at top (0).
            pixel_y = 80 - (valid_points[:, 0] / lidar_range * 80).astype(int)
            
            # Y (Left/Right) maps to X (0 is left, 80 is right).
            # -10m is left (0), +10m is right (80). Center is 40.
            pixel_x = ((valid_points[:, 1] + (lidar_range/2)) / lidar_range * 80).astype(int)
            
            # Clip values to stay within image bounds (0-79)
            pixel_y = np.clip(pixel_y, 0, 79)
            pixel_x = np.clip(pixel_x, 0, 79)
            
            # 5. Draw Obstacles
            # Set pixel to 1.0 (White) where there is a LiDAR hit
            lidar_img[pixel_y, pixel_x] = 1.0

        self.data['lidar'] = lidar_img


    # IMU Callback - PHYSICS PERCEPTION
    def _process_imu(self, data):
        # 1. Accelerometer: Measures linear force (m/s^2)
        # x = Forward/Backward acceleration (Gas/Brake)
        # y = Left/Right acceleration (Centrifugal force on turns)
        # z = Up/Down acceleration (Gravity + Bumps)
        acc = data.accelerometer
        
        # 2. Gyroscope: Measures rotation rate (rad/s)
        # x = Roll rate (Car tilting side-to-side)
        # y = Pitch rate (Car tilting nose-up/down)
        # z = Yaw rate (Car turning speed)
        gyro = data.gyroscope
        
        # 3. Processing & Normalization
        # Raw IMU data can be spiky (e.g., a crash sends acc.x to 1000).
        # We use math.tanh() to squash any number into the range [-1.0, 1.0].
        # 0.0 -> 0.0
        # 10.0 -> 1.0 (Max Forward)
        # -10.0 -> -1.0 (Max Brake)
        
        imu_vec = [
            math.tanh(acc.x),
            math.tanh(acc.y),
            math.tanh(acc.z),
            math.tanh(gyro.x),
            math.tanh(gyro.y),
            math.tanh(gyro.z)
        ]
        
        self.data['imu'] = imu_vec

    def _process_collision(self, event):
        self.data['collision'].append(event)

    def _process_lane_invasion(self, event):
        self.data['lane_invasion'].append(event)

    
    # --- GNSS PROCESSING LOGIC ---
    def _get_navigation(self):
        if not self.vehicle or not self.destination: return [0.0]*23, 0.0, 0.0, 0.0, False, "Unknown", []
        
        car_tf = self.vehicle.get_transform()
        dist = math.sqrt((self.destination.x - car_tf.location.x)**2 + (self.destination.y - car_tf.location.y)**2)
        
        # Local Waypoint Tracking
        carla_map = self.world.get_map()
        
        # Get the projected waypoint across ALL lane types
        exact_wp = carla_map.get_waypoint(car_tf.location, project_to_road=True, lane_type=carla.LaneType.Any)
        is_off_road = False
        off_road_lane_type = "Unknown"
        
        # Sidewalk, Median, Border are fatal. Shoulder is just penalized heavily by CTE without terminating.
        forbidden_lanes = [carla.LaneType.Sidewalk, carla.LaneType.Median, carla.LaneType.Border]
        
        if exact_wp is None:
            is_off_road = True
            off_road_lane_type = "Off-Map"
        else:
            # Calculate distance from car to the center of this exact snapped waypoint
            wp_loc = exact_wp.transform.location
            dist_to_wp = math.sqrt((car_tf.location.x - wp_loc.x)**2 + (car_tf.location.y - wp_loc.y)**2)
            
            # If the car is physically > 3m away from the nearest valid surface, it's out of bounds (Grass)
            if dist_to_wp > 3.0:
                is_off_road = True
                off_road_lane_type = "Grass/Terrain"
            elif exact_wp.lane_type in forbidden_lanes:
                is_off_road = True
                off_road_lane_type = exact_wp.lane_type.name
        
        # Get the projected driving waypoint for navigation so the AI always knows where it *should* be
        current_wp = carla_map.get_waypoint(car_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        # Calculate distance to nearest driving lane center (used only for off-road detection above)
        wp_loc = current_wp.transform.location
        wp_dist = math.sqrt((car_tf.location.x - wp_loc.x)**2 + (car_tf.location.y - wp_loc.y)**2)
        
        # 1. Purge passed waypoints robustly
        car_loc = car_tf.location
        f_vec = car_tf.get_forward_vector()
        r_vec = car_tf.get_right_vector()
        
        while self.current_route_index < len(self.global_route) - 1:
            wp_c = self.global_route[self.current_route_index][0]
            wp_n = self.global_route[self.current_route_index + 1][0]
            
            dx = wp_c.transform.location.x - car_loc.x
            dy = wp_c.transform.location.y - car_loc.y
            dot_forward = dx * f_vec.x + dy * f_vec.y
            
            d_curr = car_loc.distance(wp_c.transform.location)
            d_next = car_loc.distance(wp_n.transform.location)
            
            # Move to next if the current one is behind us, OR if next is closer!
            if dot_forward < -1.0 or d_next < d_curr:
                self.current_route_index += 1
            else:
                break
                
        # 2. Generate exactly 10 points spaced by 2.0m using the Global Route as an anchor guide
        waypoints_local = []
        wp = current_wp # The physically closest Driving waypoint to car
        anchor_idx = self.current_route_index
        target_loc = car_loc
        
        for i in range(10):
            next_wps = wp.next(2.0)
            if not next_wps:
                pass # Can't go further? Just repeat the last one
            elif len(next_wps) == 1:
                wp = next_wps[0]
            else:
                # Intersection! Use global_route to pick the right branch
                lookahead_idx = min(anchor_idx + 20, len(self.global_route) - 1)
                target_anchor = self.global_route[lookahead_idx][0]
                
                # Look ahead in the global route for an anchor point across the intersection
                search_end = min(anchor_idx + 50, len(self.global_route))
                for j in range(anchor_idx, search_end):
                    candidate_wp = self.global_route[j][0]
                    dist_to_cand = wp.transform.location.distance(candidate_wp.transform.location)
                    if dist_to_cand > 5.0:
                        target_anchor = candidate_wp
                        anchor_idx = j
                        break
                
                # Pick the branch that physically points toward target_anchor
                best_dist = float('inf')
                best_wp = next_wps[0]
                for branch in next_wps:
                    # Cast a ray 10 meters into the future along this branch
                    branch_future = branch.next(10.0)
                    eval_wp = branch_future[0] if branch_future else branch
                    
                    d = eval_wp.transform.location.distance(target_anchor.transform.location)
                    if d < best_dist:
                        best_dist = d
                        best_wp = branch
                wp = best_wp

            if i == 0:
                target_loc = wp.transform.location

            dx = wp.transform.location.x - car_tf.location.x
            dy = wp.transform.location.y - car_tf.location.y

            
            
            # Convert to local coordinates relative to the car's hood
            local_x = dx * f_vec.x + dy * f_vec.y
            local_y = dx * r_vec.x + dy * r_vec.y
            
            # Normalize reasonably (-1.0 to 1.0) using a 25m window to avoid squash saturation
            waypoints_local.extend([local_x / 25.0, local_y / 25.0])

        # --- Heading Alignment (used for reward only, NOT fed to the network) ---
        next_idx = min(self.current_route_index + 5, len(self.global_route)-1)
        future_wp = self.global_route[next_idx][0]
        route_wp = self.global_route[self.current_route_index][0]
        
        # Direction of the absolute route
        route_dx = future_wp.transform.location.x - route_wp.transform.location.x
        route_dy = future_wp.transform.location.y - route_wp.transform.location.y
        mag = math.sqrt(route_dx**2 + route_dy**2)
        
        if mag > 0:
            rx = route_dx / mag
            ry = route_dy / mag
        else:
            rx, ry = f_vec.x, f_vec.y

        # Heading Alignment (Dot Product) -> 1.0 is perfectly aligned
        heading_alignment = f_vec.x * rx + f_vec.y * ry

        # --- Route-Relative Signed Cross-Track Error ---
        car_to_route_dx = car_loc.x - route_wp.transform.location.x
        car_to_route_dy = car_loc.y - route_wp.transform.location.y
        route_cte = rx * car_to_route_dy - ry * car_to_route_dx
        route_cte_norm = max(-1.0, min(1.0, route_cte / 3.0))
            
        v = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        # Build the nav vector: [dist, speed, route_cte] + 20 raw (x,y) waypoint coordinates = 23 elements
        nav_vector = [
            min(dist / 500.0, 1.0), 
            min(speed / 50.0, 1.0),
            route_cte_norm
        ] + waypoints_local

        # Return heading_alignment via the angle_diff slot so reward function can use it
        return nav_vector, dist, heading_alignment, abs(route_cte), is_off_road, off_road_lane_type, waypoints_local
    
    # --- REWARD FUNCTION (New Seperate Function) ---
    def _calculate_reward(self, action):
        reward = 0.0
        steer, accel = float(action[0]), float(action[1])
        throttle = max(0.0, accel)
        brake = abs(min(0.0, accel))
        
        nav_vector, distance, heading_alignment, cte, is_off_road, off_road_lane_type, wp_xy = self._get_navigation()
        self.stats["waypoints_xy"] = wp_xy  # Send raw Cartesian targets to main.py for cv2.circle drawing
        norm_speed = nav_vector[1]

        # 1. Catastrophic Failures
        if len(self.data['collision']) > 0:
            reward += Config.REWARD_COLLISION
            self.stats["collision_count"] += 1
            self.data['collision'] = []
            return reward, True, nav_vector
            
        if is_off_road:
            reward = Config.REWARD_OFF_ROAD
            print(f"Agent drove off-road! Terminating episode... ({off_road_lane_type})")
            self.stats["off_road_lane_type"] = off_road_lane_type
            return reward, True, nav_vector

        if distance < 5.0:
            reward += Config.REWARD_GOAL_REACHED
            self.stats["distance_to_goal"] = distance
            return reward, True, nav_vector

        # 2. Local & Global Progress
        step_progress = (norm_speed * heading_alignment) * Config.PROGRESS_REWARD_WEIGHT
        reward += step_progress
        self.last_distance = distance
        self.stats["total_progress_reward"] += step_progress
        self.stats["distance_to_goal"] = distance

        # Direct speed bonus - always reward ANY forward movement to prevent idle collapse
        #reward += norm_speed * 1.0
        
        # --- NEW: DENSE OBSTACLE AVOIDANCE REWARD ---
        # If an obstacle is directly in front of us, penalize high speeds!
        min_dist = self.stats.get("forward_obstacle_dist", Config.LIDAR_RANGE)
        danger_zone = 7.0 # meters
        
        if min_dist < danger_zone:
            # The closer the object, the higher the danger multiplier (0.0 to 1.0)
            danger_multiplier = 1.0 - (min_dist / danger_zone)
            # The faster we are going, the harder we are penalized.
            # If we are stopped (norm_speed = 0), penalty is 0, which teaches the AI TO STOP!
            proximity_penalty = danger_multiplier * norm_speed * 10.0
            reward -= proximity_penalty
            self.stats["proximity_penalty"] = proximity_penalty
        else:
            self.stats["proximity_penalty"] = 0.0

        # Optional: Extra bonus for applying brakes when an object is close
        if min_dist < danger_zone and brake > 0.1:
            reward += brake * danger_multiplier * 2.0
        
        # 3. Lane Centering Penalty (Cross-Track Error)
        # Penalize the agent the further it gets from the center of the lane
        if cte > 0.5: # 0.5 meters of 'wiggle room'
            reward -= (cte * Config.CTE_PENALTY_WEIGHT)
            
        reward -= 0.1 # Step penalty

        if len(self.data['lane_invasion']) > 0:
            reward += Config.REWARD_LANE_INVASION
            self.stats["lane_invasions"] += 1
            self.data['lane_invasion'] = []

        # --- THE FIX: The Idle Timeout ---
        # We no longer give an instant massive negative reward every single step.
        # Instead, we just track idle steps, and terminate gracefully with a penalty if stuck too long.
        if norm_speed <= 0.05:
            self.stats["idle_steps"] += 1
            reward -= 5.0 # High per-step penalty, to prevent delaying the terminal penalty
        else:
            self.stats["idle_steps"] = 0 # Reset if it moves!

        # If it sits still for 100 steps, KILL the episode with a moderate penalty
        if self.stats["idle_steps"] >= 100:
            # BUG FIX: Ensure the catastrophic idle penalty is actually applied
            reward = Config.REWARD_IDLE 
            print("AI Paralyzed! Terminating episode...")
            return reward, True, nav_vector

        # Comfort penalties
        jerk = abs(steer - self.last_steer)
        if jerk > 0.1:
            j_pen = Config.REWARD_JERK_PENALTY * jerk
            reward += j_pen
            self.stats["total_jerk_penalty"] += j_pen
        self.last_steer = steer
        
        
        reward += (steer ** 2) *  Config.REWARD_SWERVE
        
        # --- NEW: The Spinning Tax ---
        # Heavily penalize high steering angles if the car is barely moving forward
        if abs(steer) > 0.3 and norm_speed < 0.2:
            reward -= 2.0  # Flat penalty for doing low-speed donuts
            
        # Penalize braking when going fast
        reward += (brake * norm_speed) * Config.REWARD_HARD_BREAK

        return reward, False, nav_vector

    def step(self, action):
        # --- SPECTATOR LOGIC ---
        if self.vehicle and self.vehicle.is_alive:
            car_tf = self.vehicle.get_transform()
            yaw = math.radians(car_tf.rotation.yaw)
            target_loc = carla.Location(
                x=car_tf.location.x - 10 * math.cos(yaw),
                y=car_tf.location.y - 10 * math.sin(yaw),
                z=car_tf.location.z + 5.0
            )

            current_tf = self.spectator.get_transform()

            #smoothing factor
            alpha = 1.0
            smooth_loc = carla.Location(
                x=current_tf.location.x * (1 - alpha) + target_loc.x * alpha,
                y=current_tf.location.y * (1 - alpha) + target_loc.y * alpha,
                z=current_tf.location.z * (1 - alpha) + target_loc.z * alpha
            )

            cam_rot = carla.Rotation(pitch=-20, yaw=car_tf.rotation.yaw, roll=0)

            self.spectator.set_transform(carla.Transform(smooth_loc, cam_rot))

        # --- CONTROL LOGIC ---
        steer = float(action[0])
        accel = float(action[1])
        
        # Action space mapping: accel > 0 is throttle, accel < 0 is brake
        throttle = max(0.0, accel)
        brake = abs(min(0.0, accel))
            
        self.vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle, brake=brake))
        #self.vehicle.set_autopilot(True)
        
        self.world.tick()
        
        #Calculate Reward & Done using helper functions
        # BUG FIX: the reward function expects the 2D action [steer, accel] now.
        reward, done, nav_vector = self._calculate_reward([steer, accel])
        
        # UPDATE STATS
        self.stats["step_count"] += 1
        self.stats["total_reward"] += reward
        current_speed = nav_vector[1] * 50 # Convert normalized back to km/h
        if current_speed > self.stats["max_speed"]:
            self.stats["max_speed"] = current_speed
            
            
        # 2. IMU (6 values: Accel X/Y/Z, Gyro X/Y/Z)
        # If IMU is somehow None (rare), provide zeros
        imu_vector = self.data['imu'] if self.data['imu'] is not None else [0.0]*6

        full_vector_state = nav_vector + imu_vector

        #RETURN [Camera(Image),Lidar(Image),Physics(Vector)]
        return [self.data['rgb'],self.data['lidar'],full_vector_state], reward, done, self.stats

    def destroy_agents(self):
        print("Cleaning up actors...")
        if self.world is not None:
            #fix for disconnection issue
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
            # 1. Stop sensors safely before deleting
            for actor in self.actor_list:
                if actor is not None and actor.is_alive:
                    if actor.type_id.startswith('sensor'):
                        actor.stop()
            
            # 2. Batch destroy everything instantly using Sync
            # Using apply_batch_sync ensures the server actually deletes them before we proceed,
            # preventing memory leaks and zombie actors from crashing the Unreal Engine backend.
            batch = [carla.command.DestroyActor(x) for x in self.actor_list if x is not None]
            
            # Send the batch command to the server synchronously
            self.client.apply_batch_sync(batch)
            
            self.actor_list.clear()

            #wait half a second for windows to flush RAM and stuff
            time.sleep(0.5)
            
            # Tick the world once without sync to allow the Unreal Engine to process the deletion frames
            self.world.tick()
            
            #restore sync mode for next episode
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
                
        print("Cleanup done.")