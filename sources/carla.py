import sys
import glob
import os
import time
import random
import math
import platform
import numpy as np
from settings import Config
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
        self.world.set_weather(random.choice(weather_presets))

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
        
        self.destination = random.choice(spawn_points).location
        dist = math.sqrt((self.destination.x - spawn_point.location.x)**2 + (self.destination.y - spawn_point.location.y)**2)
        
        # Keep picking a new destination until it is at least 50 meters away
        while dist < 50.0:
            self.destination = random.choice(spawn_points).location
            dist = math.sqrt((self.destination.x - spawn_point.location.x)**2 + (self.destination.y - spawn_point.location.y)**2)
        
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
        cam_bp.set_attribute('image_size_x', str(Config.IMG_WIDTH))
        cam_bp.set_attribute('image_size_y', str(Config.IMG_HEIGHT))
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
        
        nav_vector, distance_to_goal, angle_diff, cte, is_off_road, _ = self._get_navigation()
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
        # We process it (so it's ready for AI later), but we don't return it
        i = np.array(image.raw_data)

        #reshape to 80x80
        i2 = i.reshape((Config.IMG_HEIGHT, Config.IMG_WIDTH, 4))

        #drop the alpha channel
        rgb = i2[:, :, :3]

        #convert to grayscale
        grayscale = np.dot(rgb[...,:3], [0.299,0.587,0.114]) 

        #Normalize
        normalized = grayscale / 255.0

        #Add channel demension for CNN
        final_state_image = np.expand_dims(normalized,axis=-1)

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
        if not self.vehicle or not self.destination: return [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, False
        
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
        
        # Calculate Cross-Track Error (CTE) -> How far is the car from the center of the lane?
        wp_loc = current_wp.transform.location
        cte = math.sqrt((car_tf.location.x - wp_loc.x)**2 + (car_tf.location.y - wp_loc.y)**2)
        
        next_wps = current_wp.next(10.0)
        target_loc = next_wps[0].transform.location if next_wps else current_wp.transform.location

        target_angle = math.degrees(math.atan2(target_loc.y - car_tf.location.y, target_loc.x - car_tf.location.x))
        angle_diff = (target_angle - car_tf.rotation.yaw) % 360.0
        if angle_diff > 180.0: angle_diff -= 360.0
        
        v = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        return [min(dist / 500.0, 1.0), angle_diff / 180.0, min(speed / 50.0, 1.0)], dist, angle_diff, cte, is_off_road, off_road_lane_type
    
    # --- REWARD FUNCTION (New Seperate Function) ---
    def _calculate_reward(self, action):
        reward = 0.0
        steer, accel = float(action[0]), float(action[1])
        throttle = max(0.0, accel)
        brake = abs(min(0.0, accel))
        
        nav_vector, distance, angle_diff, cte, is_off_road, off_road_lane_type = self._get_navigation()
        norm_speed, angle_rad = nav_vector[2], math.radians(angle_diff)

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
        step_progress = (norm_speed * math.cos(angle_rad)) * Config.PROGRESS_REWARD_WEIGHT
        reward += step_progress
        self.last_distance = distance
        self.stats["total_progress_reward"] += step_progress
        self.stats["distance_to_goal"] = distance
        
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
            
        # Optional: Bonus for driving perfectly straight
        if abs(steer) < 0.05:
            reward += 0.5
            
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
        current_speed = nav_vector[2] * 50 # Convert normalized back to km/h
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