import sys
import glob
import os
import time
import random
import math
import numpy as np
from settings import Config
# --- 1. SETUP PATHS ---
egg_path = r"C:\Adithyan\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.14-py3.7-win-amd64.egg"
sys.path.append(egg_path)

try:
    import carla
except ImportError:
    print("ERROR: Python could not load the CARLA library.")
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
        self.actor_list.append(self.vehicle)
        self.vehicle.set_autopilot(False) 

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
        while self.data['rgb'] is None or self.data['lidar'] is None:
            self.world.tick()
            time.sleep(0.01)
            
        # GET INITIAL STATE
        nav, _, _ = self._get_navigation()
        imu = self.data['imu'] if self.data['imu'] else [0.0]*6
        
        # RETURN THE STATE (Not None)
        return [self.data['rgb'], self.data['lidar'], nav + imu]
    
    
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
        if not self.vehicle or not self.destination:
            return [0.0, 0.0, 0.0] , 0.0 , 0.0

        # 1. Get current car location (The "Processed GPS")
        current_loc = self.vehicle.get_location()
        dest_loc = self.destination

        # 2. Calculate Vector to Destination
        dx = dest_loc.x - current_loc.x
        dy = dest_loc.y - current_loc.y
        
        # 3. Distance
        distance = math.sqrt(dx**2 + dy**2)
        
        # 4. Angle (bearing) to destination
        target_angle = math.degrees(math.atan2(dy, dx))
        
        # 5. Car's current yaw
        car_yaw = self.vehicle.get_transform().rotation.yaw
        
        # 6. Relative Angle (Error)
        # 0 = Facing destination directly
        # -ve = Dest is to the left, +ve = Dest is to the right
        angle_diff = (target_angle - car_yaw) % 360.0
        if angle_diff > 180.0: angle_diff -= 360.0
        
        # Return [Distance, Angle, Speed]
        v = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        norm_dist = min(distance / 500.0, 1.0)
        norm_angle = angle_diff / 180.0
        norm_speed = min(speed / 50.0, 1.0) # 50 km/h max
        
        # Return: [AI Vector], Raw Dist, Raw Angle
        return [norm_dist, norm_angle, norm_speed] , distance , angle_diff


    # --- REWARD FUNCTION (New Seperate Function) ---
    def _calculate_reward(self, action):
        reward = 0
        done = False
        
        # 1. Collision (Critical Penalty)
        if len(self.data['collision']) > 0:
            reward += Config.REWARD_COLLISION
            done = True
            self.data['collision'] = []

        # 2. Lane Invasion (Minor Penalty)
        if len(self.data['lane_invasion']) > 0:
            reward += Config.REWARD_LANE_INVASION
            self.data['lane_invasion'] = []

        # 3. Progress Reward (Speed * Angle Match)
        nav_vector, distance, angle_diff = self._get_navigation()
        norm_speed = nav_vector[2]
        
        angle_rad = math.radians(angle_diff)
        progress_reward = norm_speed * math.cos(angle_rad)
        reward += progress_reward * Config.PROGRESS_REWARD_WEIGHT
        
        # 5. GOAL REACHED (New Use for Distance!)
        if distance < 5.0:  # If within 5 meters of target
            print("DESTINATION REACHED!")
            reward += Config.REWARD_GOAL_REACHED # Big Success Reward
            done = True

        # 4. Jerky Driving Penalty
        steer = float(action[0])
        jerk = abs(steer - self.last_steer)
        if jerk > 0.5:
            reward += Config.REWARD_JERK_PENALTY
        self.last_steer = steer

        return reward, done, nav_vector

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
        # steer = float(action[0])
        # throttle = float(action[1])
        # brake = float(action[2])
        #self.vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle, brake=brake))
        self.vehicle.set_autopilot(True)
        
        self.world.tick()
        
        #Calculate Reward & Done using helper functions
        reward, done, nav_vector = self._calculate_reward(action)
        
        # 2. IMU (6 values: Accel X/Y/Z, Gyro X/Y/Z)
        # If IMU is somehow None (rare), provide zeros
        imu_vector = self.data['imu'] if self.data['imu'] is not None else [0.0]*6

        full_vector_state = nav_vector + imu_vector

        #RETURN [Camera(Image),Lidar(Image),Physics(Vector)]
        return [self.data['rgb'],self.data['lidar'],full_vector_state], reward, done, None

    def destroy_agents(self):
        print("Cleaning up actors...")
        for actor in self.actor_list:
            if hasattr(actor, 'stop'): actor.stop()
            if actor.is_alive: actor.destroy()
        self.actor_list = []
        print("Done.")