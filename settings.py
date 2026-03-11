import carla

class Config:
    # --- SIMULATOR SETTINGS ---
    SIMULATOR_HOST = "172.31.240.1"
    SIMULATOR_PORT = 2000
    TIMEOUT = 120.0
    SYNC_MODE = True
    FIXED_DELTA_SECONDS = 0.1
    VISUALIZE = True
    MAP = "Town02"
    
    # --- ENVIRONMENT NOISE ---
    N_VEHICLES = 5
    N_WALKERS = 5
    
    # --- SENSOR SETTINGS ---
    IMG_WIDTH = 80
    IMG_HEIGHT = 80
    CAM_FOV = 110
    LIDAR_RANGE = 15.0
    LIDAR_CHANNELS = 32
    LIDAR_PPS = 100000
    
    # --- REWARD WEIGHTS ---
    REWARD_COLLISION = -300.0 #this was 100 , maybe 1000 
    REWARD_LANE_INVASION = -1.0
    REWARD_GOAL_REACHED = 200.0
    REWARD_JERK_PENALTY = -0.5
    PROGRESS_REWARD_WEIGHT = 5.0 #this was 2 
    REWARD_IDLE = -300.0
    REWARD_SWERVE = -1.0
    REWARD_HARD_BREAK = -1.0
    
    # --- LANE KEEPING ---
    REWARD_OFF_ROAD = -300.0
    CTE_PENALTY_WEIGHT = 2.0
    MAX_CTE_ERROR = 2.0
