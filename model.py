import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Input, Dense, Concatenate

def create_actor(img_shape=(80, 80, 1), lidar_shape=(32,), vec_shape=(29,)):
    # --- Head 1: Camera (RGB/Gray) ---
    cam_input = Input(shape=img_shape, name="cam_input")
    x = Conv2D(32, 5, strides=2, activation="relu")(cam_input)
    x = Conv2D(64, 3, strides=2, activation="relu")(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x) # Condense camera features

    # --- Head 2: LiDAR (1D Polar Bins) ---
    lidar_input = Input(shape=lidar_shape, name="lidar_input")
    z = Dense(64, activation="relu")(lidar_input) # Condense LiDAR collision sensors

    # Physics processing
    vec_input = Input(shape=vec_shape)
    y = Dense(32, activation="relu")(vec_input)

    # --- Merge All ---
    concat = Concatenate()([x, z, y])
    d1 = Dense(256, activation="relu")(concat)
    d2 = Dense(128, activation="relu")(d1)
    
    # 1. Steering (-1.0 to 1.0)
    steer = Dense(1, activation="tanh", name="steer_out")(d2)
    
    # 2. Acceleration (-1.0 to 1.0) where >0 is gas, <0 is brake
    accel = Dense(1, activation="tanh", name="accel_out")(d2)
    
    # 3. Recombine into: [Steer, Acceleration]
    outputs = Concatenate()([steer, accel])
    
    return tf.keras.Model(inputs=[cam_input, lidar_input, vec_input], outputs=outputs)

def create_critic(img_shape=(80, 80, 1), lidar_shape=(32,), vec_shape=(29,), action_shape=(2,)):
    # Inputs
    cam_input = Input(shape=img_shape, name="cam_input")
    lidar_input = Input(shape=lidar_shape, name="lidar_input")
    vec_input = Input(shape=vec_shape, name="vec_input")
    action_input = Input(shape=action_shape, name="action_input")

    # Camera Branch
    x = Conv2D(32, 5, strides=2, activation="relu")(cam_input)
    x = Conv2D(64, 3, strides=2, activation="relu")(x) # Match Actor depth
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x) # Condense camera features

    # LiDAR Branch (1D Polar Bins)
    z = Dense(64, activation="relu")(lidar_input) # Condense LiDAR collision sensors

    # Physics Branch
    y = Dense(32, activation="relu")(vec_input)
    
    # Action Branch processing (Prevents drowning out the action vector)
    a = Dense(32, activation="relu")(action_input)
    
    # Merge States + Action
    concat = Concatenate()([x, z, y, a])
    d1 = Dense(256, activation="relu")(concat)
    d2 = Dense(128, activation="relu")(d1)
    output = Dense(1)(d2)
    
    return tf.keras.Model(inputs=[cam_input, lidar_input, vec_input, action_input], outputs=output)