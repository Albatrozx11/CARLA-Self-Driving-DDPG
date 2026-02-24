import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Input, Dense, Concatenate  

def create_actor(img_shape=(80, 80, 1), vec_shape=(9,)):
    # --- Head 1: Camera (RGB/Gray) ---
    cam_input = Input(shape=img_shape, name="cam_input")
    x = Conv2D(32, 5, strides=2, activation="relu")(cam_input)
    x = Conv2D(64, 3, strides=2, activation="relu")(x)
    x = Flatten()(x)

    # --- Head 2: LiDAR (Top-down view) ---
    lidar_input = Input(shape=img_shape, name="lidar_input")
    z = Conv2D(32, 5, strides=2, activation="relu")(lidar_input)
    z = Conv2D(64, 3, strides=2, activation="relu")(z)
    z = Flatten()(z)

    # Physics processing
    vec_input = Input(shape=vec_shape)
    y = Dense(32, activation="relu")(vec_input)

    # --- Merge All ---
    concat = Concatenate()([x, z, y])
    d1 = Dense(256, activation="relu")(concat)
    d2 = Dense(128, activation="relu")(d1)
    
    # 1. Steering gets Tanh (Outputs -1.0 to 1.0)
    steer = Dense(1, activation="tanh", name="steer_out")(d2)
    
    # 2. Gas and Brake get Sigmoid (Outputs 0.0 to 1.0)
    pedals = Dense(2, activation="sigmoid", name="pedals_out")(d2)
    
    # 3. Recombine them back into a single array: [Steer, Throttle, Brake]
    outputs = Concatenate()([steer, pedals])
    
    return tf.keras.Model(inputs=[cam_input, lidar_input, vec_input], outputs=outputs)

def create_critic(img_shape=(80, 80, 1), vec_shape=(9,), action_shape=(3,)):
    # Inputs
    cam_input = Input(shape=img_shape, name="cam_input")
    lidar_input = Input(shape=img_shape, name="lidar_input")
    vec_input = Input(shape=vec_shape, name="vec_input")
    action_input = Input(shape=action_shape, name="action_input")

    # Camera Branch
    x = Conv2D(32, 5, strides=2, activation="relu")(cam_input)
    x = Flatten()(x)

    # LiDAR Branch
    z = Conv2D(32, 5, strides=2, activation="relu")(lidar_input)
    z = Flatten()(z)

    # Physics Branch
    y = Dense(32, activation="relu")(vec_input)
    
    # Merge States + Action
    concat = Concatenate()([x, z, y, action_input])
    d1 = Dense(256, activation="relu")(concat)
    d2 = Dense(128, activation="relu")(d1)
    output = Dense(1)(d2)
    
    return tf.keras.Model(inputs=[cam_input, lidar_input, vec_input, action_input], outputs=output)