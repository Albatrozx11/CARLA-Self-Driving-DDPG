import numpy as np
import tensorflow as tf
from model import create_actor, create_critic
from ddpg_learner import ReplayBuffer, OUNoise
from sources.carla import CarlaEnv

# 1. Test Model Shapes
print("--- Testing Models ---")
actor = create_actor()
critic = create_critic()

cam_sample = np.random.rand(1, 80, 80, 1)
lidar_sample = np.random.rand(1, 80, 80, 1)
vec_sample = np.random.rand(1, 9)

# Actor should return shape (1, 2)
action = actor.predict([cam_sample, lidar_sample, vec_sample], verbose=0)
print(f"Actor output shape: {action.shape}, expected (1, 2)")
assert action.shape == (1, 2)

# Critic should accept the action shape 2
q_val = critic.predict([cam_sample, lidar_sample, vec_sample, action], verbose=0)
print(f"Critic output shape: {q_val.shape}, expected (1, 1)")
assert q_val.shape == (1, 1)

print("Models OK!")

# 2. Test ReplayBuffer & Noise Shapes
print("--- Testing ReplayBuffer & OUNoise ---")
buffer = ReplayBuffer(capacity=10)
noise_gen = OUNoise(action_dimension=2)

action_noise = noise_gen.noise()
print(f"Noise output shape: {action_noise.shape}, expected (2,)")
assert action_noise.shape == (2,)

# Dummy states
state = [cam_sample[0], lidar_sample[0], vec_sample[0]]
next_state = [cam_sample[0], lidar_sample[0], vec_sample[0]]
buffer.store(state, action_noise, 0.5, next_state, False)

samples = buffer.sample(1)
actions = samples[1]
print(f"Sampled actions shape: {actions.shape}, expected (1, 2)")
assert actions.shape == (1, 2)

print("ReplayBuffer & OUNoise OK!")
