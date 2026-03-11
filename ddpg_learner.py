import numpy as np
import tensorflow as tf


class DDPGTrainer:
    def __init__(self, actor, critic, target_actor, target_critic, gamma=0.99, tau=0.005):
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        
        self.gamma = gamma
        self.tau = tau
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    def update(self, states, actions, rewards, next_states, dones):
        # --- PRE-CALCULATE TARGETS OUTSIDE THE TAPE ---
        target_actions = self.target_actor(next_states, training=False)
        target_q = self.target_critic(next_states + [target_actions], training=False)
        y = rewards + self.gamma * tf.stop_gradient(target_q) * (1 - dones)
        
        # Clip Q-targets to prevent early training Critic explosion spikes 
        # (Bounded to match max terminal rewards in settings.py: -300 collision/idle, +200 goal)
        y = tf.clip_by_value(y, -300.0, 300.0)

        # --- 1. Update Critic ---
        with tf.GradientTape() as tape:
            critic_value = self.critic(states + [actions], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # --- 2. Update Actor ---
        with tf.GradientTape() as tape:
            new_actions = self.actor(states, training=True)
            actor_loss = -tf.math.reduce_mean(self.critic(states + [new_actions], training=True))

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # Soft-update target networks every step
        self.update_targets()

        return actor_loss, critic_loss

    def update_targets(self):
        # Soft update: target = tau * live + (1 - tau) * target
        for (a, b) in zip(self.target_actor.variables, self.actor.variables):
            a.assign(b * self.tau + a * (1 - self.tau))
        for (a, b) in zip(self.target_critic.variables, self.critic.variables):
            a.assign(b * self.tau + a * (1 - self.tau))


class ReplayBuffer:
    def __init__(self, capacity, img_shape=(80, 80, 1), lidar_shape=(32,), vec_shape=(29,)):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        # Pre-allocate memory for both Camera and LiDAR images
        self.state_cam = np.zeros((capacity, *img_shape), dtype=np.float32)
        self.state_lidar = np.zeros((capacity, *lidar_shape), dtype=np.float32)
        self.state_vec = np.zeros((capacity, *vec_shape), dtype=np.float32)

        self.next_state_cam = np.zeros((capacity, *img_shape), dtype=np.float32)
        self.next_state_lidar = np.zeros((capacity, *lidar_shape), dtype=np.float32)
        self.next_state_vec = np.zeros((capacity, *vec_shape), dtype=np.float32)

        self.actions = np.zeros((capacity, 2), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        """
        Expects state/next_state to be the list: [Camera_Img, Lidar_Img, Physics_Vec]
        """
        # Current State
        self.state_cam[self.ptr] = state[0]
        self.state_lidar[self.ptr] = state[1]
        self.state_vec[self.ptr] = state[2]
        
        # Next State
        self.next_state_cam[self.ptr] = next_state[0]
        self.next_state_lidar[self.ptr] = next_state[1]
        self.next_state_vec[self.ptr] = next_state[2]
        
        # Action, Reward, Done
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        # Returns states as lists [Cam, Lidar, Vec] to match Keras multi-input
        states = [self.state_cam[idxs], self.state_lidar[idxs], self.state_vec[idxs]]
        next_states = [self.next_state_cam[idxs], self.next_state_lidar[idxs], self.next_state_vec[idxs]]
        
        return (
            states,
            self.actions[idxs],
            self.rewards[idxs],
            next_states,
            self.dones[idxs]
        )


class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        # The OU formula: dx = theta * (mu - x) + sigma * random_gaussian
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state