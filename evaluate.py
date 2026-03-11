"""
Evaluation Mode: Run the trained agent without any training or noise.
Usage: python evaluate.py
"""
import cv2
import numpy as np
from sources import CarlaEnv
from model import create_actor
import time

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def evaluate():
    env = CarlaEnv()
    actor = create_actor()

    actor_path = "echo_drive_actor.h5"
    try:
        actor.load_weights(actor_path)
        print(f"Loaded weights from {actor_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Note: If you get a shape mismatch, it means these weights were not trained on this branch's architecture.")
        return

    try:
        episode = 1
        while True:
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0

            print(f"\n=== Evaluation Episode {episode} ===")

            while not done:
                if state[0] is None or state[1] is None:
                    next_state, reward, done, stats = env.step([0.0, 0.0])
                    state = next_state
                    if done:
                        break
                    continue

                # Pure inference — NO noise, NO training
                cur_state_img = state[0][np.newaxis, ...]
                
                # Check lidar shape dynamically
                cur_lidar = state[1][np.newaxis, ...]
                if len(state[1].shape) == 2:  # if it comes back 80x80, add channel
                    cur_lidar = np.expand_dims(cur_lidar, axis=-1)
                    
                cur_state_vec = np.array([state[2]])

                try:
                    action = actor([cur_state_img, cur_lidar, cur_state_vec], training=False).numpy()[0]
                except Exception as e:
                    print(f"Forward pass failed! Architecture mismatch: {e}")
                    return

                action = np.clip(action, -1.0, 1.0)

                next_state, reward, done, stats = env.step(action)
                total_reward += reward
                steps += 1
                state = next_state

                # Visualization - RGB Camera
                if state[0] is not None:
                    display_cam = (state[0][:, :, 0] if len(state[0].shape)==3 else state[0]) * 255
                    display_cam = display_cam.astype(np.uint8)
                    display_cam = cv2.resize(display_cam, (400, 400), interpolation=cv2.INTER_NEAREST)
                    display_cam = cv2.cvtColor(display_cam, cv2.COLOR_GRAY2BGR)

                    cv2.putText(display_cam, f"Steer: {action[0]:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_cam, f"Accel: {action[1]:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_cam, f"Reward: {total_reward:.0f}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    cv2.imshow("EchoDrive - Camera", display_cam)

                # Visualization - LiDAR
                if state[1] is not None:
                    if len(state[1].shape) in [2, 3] and state[1].shape[0] == 80:  # 80x80 grid (old format)
                        lidar_img = (state[1][:, :, 0] if len(state[1].shape)==3 else state[1]) * 255
                        lidar_img = lidar_img.astype(np.uint8)
                        lidar_img = cv2.resize(lidar_img, (400, 400), interpolation=cv2.INTER_NEAREST)
                        cv2.imshow("EchoDrive - LiDAR (Grid)", lidar_img)
                    elif len(state[1].shape) == 1 and len(state[1]) == 32:  # 32 bins (new format)
                        display_lidar = np.zeros((400, 400, 3), dtype=np.uint8)
                        origin = (200, 380)
                        cv2.circle(display_lidar, origin, 4, (255, 255, 255), -1)
                        num_bins = len(state[1])
                        for i, dist in enumerate(state[1]):
                            ray_len = int((1 - dist) * 350)
                            angle = (i + 0.5) / num_bins * np.pi - (np.pi / 2)
                            ray_dx = np.sin(angle)
                            ray_dy = -np.cos(angle)
                            end_point = (int(origin[0] + ray_dx * ray_len), int(origin[1] + ray_dy * ray_len))
                            color = (0, 0, 255) if dist < 0.2 else (0, 255, 0)
                            cv2.line(display_lidar, origin, end_point, color, 2)
                        cv2.imshow("EchoDrive - LiDAR (Polar)", display_lidar)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting...")
                    return

            dist = stats.get("distance_to_goal", -1)
            result = "SUCCESS!" if dist < 5.0 else "Failed"
            print(f"Result: {result} | Reward: {total_reward:.1f} | Steps: {steps} | Dist: {dist:.1f}m")

            env.destroy_agents()
            episode += 1

    finally:
        env.destroy_agents()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    evaluate()
