"""
Evaluation Mode: Run the trained agent without any training or noise.
Usage: python evaluate.py
"""
import cv2
import os
import csv
import datetime
import numpy as np
from sources import CarlaEnv
from model import create_actor
import time

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

EVAL_LOG_FILE = "evaluation_logs.csv"
EVAL_LOG_HEADER = ["timestamp", "episode", "result", "reward", "steps", "distance_to_goal", "collisions", "lane_invasions", "max_speed"]

def log_eval_episode(episode, result, total_reward, steps, stats):
    """Append one row to the evaluation CSV log."""
    file_exists = os.path.exists(EVAL_LOG_FILE)
    with open(EVAL_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(EVAL_LOG_HEADER)
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            episode,
            result,
            round(total_reward, 2),
            steps,
            round(stats.get("distance_to_goal", -1), 2),
            stats.get("collision_count", 0),
            stats.get("lane_invasions", 0),
            round(stats.get("max_speed", 0), 2)
        ])

def evaluate():
    env = CarlaEnv()
    actor = create_actor()

    actor_path = "echo_drive_actor.h5"
    actor.load_weights(actor_path)
    print(f"Loaded weights from {actor_path}")

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
                cur_lidar = state[1][np.newaxis, ...]
                cur_state_vec = np.array([state[2]])

                action = actor([cur_state_img, cur_lidar, cur_state_vec], training=False).numpy()[0]
                action = np.clip(action, -1.0, 1.0)

                next_state, reward, done, stats = env.step(action)
                total_reward += reward
                steps += 1
                state = next_state

                # Visualization
                if state[0] is not None:
                    display_cam = (state[0][:, :, 0] * 255).astype(np.uint8)
                    display_cam = cv2.resize(display_cam, (400, 400), interpolation=cv2.INTER_NEAREST)
                    display_cam = cv2.cvtColor(display_cam, cv2.COLOR_GRAY2BGR)

                    waypoint_data = stats.get("waypoints_xy", [])
                    origin_x, origin_y = 200, 380
                    if len(waypoint_data) == 20:
                        for i in range(10):
                            wp_x = waypoint_data[i * 2] * 25.0
                            wp_y = waypoint_data[i * 2 + 1] * 25.0
                            pixel_x = int(origin_x + (wp_y * 8.0))
                            pixel_y = int(origin_y - (wp_x * 8.0))
                            cv2.circle(display_cam, (pixel_x, pixel_y), 4, (0, 255, 255), -1)

                    # Add telemetry overlay
                    cv2.putText(display_cam, f"Steer: {action[0]:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_cam, f"Accel: {action[1]:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_cam, f"Reward: {total_reward:.0f}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    cv2.imshow("EchoDrive - Evaluation Mode", display_cam)

                if state[1] is not None and len(state[1]) == 32:
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
                    cv2.imshow("LiDAR View", display_lidar)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting...")
                    return

            dist = stats.get("distance_to_goal", -1)
            result = "SUCCESS" if dist < 5.0 else "FAILED"
            print(f"Result: {result} | Reward: {total_reward:.1f} | Steps: {steps} | Dist: {dist:.1f}m")
            
            # Log to evaluation CSV
            log_eval_episode(episode, result, total_reward, steps, stats)

            env.destroy_agents()
            episode += 1

    finally:
        env.destroy_agents()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    evaluate()
