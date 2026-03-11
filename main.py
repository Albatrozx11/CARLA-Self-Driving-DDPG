import cv2
import datetime
import os 
import json
import numpy as np
from sources import CarlaEnv
from logger import TrainingLogger  # 1. Import your new logger
from model import create_actor, create_critic
from ddpg_learner import ReplayBuffer, DDPGTrainer, OUNoise 
from settings import Config
import time

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def main():
    env = CarlaEnv()
    logger = TrainingLogger()      # 2. Initialize logger
    
    # Initialize Networks
    actor = create_actor()
    critic = create_critic()
    target_actor = create_actor()
    target_critic = create_critic()

    # --- NEW: CHECKPOINT LOADER ---
    actor_path = "echo_drive_actor.h5"
    critic_path = "echo_drive_critic.h5"

    if os.path.exists(actor_path) and os.path.exists(critic_path):
        print("Found existing checkpoints! Loading weights...")
        actor.load_weights(actor_path)
        critic.load_weights(critic_path)
        print("Successfully resumed EchoDrive's brain.")
    else:
        print("No checkpoints found. Starting fresh from Episode 1.")
    # ------------------------------

    # Sync target weights initially (MUST happen after loading checkpoints)
    target_actor.set_weights(actor.get_weights())
    target_critic.set_weights(critic.get_weights())

    # Initialize Helper Classes
    buffer = ReplayBuffer(capacity=20000)
    trainer = DDPGTrainer(actor, critic, target_actor, target_critic)
    noise_gen = OUNoise(action_dimension=2)

    # Initialize TensorBoard Writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/echodrive/' + current_time
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    # --- Epsilon Decay Settings ---
    epsilon = 1.0          # Start with 100% noise
    epsilon_decay = 0.995  # Multiply by this after every episode
    epsilon_min = 0.05     # Never go below 5% noise (keeps a tiny bit of exploration)
    start_episode = 1
    
    state_file = "training_state.json"
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state_data = json.load(f)
                start_episode = state_data.get("episode", 1) + 1 # Start from the next episode
                epsilon = state_data.get("epsilon", 1.0)
            print(f"Resuming from Episode {start_episode} with epsilon {epsilon:.4f}")
        except Exception as e:
            print(f"Failed to load training state: {e}")

    try:
        # Loop for multiple episodes
        for episode in range(start_episode, start_episode + 10000): 
            state = env.reset()
            done = False
            noise_gen.reset()

            ep_actor_loss = 0
            ep_critic_loss = 0
            training_steps = 0

            # --- FPS TRACKING SETUP ---
            fps_step_count = 0
            start_time = time.time()
            

            # episode_step_limit = 500 
            # current_step = 0
            print(f"Episode {episode} started. Press 'q' to quit.")

            while not done:
                # 1. Prepare State for Model
                # state[0] is Camera, state[1] is lidar state[2] is Physics vector
                cur_state_img = state[0][np.newaxis, ...]
                cur_lidar = state[1][np.newaxis, ...]
                cur_state_vec = np.array(state[2])[np.newaxis, ...]

                # 2. Get Action + Decaying Noise
                mu_action = actor([cur_state_img, cur_lidar, cur_state_vec], training=False).numpy()[0]
                noise = noise_gen.noise() * epsilon  # <--- Scale the noise here
                
                # Scale steer [-1, 1] and accel [-1, 1]
                final_action = np.clip(mu_action + noise, [-1.0, -1.0], [1.0, 1.0])

                # 3. Environment Step
                next_state, reward, done, stats = env.step(final_action)

                # 4. Store and Train
                buffer.store(state, final_action, reward, next_state, done)

                if buffer.size > 32:
                    samples = buffer.sample(batch_size=32)
                    a_loss, c_loss = trainer.update(*samples)
                    ep_actor_loss += a_loss
                    ep_critic_loss += c_loss
                    training_steps += 1

                state=next_state

                # --- Visualization ---
                if state[0] is not None:
                    display_cam = (state[0][:,:,0] * 255).astype(np.uint8)
                    display_cam = cv2.resize(display_cam, (400,400), interpolation=cv2.INTER_NEAREST)
                    display_cam = cv2.cvtColor(display_cam, cv2.COLOR_GRAY2BGR)

                    waypoint_data = stats.get("waypoints_xy", [])

                    origin_x = 200
                    origin_y = 380

                    # Draw ego vehicle anchor
                    cv2.circle(display_cam, (origin_x, origin_y), 6, (255,255,255), -1)

                    # --- Draw Waypoint Path ---
                    if len(waypoint_data) == 20:
                        prev_pt = None
                        for i in range(10):
                            wp_x = waypoint_data[i*2] * 25.0
                            wp_y = waypoint_data[i*2+1] * 25.0

                            pixel_x = int(origin_x + wp_y * 8.0)
                            pixel_y = int(origin_y - wp_x * 8.0)

                            pt = (pixel_x, pixel_y)

                            # draw waypoint
                            cv2.circle(display_cam, pt, 4, (0,255,255), -1)

                            # connect waypoints with path line
                            if prev_pt is not None:
                                cv2.line(display_cam, prev_pt, pt, (255,200,0), 2)

                            prev_pt = pt

                    # --- Draw Route Direction Arrow ---
                    if len(waypoint_data) >= 2:
                        wp_x = waypoint_data[0] * 25.0
                        wp_y = waypoint_data[1] * 25.0

                        arrow_x = int(origin_x + wp_y * 8.0)
                        arrow_y = int(origin_y - wp_x * 8.0)

                        cv2.arrowedLine(display_cam,
                                        (origin_x, origin_y),
                                        (arrow_x, arrow_y),
                                        (0,255,0),
                                        2,
                                        tipLength=0.3)

                    # --- Draw speed indicator ---
                    speed = stats.get("max_speed", 0)
                    cv2.putText(display_cam,
                                f"Speed: {speed:.1f} km/h",
                                (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255,255,255),
                                2)

                    cv2.imshow("Camera View", display_cam)
                if state[1] is not None:
                    display_lidar = (state[1][:, :, 0] * 255).astype(np.uint8)
                    display_lidar = cv2.resize(display_lidar, (400, 400), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("LiDAR View", display_lidar)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting loop...")
                    # LOG BEFORE QUITTING
                    logger.log_episode({
                        "episode": episode,
                        "total_reward": round(stats["total_reward"], 2),
                        "distance_to_goal": 0,
                        "avg_speed": round(stats["max_speed"] / 2, 2),
                        "collision_count": stats.get("collision_count", 0),
                        "lane_invasions": stats["lane_invasions"],
                        "progress_reward": round(stats["total_progress_reward"], 2),
                        "jerk_penalty": round(stats["total_jerk_penalty"], 2),
                        "result": "Manual Quit"
                    })
                    return # Exit the entire main function
                '''
                # --- Visualization ---         With Q value and Steer
                if state[0] is not None:
                    # 1. Ask the Critic how "good" it thinks this current action is
                    # Note: We pass final_action as a batch of 1: final_action[np.newaxis, ...]
                    current_q = critic.predict([cur_state_img, cur_lidar, cur_state_vec, final_action[np.newaxis, ...]], verbose=0)[0][0]

                    # 2. Un-normalize and scale up the 80x80 image for human viewing
                    # Multiply by 255, convert to 8-bit integer, and drop the channel dimension
                    display_cam = (state[0][:, :, 0] * 255).astype(np.uint8)
                    
                    # Convert to BGR so we can draw colored text on it
                    display_cam = cv2.cvtColor(display_cam, cv2.COLOR_GRAY2BGR)
                    
                    # Resize to 400x400 so the text is actually readable
                    display_cam = cv2.resize(display_cam, (400, 400), interpolation=cv2.INTER_NEAREST)

                    # 3. Draw Telemetry Text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    # Q-Value (Yellow)
                    cv2.putText(display_cam, f"Q-Value: {current_q:.2f}", (10, 30), font, 0.7, (0, 255, 255), 2)
                    
                    # Steer (White)
                    cv2.putText(display_cam, f"Steer: {final_action[0]:.2f}", (10, 60), font, 0.7, (255, 255, 255), 2)
                    
                    # Throttle/Brake Mapping
                    throttle_mapped = max(0.0, final_action[1])
                    brake_mapped = abs(min(0.0, final_action[1]))

                    # Throttle (Green)
                    cv2.putText(display_cam, f"Gas:   {throttle_mapped:.2f}", (10, 90), font, 0.7, (0, 255, 0), 2)
                    
                    # Brake (Red)
                    cv2.putText(display_cam, f"Brake: {brake_mapped:.2f}", (10, 120), font, 0.7, (0, 0, 255), 2)

                    # Show the final overlay
                    cv2.imshow("EchoDrive Dashcam", display_cam)

                if state[1] is not None:
                    # Scale up LiDAR as well just to match
                    display_lidar = (state[1][:, :, 0] * 255).astype(np.uint8)
                    display_lidar = cv2.resize(display_lidar, (400, 400), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("EchoDrive LiDAR", display_lidar)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting loop...")
                    # LOG BEFORE QUITTING
                    logger.log_episode({
                        "episode": episode,
                        "total_reward": round(stats["total_reward"], 2),
                        "distance_to_goal": 0,
                        "avg_speed": round(stats["max_speed"] / 2, 2),
                        "collision_count": stats.get("collision_count", 0),
                        "lane_invasions": stats["lane_invasions"],
                        "progress_reward": round(stats["total_progress_reward"], 2),
                        "jerk_penalty": round(stats["total_jerk_penalty"], 2),
                        "result": "Manual Quit"
                    })
                    return # Exit the entire main function
                '''
                # --- FPS TRACKING LOGIC ---
                fps_step_count += 1
                if fps_step_count % 10 == 0:
                    elapsed_time = time.time() - start_time
                    fps = 10 / elapsed_time
                    # \r overwrites the line so it doesn't spam your terminal
                    print(f"Episode {episode} | Step {fps_step_count} | Speed: {fps:.1f} FPS", end="\r")
                    start_time = time.time()
                    
            # print()
            if episode % 10 == 0:
                actor.save_weights("echo_drive_actor.h5")
                critic.save_weights("echo_drive_critic.h5")
                try:
                    with open("training_state.json", "w") as f:
                        json.dump({"episode": episode, "epsilon": epsilon}, f)
                except Exception as e:
                    print(f"Failed to save training state: {e}")
                print("Checkpoint Saved.")

            if stats.get("idle_steps", 0) >= 100:
                final_result = "Paralyzed/Idle"
            elif "off_road_lane_type" in stats:
                final_result = f"Off-Road ({stats['off_road_lane_type']})"
            else:
                final_result = "Success" if stats["distance_to_goal"] < 5.0 else "Crashed/Failed"

            logger.log_episode({
                "episode": episode,
                "total_reward": round(stats["total_reward"], 2),
                "distance_to_goal": round(stats["distance_to_goal"], 2), # Use tracked dist
                "avg_speed": round(stats["max_speed"] / 2, 2),
                "collision_count": stats["collision_count"],
                "lane_invasions": stats["lane_invasions"],
                "progress_reward": round(stats["total_progress_reward"], 2),
                "jerk_penalty": round(stats["total_jerk_penalty"], 2),
                "result": final_result
            })
            with summary_writer.as_default():
                tf.summary.scalar('Reward/Total_Reward', stats["total_reward"], step=episode)
                tf.summary.scalar('Reward/Distance_to_Goal', stats["distance_to_goal"], step=episode)
                tf.summary.scalar('Metrics/Max_Speed', stats["max_speed"], step=episode)
                tf.summary.scalar('Metrics/Collisions', stats["collision_count"], step=episode)
                
                # --- Decay Epsilon ---
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay
                    
                tf.summary.scalar('Metrics/Epsilon', epsilon, step=episode)
                
                # Only log loss if we actually trained this episode
                if training_steps > 0:
                    avg_a_loss = ep_actor_loss / training_steps
                    avg_c_loss = ep_critic_loss / training_steps
                    tf.summary.scalar('Loss/Actor', avg_a_loss, step=episode)
                    tf.summary.scalar('Loss/Critic', avg_c_loss, step=episode)


            
    finally:
        print("Cleaning up...")
        env.destroy_agents()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()