import cv2
import time
from sources import CarlaEnv
from logger import TrainingLogger  # 1. Import your new logger
from settings import Config

def main():
    env = CarlaEnv()
    logger = TrainingLogger()      # 2. Initialize logger
    
    try:
        # Loop for multiple episodes
        for episode in range(1, 1001): 
            state = env.reset()
            done = False
            print(f"Episode {episode} started. Press 'q' to quit.")

            while not done:
                # Placeholder action: [Steer, Throttle, Brake]
                action = [0.0, 0.4, 0.0] 
                state, reward, done, stats = env.step(action)

                # --- Visualization ---
                if state[0] is not None:
                    cv2.imshow("Camera View", state[0])
                if state[1] is not None:
                    cv2.imshow("LiDAR View", state[1])
                
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
                
            logger.log_episode({
                "episode": episode,
                "total_reward": round(stats["total_reward"], 2),
                "distance_to_goal": round(stats["distance_to_goal"], 2), # Use tracked dist
                "avg_speed": round(stats["max_speed"] / 2, 2),
                "collision_count": stats["collision_count"],
                "lane_invasions": stats["lane_invasions"],
                "progress_reward": round(stats["total_progress_reward"], 2),
                "jerk_penalty": round(stats["total_jerk_penalty"], 2),
                "result": "Success" if stats["distance_to_goal"] < 5.0 else "Crashed/Failed"
            })


            
    finally:
        print("Cleaning up...")
        env.destroy_agents()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()