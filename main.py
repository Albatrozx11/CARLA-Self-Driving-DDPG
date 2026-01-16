import cv2
import time
from sources import CarlaEnv

def main():
    # 1. Init Environment
    env = CarlaEnv()
    
    try:
        # 2. Reset (Spawn car)
        env.reset()
        print("Car spawned. ctrl + C to quit")

        while True:
            # 3. Define Action: [Steer (0.1=Right), Throttle (0.5=Half), Brake (0.0)]
            # CHANGE THESE NUMBERS TO CONTROL THE CAR
            action = [0.0, 0.5, 0.0] 
            
            # 4. Step
            state, reward, done, _ = env.step(action)

            lidar_image = state[1]
            camera_image = state[0]
            
            if camera_image is not None:
                cv2.imshow("Camera hood view",camera_image)
                
            if lidar_image is not None:
                cv2.imshow("LiDAR Top-Down", lidar_image)
              

            if cv2.waitKey(1) & 0xFF == ord('q'):        
                break

    finally:
        print("Cleaning up...")
        env.destroy_agents()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()