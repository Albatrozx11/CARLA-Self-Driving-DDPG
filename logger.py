import csv
import os
from datetime import datetime
from settings import Config

class TrainingLogger:
    def __init__(self, filename="training_logs.csv"):
        self.filename = filename
        # Define the column headers for your log
        self.headers = [
            "timestamp", "episode", "total_reward", "distance_to_goal", 
            "avg_speed", "collision_count", "lane_invasions",
            "progress_reward", "jerk_penalty","result"
        ]
        
        # Initialize the file with headers if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_episode(self, episode_data):
        """
        Expects a dictionary with keys matching self.headers
        """
        episode_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(episode_data)
        print(f"Successfully logged Episode {episode_data['episode']}")