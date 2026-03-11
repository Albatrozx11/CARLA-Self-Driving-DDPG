import subprocess
import time
import os
import signal
import sys

def get_carla_executable():
    """Determine the path to the CARLA executable."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    
    carla_path_env = os.environ.get("CARLA_SERVER")
    if carla_path_env and os.path.exists(carla_path_env):
        return carla_path_env

    return r"C:\Adithyan\CARLA_0.9.14\WindowsNoEditor\CarlaUE4.exe"

def start_carla(carla_executable):
    print(f"Asking Windows to natively start CARLA Simulator from: {carla_executable} ...")
    
    carla_cmd = [
        "powershell.exe",
        "-Command",
        f'Start-Process -FilePath "{carla_executable}" -ArgumentList "-quality-level=Low -ResX=84 -ResY=84 -NoVsync"'
    ]
    
    subprocess.run(carla_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Sent launch command to Windows successfully.")
    return None 

def start_training():
    print("Starting Training Script (main.py)...")
    training_cmd = [sys.executable, "-u", "main.py"]
    return subprocess.Popen(training_cmd)

def kill_process(proc, name):
    """Gracefully terminate a WSL process."""
    if name == "CARLA":
        print("Force killing any running CARLA Engine on Windows...")
        subprocess.run(["taskkill.exe", "/F", "/IM", "CarlaUE4.exe", "/T"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    if proc and proc.poll() is None:
        print(f"Terminating {name}...")
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"{name} did not terminate gracefully. Forcing kill...")
            proc.kill()
            proc.wait(timeout=2)
        except Exception as e:
            print(f"Error terminating {name}: {e}")

def check_carla_running():
    """Checks Windows tasklist to see if CarlaUE4.exe is currently running."""
    try:
        output = subprocess.check_output(["tasklist.exe", "/FI", "IMAGENAME eq CarlaUE4.exe"], text=True)
        return "CarlaUE4.exe" in output
    except Exception:
        return False

def main():
    carla_executable = get_carla_executable()
    
    while True:
        training_proc = None
        try:
            kill_process(None, "CARLA")
            time.sleep(2)

            start_carla(carla_executable)
            print("Waiting 15 seconds for CARLA to initialize...")
            time.sleep(15)
            
            if not check_carla_running():
                print("CARLA failed to start on Windows (or crashed immediately). Restarting in 5s...")
                time.sleep(5)
                continue
                
            training_proc = start_training()
            
            while True:
                time.sleep(2)
                
                training_exit = training_proc.poll()
                carla_is_running = check_carla_running()
                
                if not carla_is_running:
                    print("CARLA simulator crashed or exited unexpectedly.")
                    break 
                    
                if training_exit is not None:
                    if training_exit == 0:
                        print("Training script finished successfully or was manually closed. Exiting.")
                        kill_process(None, "CARLA")
                        return 
                    else:
                        print(f"Training script crashed (Code: {training_exit}).")
                        break 
                        
        finally:
            print("Cleaning up processes before restart/exit...")
            kill_process(training_proc, "Training Script")
            kill_process(None, "CARLA")
            
        print("Waiting 5 seconds before restarting sequence...\n" + "-"*40)
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSupervisor interrupted by user. Shutting down...")
        pass
