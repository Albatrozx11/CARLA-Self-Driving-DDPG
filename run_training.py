import subprocess
import time
import os
import signal
import sys

import argparse

def get_carla_executable():
    """Determine the path to the CARLA executable."""
    # 1. Check command line argument (if we add argparse later, but let's keep it simple first)
    if len(sys.argv) > 1:
        return sys.argv[1]
    
    # 2. Check environment variable
    carla_path_env = os.environ.get("CARLA_SERVER")
    if carla_path_env and os.path.exists(carla_path_env):
        return carla_path_env

    # 3. Default fallback (Mapped Windows path to WSL)
    # We must provide the WINDOWS PATH here since we are telling Windows CMD to run it!
    return r"C:\Adithyan\CARLA_0.9.14\WindowsNoEditor\CarlaUE4.exe"

def start_carla(carla_executable):
    print(f"Asking Windows to natively start CARLA Simulator from: {carla_executable} ...")
    
    # We use powershell "Start-Process" so Windows launches it completely detached from WSL.
    # This guarantees CARLA's RAM usage doesn't count against WSL's strict 50% limit.
    carla_cmd = [
        "powershell.exe",
        "-Command",
        f'Start-Process -FilePath "{carla_executable}" -ArgumentList "-quality-level=Low -ResX=84 -ResY=84 -NoVsync"'
    ]
    
    # We run the command. It will return immediately because Start-Process spawns it in the background on Windows.
    subprocess.run(carla_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("Sent launch command to Windows successfully.")
    return None # We no longer track the process object directly in WSL

def start_training():
    print("Starting Training Script (main.py)...")
    # -u flag forces unbuffered output so we see the prints instantly
    training_cmd = [sys.executable, "-u", "main.py"]
    return subprocess.Popen(training_cmd)

def kill_process(proc, name):
    """Gracefully terminate a WSL process."""
    if name == "CARLA":
        # Since CARLA is running natively in Windows, we must use Windows taskkill to hunt it down by name.
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
            # First, kill any zombie CARLA instances from a previous crash
            kill_process(None, "CARLA")
            time.sleep(2)

            start_carla(carla_executable)
            print("Waiting 15 seconds for CARLA to initialize...")
            time.sleep(15)
            
            # Verify Windows actually started it
            if not check_carla_running():
                print("CARLA failed to start on Windows (or crashed immediately). Restarting in 5s...")
                time.sleep(5)
                continue
                
            training_proc = start_training()
            
            # Monitor loop
            while True:
                time.sleep(2)
                
                training_exit = training_proc.poll()
                carla_is_running = check_carla_running()
                
                if not carla_is_running:
                    print("CARLA simulator crashed or exited unexpectedly.")
                    break # Break inner loop to restart both
                    
                if training_exit is not None:
                    if training_exit == 0:
                        print("Training script finished successfully or was manually closed. Exiting.")
                        kill_process(None, "CARLA")
                        return # Exit the supervisor completely
                    else:
                        print(f"Training script crashed (Code: {training_exit}).")
                        break # Break inner loop to restart both
                        
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
        # The finally block in main() will not catch this if it's already bubbling up,
        # but the subprocesses might be left alive if we don't handle it.
        # Actually, Python's KeyboardInterrupt propagates and `finally` runs.
        pass
