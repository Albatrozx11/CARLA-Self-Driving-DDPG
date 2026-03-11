import subprocess
import time
import os
import signal
import sys

def get_carla_executable():
    if len(sys.argv) > 1:
        return sys.argv[1]
    carla_path_env = os.environ.get("CARLA_SERVER")
    if carla_path_env and os.path.exists(carla_path_env):
        return carla_path_env
    return r"C:\Adithyan\CARLA_0.9.14\WindowsNoEditor\CarlaUE4.exe"

def start_carla(carla_executable):
    print(f"Asking Windows to start CARLA from: {carla_executable} ...")
    carla_cmd = [
        "powershell.exe",
        "-Command",
        f'Start-Process -FilePath "{carla_executable}" -ArgumentList "-quality-level=Low -ResX=84 -ResY=84 -NoVsync"'
    ]
    subprocess.run(carla_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Sent launch command to Windows successfully.")

def start_evaluate():
    print("Starting Evaluation Script (evaluate.py)...")
    return subprocess.Popen([sys.executable, "-u", "evaluate.py"])

def kill_process(proc, name):
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
            proc.kill()
            proc.wait(timeout=2)
        except Exception as e:
            print(f"Error terminating {name}: {e}")

def check_carla_running():
    try:
        output = subprocess.check_output(["tasklist.exe", "/FI", "IMAGENAME eq CarlaUE4.exe"], text=True)
        return "CarlaUE4.exe" in output
    except Exception:
        return False

def main():
    carla_executable = get_carla_executable()
    eval_proc = None
    try:
        kill_process(None, "CARLA")
        time.sleep(2)
        start_carla(carla_executable)
        print("Waiting 15 seconds for CARLA to initialize...")
        time.sleep(15)

        if not check_carla_running():
            print("CARLA failed to start. Exiting.")
            return

        eval_proc = start_evaluate()

        while True:
            time.sleep(2)
            eval_exit = eval_proc.poll()
            if not check_carla_running():
                print("CARLA crashed unexpectedly.")
                break
            if eval_exit is not None:
                print("Evaluation finished." if eval_exit == 0 else f"Evaluation crashed (Code: {eval_exit}).")
                break
    finally:
        print("Cleaning up processes...")
        kill_process(eval_proc, "Evaluation Script")
        kill_process(None, "CARLA")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down...")
