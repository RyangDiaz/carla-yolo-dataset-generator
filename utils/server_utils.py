import subprocess
import os
import time

# NOTE: Modify this to the appropriate Carla server launch command depending on your method of installation
LAUNCH_STRING = './CarlaUE4.sh'

def stop_carla_server():
    kill_process = subprocess.Popen('killall -9 -r CarlaUE4-Linux', shell=True)
    kill_process.wait()
    time.sleep(1)
    log.info("[INFO] Shut down existing Carla servers")

def start_carla_server(carla_sh_str):
    stop_carla_server()
    server_process = subprocess.Popen(LAUNCH_STRING, shell=True, preexec_fn=os.setsid)
    time.sleep(self._t_sleep)
