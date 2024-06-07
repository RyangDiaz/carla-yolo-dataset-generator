import carla
import numpy as np
import torch
import cv2
import argparse
from PIL import Image
from ultralytics import YOLO

import utils.world_utils as world_utils
import utils.cva_utils as cva_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    '--map', 
    type=str, 
    default='Town01',
    help="CARLA map that data will be collected on"
)

parser.add_argument(
    '--num_vehicles',
    type=int,
    default=70,
    help="Number of vehicles spawned in simulation"
)

parser.add_argument(
    '--num_walkers',
    type=int,
    default=150,
    help="Number of pedestrians spawned in simulation"
)

args = parser.parse_args()

client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
client.load_world(args.map)
world  = client.get_world()
original_settings = world.get_settings()

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.03
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

actor_list, walkers_list, all_id = world_utils.spawn_actors(client, world, args.num_vehicles, args.num_walkers)
vehicle = actor_list[0]
sensor_list, q_list, sensor_idxs = world_utils.spawn_sensors(world, vehicle, inference=True)
camera = sensor_list[0]

# Set up YOLO model for inference
model = YOLO("best_all.pt")

while True:
    # Retrieve the image
    nowFrame = world.tick()

    data = [cva_utils.retrieve_data(q,nowFrame) for q in q_list]
    assert all(x.frame == nowFrame for x in data if x is not None)

    # Skip if any sensor data is not available
    if None in data:
        continue
        
    snap = data[sensor_idxs['tick']]
    image = data[sensor_idxs['rgb']]

    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))[...,:3]
    with torch.no_grad():
        results = model.predict(img, device=1)

        # Visualize the results
        for i, r in enumerate(results):
            # Plot results image
            im_bgr = r.plot()  # BGR-order numpy array
            cv2.imshow('preds', im_bgr)
            cv2.waitKey(1)
            # im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
        del results
        torch.cuda.empty_cache()
        