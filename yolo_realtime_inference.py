import carla
import numpy as np
import torch
import cv2
import argparse
import imageio
from PIL import Image
from ultralytics import YOLO

import utils.world_utils as world_utils
import utils.cva_utils as cva_utils
import utils.server_utils as server_utils

def main(args):
    server_utils.start_carla_server()

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
    model = YOLO(args.model)
    current_step = 0

    predict_frames = []

    try:
        while args.num_steps < 0 or current_step < args.num_steps:
            # Turn all traffic lights green
            for tl in world.get_actors().filter('*traffic_light*'):
                tl.set_state(carla.TrafficLightState.Green)
                
            # Retrieve the image
            nowFrame = world.tick()

            data = [cva_utils.retrieve_data(q,nowFrame) for q in q_list]
            assert all(x.frame == nowFrame for x in data if x is not None)

            # Skip if any sensor data is not available
            if None in data:
                continue
                
            snap = data[sensor_idxs['tick']]
            image = data[sensor_idxs['rgb']]

            # Throw out the alpha channel of the input image
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))[...,:3]
            with torch.no_grad():
                results = model.predict(img, device=1)

                # Visualize the results
                for i, r in enumerate(results):
                    # Plot results image
                    im_bgr = r.plot()  # BGR-order numpy array
                    if args.show:
                        cv2.imshow('preds', im_bgr)
                        cv2.waitKey(1)
                    if args.num_steps > 0 and args.save_video:
                        predict_frames.append(im_bgr[..., ::-1])

                    # im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
                del results
                torch.cuda.empty_cache()
            current_step += 1
            print("Current step", current_step)

    finally:
        if args.num_steps > 0 and args.save_video and len(predict_frames) > 0:
            # vid_frames = np.concatenate(predict_frames, axis=-2).astype(np.uint8)
            f = 'yolo_predict.mp4'
            imageio.mimwrite(f, predict_frames, fps=20)
            print(f'Saved output prediction video to {f}')
        world.apply_settings(original_settings)
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in walkers_list])
        for sensor in sensor_list:
            sensor.destroy()
        server_utils.stop_carla_server()
        print('done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help="Trained model used to perform inference (should be relative to current folder)"
    )
    parser.add_argument(
        '--num_steps', 
        type=int, 
        default=500,
        help="Number of frames to perform inference on (-1 for nonstop runtime)"
    )

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

    parser.add_argument(
        '--show',
        action='store_true',
        help="Render bounding box predictions on screen"
    )

    parser.add_argument(
        '--save_video',
        action='store_true',
        help="(only valid when num_steps > 0) Save all frames with bounding box predictions as a video"
    )

    args = parser.parse_args()
    main(args)