import glob
import os
import sys
import cv2
import carla
import math
import random
from queue import Queue
from queue import Empty
import numpy as np
# from pascal_voc_writer import Writer
import utils.cva_utils as cva_utils
import utils.world_utils as world_utils
import utils.img_utils as img_utils
import utils.bbox_utils as bbox_utils
import utils.server_utils as server_utils
import argparse

def retrieve_data(sensor_queue, frame, timeout=5):
    while True:
        try:
            data = sensor_queue.get(True,timeout)
        except Empty:
            return None
        if data.frame == frame:
            return data

def main(args):
    output_path = args.output_path
    if args.save:
        if not os.path.exists(output_path): 
            os.makedirs(output_path)
    
    # ==============================================
    # Set up CARLA server
    # ==============================================
    server_utils.start_carla_server()

    # ==============================================
    # Set up CARLA world
    # ==============================================

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    print("Loading world", args.map)
    client.load_world(args.map)
    world  = client.get_world()
    original_settings = world.get_settings()

    try:

        bp_lib = world.get_blueprint_library()

        # Set up the simulator in synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True # Enables synchronous mode
        settings.fixed_delta_seconds = 0.03
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        actor_list, walkers_list, all_id = world_utils.spawn_actors(client, world, args.num_vehicles, args.num_walkers)
        vehicle = actor_list[0]
        sensor_list, q_list, sensor_idxs = world_utils.spawn_sensors(world, vehicle)
        camera = sensor_list[0]

        # TODO: Discard images with less than X labels
        image_count = 0
        num_saved = 0
        save_every = 40

        weather_every = 80
        weather_tick = 20

        while (not args.save) or (num_saved < args.num_save):
            # Turn all traffic lights green
            for tl in world.get_actors().filter('*traffic_light*'):
                tl.set_state(carla.TrafficLightState.Green)
            
            # Change weather
            if weather_tick == 0 and not args.constant_weather:
                print("Changing weather...")
                world.set_weather(random.choice(world_utils.weather_presets))
            weather_tick = (weather_tick + 1) % weather_every

            # Retrieve the image
            nowFrame = world.tick()

            data = [cva_utils.retrieve_data(q,nowFrame) for q in q_list]
            assert all(x.frame == nowFrame for x in data if x is not None)

            # Skip if any sensor data is not available
            if None in data:
                continue
                
            snap = data[sensor_idxs['tick']]
            image = data[sensor_idxs['rgb']]
            depth_image = data[sensor_idxs['depth']]
            lidar_image = data[sensor_idxs['lidar']]
            semantic_image = data[sensor_idxs['semantic']]

            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            boundingbox_path = os.path.join(output_path, "boundingbox")
            if args.save:
                if not os.path.exists(boundingbox_path): 
                    os.makedirs(boundingbox_path)

            bbox_draw = []
            bbox_save = []

            # Get all vehicle bounding boxes
            vehicles_raw = world.get_actors().filter('*vehicle*')
            vehicles = cva_utils.snap_processing(vehicles_raw, snap)
            vehicle_bbox_draw, vehicle_bbox_save = bbox_utils.actor_bbox_lidar(
                actor_list=vehicles,
                camera=camera,
                image=image,
                lidar_image=lidar_image,
                max_dist=40,
                min_detect=6,
                class_id=0
            )
            bbox_draw.extend(vehicle_bbox_draw)
            bbox_save.extend(vehicle_bbox_save)

            # Get all pedestrian bounding boxes
            pedestrians_raw = world.get_actors().filter('*pedestrian*')
            pedestrians = cva_utils.snap_processing(pedestrians_raw, snap)
            depth_meter = cva_utils.extract_depth(depth_image)

            walker_bbox_draw, walker_bbox_save = bbox_utils.actor_bbox_depth_semantic(
                actor_list=pedestrians, 
                camera=camera, 
                image=image,
                semantic_image=semantic_image,
                depth_image=depth_meter,
                max_dist=40, 
                depth_margin=8, 
                patch_ratio=0.4, 
                resize_ratio=0.5, 
                semantic_label=12, # Refer to https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera for semantic labels
                semantic_threshold=0.3, # The bounding box is included if at least 30% of the semantic labels inside are semantic_label
                class_id=1
            )
            bbox_draw.extend(walker_bbox_draw)
            bbox_save.extend(walker_bbox_save)

            # Getting traffic light bounding boxes
            traffic_lights = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
            depth_meter = cva_utils.extract_depth(depth_image)
            tl_bbox_draw, tl_bbox_save = bbox_utils.object_bbox_depth_semantic(
                bbox_list=traffic_lights, 
                camera=camera, 
                image=image,
                semantic_image=semantic_image,
                depth_image=depth_meter, 
                vehicle=vehicle, 
                max_dist=30, 
                semantic_threshold=0.5,
                semantic_label=7,
                class_id=2
            )

            bbox_draw.extend(tl_bbox_draw)
            bbox_save.extend(tl_bbox_save)
            
            # Getting traffic sign bounding boxes
            traffic_signs = world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
            depth_meter = cva_utils.extract_depth(depth_image)
            ts_bbox_draw, ts_bbox_save = bbox_utils.object_bbox_depth_semantic(
                bbox_list=traffic_signs, 
                camera=camera, 
                image=image,
                semantic_image=semantic_image,
                depth_image=depth_meter, 
                vehicle=vehicle, 
                max_dist=25,
                semantic_threshold=0.5,
                semantic_label=8,
                class_id=3
            )
            bbox_draw.extend(ts_bbox_draw)
            bbox_save.extend(ts_bbox_save)

            # Now, draw bboxes and show images
            annotation_str = ""
            for bbox_d, bbox_s in zip(bbox_draw, bbox_save):
                u1, v1, u2, v2 = bbox_d
                class_id, x_center, y_center, width, height = bbox_s

                img_utils.draw_boundingbox(img, u1, v1, u2, v2)
                annotation_str += f"{class_id} {x_center} {y_center} {width} {height}\n"
            
            # Save image and labels if settings allow it
            if args.save and image_count % save_every == 0:
                if len(bbox_save) < args.num_detections_save:
                    print("Minimum detections not reached, skipping...")
                else:
                    num_saved += 1
                    print(f"Saving image {num_saved}")
                    image.save_to_disk(os.path.join(output_path, args.map + '_' + '%06d.png' % image.frame))
                    with open(os.path.join(output_path, args.map + '_' + '%06d.txt' % image.frame), "a") as f:
                        f.write(annotation_str)
                
                    # Save image with bounding box
                    output_file_path = os.path.join(boundingbox_path, args.map + '_' + '%06d_b.png' % image.frame)
                    cv2.imwrite(output_file_path, img)
                    
            # Show image with bounding box
            if args.show:
                cv2.imshow('ImageWindowName',img)
                if cv2.waitKey(1) == ord('q'):
                    break
    
            image_count += 1


        cv2.destroyAllWindows()

    finally:
        world.apply_settings(original_settings)
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in walkers_list])
        for sensor in sensor_list:
            sensor.destroy()
        server_utils.stop_carla_server()
        print('done.')

if __name__ == '__main__':
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

    parser.add_argument(
        '--constant_weather',
        action='store_true',
        help="Turns mid-simulation weather switching off during data collection"
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Shows annotations on screen during data collection'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='Saves image and label data to disk'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default='./carla_data',
        help='Relative path where raw data is saved'
    )

    parser.add_argument(
        '--num_save',
        type=int,
        default=50,
        help='Number of images to save for this run.'
    )

    parser.add_argument(
        '--num_detections_save',
        type=int,
        default=1,
        help='Minimum number of detections per collected datapoint'
    )

    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')