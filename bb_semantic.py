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
import bbox_filter as cva
import cva_utils
import argparse

weather_presets = [
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.CloudyNoon,
    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.WetCloudyNoon,
    carla.WeatherParameters.MidRainyNoon,
    carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.SoftRainNoon,
    carla.WeatherParameters.ClearSunset,
    carla.WeatherParameters.CloudySunset,
    carla.WeatherParameters.WetSunset,
    carla.WeatherParameters.WetCloudySunset,
    carla.WeatherParameters.MidRainSunset,
    carla.WeatherParameters.HardRainSunset,
    carla.WeatherParameters.SoftRainSunset
]

output_path = './carla_data'
SpawnActor = carla.command.SpawnActor

# project 3D point to 2D
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

# use the camera projection matrix to project the 3D points in camera coordinates into the 2D camera plane
def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def draw_boundingbox(img, x_min, y_min, x_max, y_max, color=(0,0,255,255)):
    cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), color, 1)
    cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), color, 1)
    cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), color, 1) 
    cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), color, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Town01')

    args = parser.parse_args()

    vehicle_num = 70
    walker_num = 150
    map_name = args.map
    actor_list = []
    sensor_list = []
    walkers_list = []
    all_id = []

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    client.load_world(map_name)
    world  = client.get_world()
    original_settings = world.get_settings()

    write_files = True

    try:

        bp_lib = world.get_blueprint_library()

        # Set up the simulator in synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True # Enables synchronous mode
        settings.fixed_delta_seconds = 0.03
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        # spawn vehicle
        vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
        # Get the map spawn points
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        # Set up automatic drive
        vehicle.set_autopilot(True)
        # collect all actors to destroy when we quit the script
        actor_list.append(vehicle)

        # generate npc vehicle
        for i in range(vehicle_num):
            vehicle_npc = random.choice(bp_lib.filter('vehicle'))
            # vehicle_npc = bp_lib.find('vehicle.carlamotors.firetruck')
            npc = world.try_spawn_actor(vehicle_npc, random.choice(spawn_points))

            # set light state
            # light_state = carla.VehicleLightState(carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2)
            if npc:
                npc.set_autopilot(True)
                # apply light state
                # npc.set_light_state(light_state)
                actor_list.append(npc)
        
        # spawn pedestrians
        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(walker_num):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                pass
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
                print("Spawned pedestrian")
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                pass
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        q_list = []
        idx = 0
        tick_queue = Queue()
        world.on_tick(tick_queue.put)
        q_list.append(tick_queue)
        tick_idx = idx
        idx = idx+1

        ### spawn camera
        camera_bp = bp_lib.find('sensor.camera.rgb')
        # Set camera blueprint properties
        camera_bp.set_attribute('bloom_intensity','1')
        camera_bp.set_attribute('fov','100')
        # camera_bp.set_attribute('sensor_tick', '1.0')
        # camera_bp.set_attribute('slope','0.7')
        # camera position related to the vehicle
        camera_init_trans = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
        # output_path = os.path.join("../project/image", '%06d.png')
        
        if not os.path.exists(output_path): 
            os.makedirs(output_path)
        # camera.listen(lambda image: sensor_callback(image, sensor_queue, "camera"))
        cam_queue = Queue()
        camera.listen(cam_queue.put)
        # Create a queue to store and retrieve the sensor data
        sensor_list.append(camera)
        q_list.append(cam_queue)
        cam_idx = idx
        idx = idx+1
        print('RGB camera ready')

        # Spawn depth camera
        depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute('fov','100')
        # depth_bp.set_attribute('slope','0.7')
        depth_init_trans = carla.Transform(carla.Location(x=1.5, z=2.4))
        # depth_bp.set_attribute('sensor_tick', '1.0')
        depth = world.spawn_actor(depth_bp, depth_init_trans, attach_to=vehicle)
        # cc_depth_log = carla.ColorConverter.LogarithmicDepth
        # nonvehicles_list.append(depth)
        depth_queue = Queue()
        depth_color_converter = carla.ColorConverter.LogarithmicDepth
        depth.listen(depth_queue.put)
        sensor_list.append(depth)
        q_list.append(depth_queue)
        depth_idx = idx
        idx = idx+1
        print('Depth camera ready')

        # Spawn LIDAR sensor
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        # lidar_bp.set_attribute('sensor_tick', '1.0')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '1120000')
        lidar_bp.set_attribute('upper_fov', '40')
        lidar_bp.set_attribute('lower_fov', '-40')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('rotation_frequency', '40')
        lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        # nonvehicles_list.append(lidar)
        lidar_queue = Queue()
        lidar.listen(lidar_queue.put)
        sensor_list.append(lidar)
        q_list.append(lidar_queue)
        lidar_idx = idx
        idx = idx+1
        print('LIDAR ready')

        ### Spawn semantic segmentation camera
        semantic_bp = bp_lib.find('sensor.camera.instance_segmentation')
        # Set camera blueprint properties
        semantic_bp.set_attribute('fov','100')
        # camera position related to the vehicle
        camera_init_trans = carla.Transform(carla.Location(x=1.5, z=2.4))
        semantic = world.spawn_actor(semantic_bp, camera_init_trans, attach_to=vehicle)

        semantic_queue = Queue()
        semantic.listen(semantic_queue.put)
        # Create a queue to store and retrieve the sensor data
        sensor_list.append(semantic)
        q_list.append(semantic_queue)
        semantic_idx = idx
        idx = idx+1
        print('Semantic camera ready')

        # Get the world to camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        # Get the attributes from the camera
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        # Calculate the camera projection matrix to project from 3D -> 2D
        K = build_projection_matrix(image_w, image_h, fov)

        # Retrieve all bounding boxes for traffic lights within the level
        # bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        # Filter the list to extract bounding boxes within a 50m radius
        # nearby_bboxes = []
        # for bbox in bounding_box_set:
        #     if bbox.location.distance(camera.get_transform().location) < 50:
        #         nearby_bboxes
        
        # edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

        image_count = 0
        save_every = 40

        motorcycle_list = ["vehicle.harley-davidson.low_rider", "vehicle.kawasaki.ninja", "vehicle.vespa.zx125", "vehicle.yamaha.yzf"]
        bike_list = ["vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"]
        emergency_list = ["vehicle.dodge.charger_police", "vehicle.dodge.charger_police_2020", "vehicle.carlamotors.firetruck", "vehicle.ford.ambulance"]

        weather_every = 80
        weather_tick = 20
        while image_count // save_every < 50:
            # Turn all traffic lights green
            for tl in world.get_actors().filter('*traffic_light*'):
                tl.set_state(carla.TrafficLightState.Green)
            
            # Change weather
            if weather_tick == 0:
                print("Changing weather...")
                world.set_weather(random.choice(weather_presets))
            weather_tick = (weather_tick + 1) % weather_every

            # Retrieve the image
            nowFrame = world.tick()

            data = [cva.retrieve_data(q,nowFrame) for q in q_list]
            assert all(x.frame == nowFrame for x in data if x is not None)

            # Skip if any sensor data is not available
            if None in data:
                continue
                
            snap = data[tick_idx]
            image = data[cam_idx]
            depth_image = data[depth_idx]
            lidar_image = data[lidar_idx]
            semantic_image = data[semantic_idx]

            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            # Get the camera matrix 
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # Save image
            if write_files:
                if image_count % save_every == 0:
                    print("Saving image...")
                    image.save_to_disk(os.path.join(output_path, map_name + '_' + '%06d.png' % image.frame))
                    # depth_image.save_to_disk(os.path.join(output_path, '%06d_d.png' % image.frame), depth_color_converter)
                    open(os.path.join(output_path, map_name + '_' + '%06d.txt' % image.frame), "a")
            # (PASCAL VOC format) Initialize the exporter
            # writer = Writer(output_path + '.png', image_w, image_h)


            boundingbox_path = os.path.join(output_path, "boundingbox")
            if not os.path.exists(boundingbox_path): 
                os.makedirs(boundingbox_path)

            # Get all vehicle bounding boxes
            vehicles_raw = world.get_actors().filter('*vehicle*')
            vehicles = cva_utils.snap_processing(vehicles_raw, snap)
            filtered_out, _ = cva_utils.auto_annotate_lidar(
                vehicles, 
                camera, 
                lidar_image,
                max_dist=40,
                min_detect=6, 
                show_img=None, 
                json_path=None
            )

            # print(len(filtered_out['vehicles']), "found")

            for npc, bbox in zip(filtered_out['vehicles'], filtered_out['bbox']):
                annotation_str = ""
                u1 = int(bbox[0,0])
                v1 = int(bbox[0,1])
                u2 = int(bbox[1,0])
                v2 = int(bbox[1,1])

                x_center = ((bbox[0,0] + bbox[1,0])/2) / image.width
                y_center = ((bbox[0,1] + bbox[1,1])/2) / image.height
                width = (bbox[1,0] - bbox[0,0]) / image.width
                height = (bbox[1,1] - bbox[0,1]) / image.height

                if np.max([x_center, y_center, width, height]) <= 1.0:

                    draw_boundingbox(img, u1, v1, u2, v2, color=(255,0,0,255))

                    if npc.type_id in bike_list or npc.type_id in motorcycle_list:
                        class_id = 0
                    else:
                        class_id = 1

                    annotation_str += f"{class_id} {x_center} {y_center} {width} {height}\n"
                    # print("VEHICLES", annotation_str[0])
                    if write_files:
                        if image_count % save_every == 0:
                            with open(os.path.join(output_path, map_name + '_' + '%06d.txt' % image.frame), "a") as f:
                                f.write(annotation_str)


            # Get all pedestrian bounding boxes
            pedestrians_raw = world.get_actors().filter('*pedestrian*')
            pedestrians = cva_utils.snap_processing(pedestrians_raw, snap)
            depth_meter = cva_utils.extract_depth(depth_image)
            filtered_out, _ = cva_utils.auto_annotate(
                pedestrians, 
                camera, 
                depth_meter,
                depth_margin=10,
                max_dist=40,
            )

            # print(len(filtered_out['vehicles']), "found")

            for npc, bbox in zip(filtered_out['vehicles'], filtered_out['bbox']):
                annotation_str = ""
                u1 = int(bbox[0,0])
                v1 = int(bbox[0,1])
                u2 = int(bbox[1,0])
                v2 = int(bbox[1,1])

                x_center = ((bbox[0,0] + bbox[1,0])/2) / image.width
                y_center = ((bbox[0,1] + bbox[1,1])/2) / image.height
                width = (bbox[1,0] - bbox[0,0]) / image.width
                height = (bbox[1,1] - bbox[0,1]) / image.height

                if np.max([x_center, y_center, width, height]) <= 1.0:

                    draw_boundingbox(img, u1, v1, u2, v2, color=(255,165,0,255))

                    class_id = 2

                    annotation_str += f"{class_id} {x_center} {y_center} {width} {height}\n"
                    # print("VEHICLES", annotation_str[0])
                    if write_files:
                        if image_count % save_every == 0:
                            with open(os.path.join(output_path, map_name + '_' + '%06d.txt' % image.frame), "a") as f:
                                f.write(annotation_str)

            # Getting traffic light bounding boxes
            
            bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)

            # print("---------------------------------------------------")
            # semantic_img = np.array(semantic_image.raw_data).reshape((semantic_image.height,semantic_image.width,3))[:,:,0]
            for bb in bounding_box_set:
                annotation_str = ""
                # Filter for distance from ego vehicle
                dist = bb.location.distance(vehicle.get_transform().location)
                if dist < 30:

                    # Calculate the dot product between the forward vector
                    # of the vehicle and the vector between the vehicle
                    # and the bounding box. We threshold this dot product
                    # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = bb.location - vehicle.get_transform().location

                    if forward_vec.dot(ray) > 1:
                        # Cycle through the vertices
                        verts = [v for v in bb.get_world_vertices(carla.Transform())]
                        x_max = -10000
                        x_min = 10000
                        y_max = -10000
                        y_min = 10000                        
                        for vert in verts:
                            # Join the vertices into edges
                            p = get_image_point(vert, K, world_2_camera)
                            if p[0] > x_max:
                                x_max = p[0]
                            if p[0] < x_min:
                                x_min = p[0]
                            if p[1] > y_max:
                                y_max = p[1]
                            if p[1] < y_min:
                                y_min = p[1]

                        dist_margin = 10000
                        for vert in verts:
                            dist_vert = vehicle.get_transform().location.distance(vert)
                            dist_margin = min(dist_margin, dist_vert)

                        xc = int((x_max+x_min)/2)
                        yc = int((y_max+y_min)/2)
                        wp = int((x_max-x_min) * 0.8/2)
                        hp = int((y_max-y_min) * 0.8/2)
                        u1 = xc-wp
                        u2 = xc+wp
                        v1 = yc-hp
                        v2 = yc+hp

                        # draw_boundingbox(img, u1, v1, u2, v2, color=(0,255,0,255))

                        depth_meter = cva.extract_depth(depth_image)
                        depth_bb = np.array(depth_meter[v1:v2+1,u1:u2+1])                            
                            
                        dist_delta_new = np.full(depth_bb.shape, dist - 10)
                        s_patch = np.array(depth_bb > dist_delta_new)
                        # print("dist", dist - 10)
                        # print("depth_bb: ")
                        # print(depth_bb)
                        # print("u1: ", u1, ", u2: ", u2, ", v1: ", v1, ", v2: ", v2)
                        s = np.sum(s_patch) > s_patch.shape[0]*0.1

                        if s:
                            # Draw the edges into the camera output
                            draw_boundingbox(img, u1, v1, u2, v2, color=(0,255,0,255))

                            if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h: 
                                class_id = 3
                                x_center = ((x_min + x_max) / 2) / image_w
                                y_center = ((y_min + y_max) / 2) / image_h
                                width = (x_max - x_min) / image_w
                                height = (y_max - y_min) / image_h
                                annotation_str += f"{class_id} {x_center} {y_center} {width} {height}\n"
                            
                            if write_files:
                                if image_count % save_every == 0:
                                    with open(os.path.join(output_path, map_name + '_' + '%06d.txt' % image.frame), "a") as f:
                                        f.write(annotation_str)
            
            # Getting traffic sign bounding boxes
            
            bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)

            # print("---------------------------------------------------")
            for bb in bounding_box_set:
                annotation_str = ""
                # Filter for distance from ego vehicle
                dist = bb.location.distance(vehicle.get_transform().location)
                if dist < 30:

                    # Calculate the dot product between the forward vector
                    # of the vehicle and the vector between the vehicle
                    # and the bounding box. We threshold this dot product
                    # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = bb.location - vehicle.get_transform().location

                    if forward_vec.dot(ray) > 1:
                        # Cycle through the vertices
                        verts = [v for v in bb.get_world_vertices(carla.Transform())]
                        x_max = -10000
                        x_min = 10000
                        y_max = -10000
                        y_min = 10000                        
                        for vert in verts:
                            # Join the vertices into edges
                            p = get_image_point(vert, K, world_2_camera)
                            if p[0] > x_max:
                                x_max = p[0]
                            if p[0] < x_min:
                                x_min = p[0]
                            if p[1] > y_max:
                                y_max = p[1]
                            if p[1] < y_min:
                                y_min = p[1]

                        dist_margin = 10000
                        for vert in verts:
                            dist_vert = vehicle.get_transform().location.distance(vert)
                            dist_margin = min(dist_margin, dist_vert)

                        xc = int((x_max+x_min)/2)
                        yc = int((y_max+y_min)/2)
                        wp = int((x_max-x_min) * 0.8/2)
                        hp = int((y_max-y_min) * 0.8/2)
                        u1 = xc-wp
                        u2 = xc+wp
                        v1 = yc-hp
                        v2 = yc+hp

                        # draw_boundingbox(img, u1, v1, u2, v2, color=(0,0,0,255))

                        depth_meter = cva.extract_depth(depth_image)
                        depth_bb = np.array(depth_meter[v1:v2+1,u1:u2+1])                            
                            
                        dist_delta_new = np.full(depth_bb.shape, dist - 10)
                        s_patch = np.array(depth_bb > dist_delta_new)
                        # print("dist", dist - 10)
                        # print("depth_bb: ")
                        # print(depth_bb)
                        # print("u1: ", u1, ", u2: ", u2, ", v1: ", v1, ", v2: ", v2)
                        s = np.sum(s_patch) > s_patch.shape[0]*0.1

                        if s:
                            # Draw the edges into the camera output
                            draw_boundingbox(img, u1, v1, u2, v2, color=(0,0,0,255))

                            if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h: 
                                class_id = 4
                                x_center = ((x_min + x_max) / 2) / image_w
                                y_center = ((y_min + y_max) / 2) / image_h
                                width = (x_max - x_min) / image_w
                                height = (y_max - y_min) / image_h
                                annotation_str += f"{class_id} {x_center} {y_center} {width} {height}\n"
                            
                            if write_files:
                                if image_count % save_every == 0:
                                    with open(os.path.join(output_path, map_name + '_' + '%06d.txt' % image.frame), "a") as f:
                                        f.write(annotation_str)


            # Show image with bounding box
            cv2.imshow('ImageWindowName',img)
            # Save image with bounding box
            if write_files:
                if image_count % save_every == 0:
                    output_file_path = os.path.join(boundingbox_path, map_name + '_' + '%06d_b.png' % image.frame)
                    cv2.imwrite(output_file_path, img)
            image_count += 1
            if cv2.waitKey(1) == ord('q'):
                break

            # (PASCAL VOC format) Save the bounding boxes in the scene
            # writer.save(os.path.join(output_path, '%06d.xml' % image.frame))

        cv2.destroyAllWindows()

    finally:
        world.apply_settings(original_settings)
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in walkers_list])
        for sensor in sensor_list:
            sensor.destroy()
        print('done.')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')