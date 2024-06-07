import glob
import shutil
import random
import os

for data_type in ['images', 'labels']:
    for data_split in ['train', 'val', 'test']:
        os.makedirs(os.path.join('datasets', 'carla_dataset', data_type, data_split), exist_ok=True)

for data_split in ['train', 'val', 'test']:
    os.makedirs(os.path.join('datasets', 'carla_dataset', 'bounding_boxes', data_split), exist_ok=True)

# Put train/val images in images/train folder
# Put test images in images/test folder
# Put train/val labels in labels/train folder
# Put test labels in labels/test folder
num_zeros = 0
for impath in sorted(glob.glob(os.path.join('carla_data', '*.png'))):
    frame_name = impath.split('/')[-1].split('.')[0]
    split = None
    if 'Town06' in frame_name:
        split = 'test'
    else:
        split = 'train'
    
    with open(os.path.join('carla_data', f'{frame_name}.txt'), 'r') as f:
        num_labels = len(f.readlines())
        if num_labels == 0:
            num_zeros += 1

    shutil.copy(os.path.join('carla_data', f'{frame_name}.png'), os.path.join('datasets', 'carla_dataset', 'images', split))
    shutil.copy(os.path.join('carla_data', f'{frame_name}.txt'), os.path.join('datasets', 'carla_dataset', 'labels', split))
    shutil.copy(os.path.join('carla_data', 'boundingbox', f'{frame_name}_b.png'), os.path.join('datasets', 'carla_dataset', 'bounding_boxes', split))

# Train/val: Town01 to Town04
# Test: Town05
# Train/Val split: 80-20 (800/200)
# Want to split each town dataset evenly

# Put train image names into train_yolo.txt
# Put val image names into val_yolo.txt
# Put test image names into test_yolo.txt
train_num = 0
val_num = 0
test_num = 0
for mapname in ['Town01', 'Town02', 'Town03', 'Town05']:
    train_val_set = glob.glob(os.path.join('datasets', 'carla_dataset', 'images', 'train', f'{mapname}*.png'))
    random.shuffle(train_val_set)
    train_cutoff = int(len(train_val_set)*0.8)
    train_set = sorted(train_val_set[:train_cutoff])
    val_set = sorted(train_val_set[train_cutoff:])

    with open(os.path.join('datasets', 'carla_dataset', 'train_yolo.txt'), "a") as f:
        for t in train_set:
            t = t.split('/')[-1]
            f.write(f"./images/train/{t}\n")
            train_num += 1
    
    with open(os.path.join('datasets', 'carla_dataset', 'val_yolo.txt'), "a") as f:
        for t in val_set:
            t = t.split('/')[-1]
            frame_num = t.split('.')[0]
            shutil.move(os.path.join('datasets', 'carla_dataset', 'images', 'train', t), os.path.join('datasets', 'carla_dataset', 'images', 'val'))
            shutil.move(os.path.join('datasets', 'carla_dataset', 'labels', 'train', f'{frame_num}.txt'), os.path.join('datasets', 'carla_dataset', 'labels', 'val'))
            shutil.move(os.path.join('datasets', 'carla_dataset', 'bounding_boxes', 'train', f'{frame_num}_b.png'), os.path.join('datasets', 'carla_dataset', 'bounding_boxes', 'val'))
            f.write(f"./images/val/{t}\n")
            val_num += 1

test_set = sorted(glob.glob(os.path.join('datasets', 'carla_dataset', 'images', 'test', '*.png')))
with open(os.path.join('datasets', 'carla_dataset', 'test_yolo.txt'), "a") as f:
    for t in test_set:
        t = t.split('/')[-1]
        f.write(f"./images/test/{t}\n")
        test_num += 1

print("TOTAL:", train_num + val_num + test_num)
print("TRAIN:", train_num)
print("VAL:", val_num)
print("TEST:", test_num)
print("EMPTY LABELS:", num_zeros)
