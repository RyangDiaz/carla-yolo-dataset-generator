# Multi-Class Object Localization in CARLA Using YoloV8

TODO: Image here 

This repository provides some useful tools to generate a dataset of annotated images for use in training a YOLO object detector. Data collection is done in the [CARLA](https://carla.org//) simulator (v0.9.15), and the YOLOv8 model is trained using the [Ultralytics](https://docs.ultralytics.com) API.

## Overview

[Installation](#Installation)  
[Quick Start](#Quick-Start)  
[Dataset Customization](#Dataset-Customization)  
[Acknowledgements](#Acknowledgements)  

## Installation

This code has been tested on:
- **Ubuntu** 18.04 / 22.04
- **CARLA** 0.9.15
- **Python** 3.10.x

### Installing CARLA
Follow [these](https://carla.readthedocs.io/en/0.9.15/start_quickstart/) instructions to install the CARLA simulation environment. For a quick setup, you may also [run CARLA through Docker](https://carla.readthedocs.io/en/0.9.15/build_docker/).

### Setting up Environment
```
git clone https://github.com/RyangDiaz/carla-yolo-dataset-generator.git
cd carla-yolo-dataset-generator
conda env create -f environment.yml
```

## Usage
First, launch a CARLA server (depending on the method of installation). Launch a separate terminal session for the instructions specified below.

### Quick Start
To automatically generate a dataset of 1250 images (800 train, 200 validation, 250 test) spanning five different maps (`Town01` to `Town05`) and train a YOLOv8 model on this dataset:

`bash collect_data_and_train.sh`

You can then run inference on the trained model over `NUM_STEPS` and visualize/save prediction frames as a video:

`python yolo_realtime_inference.py --model PATH/TO/TRAINED/MODEL.pt --num_steps 2000 --map Town05 --show --save_video`

This trained detector currently has four classes: `vehicle`, `pedestrian`, `traffic_light`, and `traffic_sign`

### Dataset Customization

**Adding New Classes**

**Dataset Collection Parameters**

### Acknowledgements
