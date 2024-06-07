# Multi-Class Object Localization in CARLA Using YoloV8

TODO: Image here 

This repository provides some useful tools to generate a dataset of annotated images for use in training a YOLO object detector. Data collection is done in the [CARLA](https://carla.org//) simulator (v0.9.15), and the YOLOv8 model is trained using the [Ultralytics](https://docs.ultralytics.com) API.

## Installation

This code has been tested on:
- **Ubuntu** 18.04 / 22.04
- **CARLA** 0.9.15
- **Python** 3.10.*

### Installing CARLA
Follow [these](https://carla.readthedocs.io/en/0.9.15/start_quickstart/) instructions to install the CARLA simulation environment. For a quick setup, you may also [run CARLA through Docker](https://carla.readthedocs.io/en/0.9.15/build_docker/).

### Setting up Environment
```
git clone https://github.com/RyangDiaz/carla-yolo-dataset-generator.git
cd carla-yolo-dataset-generator
conda env create -f environment.yml
```

