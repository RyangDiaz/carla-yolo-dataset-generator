# !/bin/bash
bash collect_data.sh
python convert_dataset.py
python train_yolo.py