# !/bin/bash
for MAP in 'Town01' 'Town02' 'Town03' 'Town05' 'Town06HD'
do
    for i in {1..5} 
    do
        python collect_yolo_data.py \
            --map $MAP \
            --save \
            --num_save 50 \
            --num_detections_save 2
    done
done