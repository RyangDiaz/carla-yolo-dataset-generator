# !/bin/bash
for MAP in 'Town05' 'Town06' # 'Town01' 'Town02' 'Town03' 'Town04' 'Town05'
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