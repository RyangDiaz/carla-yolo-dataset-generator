# !/bin/bash
collect_data () {
    python collect_yolo_data.py \
        --map $1 \
        --save \
        --num_save 250 \
        --num_detections_save 2
}

for MAP in 'Town01' 'Town02' 'Town03' 'Town04' 'Town05'
do
    PYTHON_RETURN = 1
    until [ $PYTHON_RETURN == 0 ]; do
        collect_data $MAP
        PYTHON_RETURN=$?
        echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}${NC}" >&2
        sleep 2
    done
done

# for MAP in 'Town01' 'Town02' 'Town03' 'Town04' 'Town05'
# do
#     python collect_yolo_data.py \
#         --map $MAP \
#         --save \
#         --num_save 250 \
#         --num_detections_save 2
# done