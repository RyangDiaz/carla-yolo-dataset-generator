# !/bin/bash
for MAP in 'Town01' 'Town02' 'Town03' 'Town04' 'Town05'
do
    for i in {1..5} 
    do
        python bb_semantic.py --map $MAP
    done
done