#!/bin/bash -x
export PYTHONPATH=${PWD}/src

python preprocess/align/align_dataset_mtcnn.py \
    --input_dir /mnt/sdb/vggface3 \
    --output_dir /mnt/sdb/vggface3_align \
    --image_size 182 \
    --margin 44 \
    --random_order \
    --thread_num 3 \
    --gpu_memory_fraction 0.88
