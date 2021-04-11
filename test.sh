#!/bin/bash
nohup python test.py --checkpoint path/to/checkpoint \
--img_folder path/to/img_folder \
--width 160 --height 48 \
--output_folder path/to/output_folder \
--gpu 0 --batch_size 64 \
>> infer_one_model_one_gpu.out &