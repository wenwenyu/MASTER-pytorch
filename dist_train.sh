#!/bin/bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=4 \
--master_addr=127.0.0.1 --master_port=5555 \
train.py -c configs/config.json -d 0,1,2,3 --local_world_size 4
