#!/bin/bash

python -m torch.distributed.launch --master_port=48770 --nproc_per_node=1 \
    --use_env zero_shot_retrieval.py \
    --config Retrieval_AdjustableCLIP_Flickr \
    --output_dir ./storage/lilt_cache/lilt_example/output_flickr \
    --checkpoint ./storage/lilt_cache/lilt_example/checkpoint_14.pth \
    --evaluate \
    --overrides text_encoder=base vision_encoder=base \
    conventional_adapter.insert=True \
    conventional_adapter.reduction_factor=4