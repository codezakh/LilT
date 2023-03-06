#!/bin/bash

# TORCH_HOME can be set to whatever you want. Remove
# this line if you want to use the default location used \
# by PyTorch.
export TORCH_HOME=./storage/4/torch_home 
# Only needed if running on A6000s or A100s. Prevents
# some weird / hard to debug errors.
export NCCL_P2P_DISABLE=1 
OUTPUT_DIR=./storage/lilt_cache/lilt_example
python -m torch.distributed.launch --master_port=43770 --nproc_per_node=4 \
	--use_env PretrainHydra.py --config CLIPAdjustable \
	--output_dir $OUTPUT_DIR \
	--overrides text_encoder=base vision_encoder=base \
	freeze_text_encoder=True freeze_vision_encoder=True \
	unlock_layernorm=True batch_size=128 +save_last_only=True \
    conventional_adapter.insert=True \
    conventional_adapter.reduction_factor=4 disable_wandb=False