# LilT
Contrastive Aligned of Vision to Language Through Parameter-Efficient Transfer Learning [ICLR 23]

**Note: Code release is a work in progress, and will be complete soon. In the meantime, follow the dataset and environment setup in zaidkhan.me/SIMLA, and check the `experiments/` directory**.

# Setup
## Dependencies
```
conda env create --name lilt --file environment.yaml
```
## Data
See individual sections below for instructions.
## Weights
# Pretraining
## Pretraining Data
The data format used for pretraining looks like this:
```
pretraining_data: List[Dict]  = [
    {
        "image": "/media/coco2017/coco-images/000000203564.jpg",
        "image_id": 203564,
        "caption": "A bicycle replica with a clock as the front wheel."
    },
    {
        "image": "/media/coco2017/coco-images/000000322141.jpg",
        "image_id": 322141,
        "caption": "A room with blue walls and a white sink and door."
    }, 
    ...
]
```
The `image_id` does not need to correspond to anything for the pretraining (in this example, it is the COCO image IDs). 
The `image` field needs to contain an _absolute_ path. 
This means the path should start with a slash `/`. 
You can download examples of pretraining files from [Salesforce Research](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/json_pretrain.zip), but note that the paths for each image will need to be changed to match your local setup.

You can download COCO from https://cocodataset.org/#home.

The other pretraining datasets can be downloaded from Huggingface.
- [Conceptual Captions](https://huggingface.co/datasets/conceptual_captions)
- [SBU Captions](https://huggingface.co/datasets/sbu_captions)

## Pretraining Commands
See `examples/pretrain_{clip, lilt, lit}.sh`.
In general, the pretraining commands look something like this:
```
python -m torch.distributed.launch --master_port=43770 --nproc_per_node=4 \
    --use_env PretrainHydra.py --config CLIPAdjustable \
    --output_dir ./storage/lilt_cache/clip_example \
    --overrides text_encoder=base vision_encoder=base \
    +save_last_only=True disable_wandb=False
```
The model checkpoints will be placed in `--output_dir`.
The `--overrides` flag can be used to specify overrides for items in the configuration, following the syntax of [Hydra](https://github.com/facebookresearch/hydra).
For example, the default size (in the config) for the text encoder is `small`, but we are overriding it to `base` from the command line.

The code was tested with V100, A6000, and A100 GPUs. 
You will need around 32GB of GPU memory (with 4x GPUs) to match the training settings of the CLIP model exactly, but the LilT and LiT models can be trained on 4x GPUs for much less memory. 
Of course, you can always lower the batch size if you want to train on a single GPU.

# Evaluation
**WIP**
## Classification
## Retrieval

# Citations

# Acknowledgements
This code borrows heavily from https://github.com/salesforce/ALBEF.
