[![Paper](http://img.shields.io/badge/paper-arxiv.2303.11866-B31B1B.svg)](https://arxiv.org/abs/2303.11866)
[![Conference](https://img.shields.io/badge/ICLR-2023-blue)](https://iclr.cc/virtual/2023/poster/11712)
# LilT
Contrastive Aligned of Vision to Language Through Parameter-Efficient Transfer Learning [ICLR 23]
# Setup
## Dependencies
```
conda env create --name lilt --file environment.yaml
```
## Data
See individual sections below for instructions.
## Weights
The links to the weights all point to Google Drive. 
To download them without a browser, do `pip install gdown` use the ID of the file you want to download. For example, to download model weights at `https://drive.google.com/file/d/1mdXQK9Jidk97FrLR2Bqzhbuptt07BPo4/view?usp=drive_link`, you would use `gdown 1mdXQK9Jidk97FrLR2Bqzhbuptt07BPo4`.
The weights were saved with `torch.save()` and can be loaded with `torch.load()`.
- [500K Training Pairs](https://drive.google.com/drive/folders/1hfNlZVohcdoWnx-rrQG-BXUHcYjTDNW1?usp=drive_link)
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
## Classification
First, set up the ImageNetv2 dataset as described [here](https://github.com/modestyachts/ImageNetV2). 
Next, edit `classification.py`, specifically this part:
```python
test_dataset = ImageNetV2Dataset(
    location="./storage/10/imagenet_v2", transform=test_transform
)
```
to point to wherever you downloaded ImageNetV2.
To run multimodal classification, you can use a command like the following:
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py \
--config Retrieval_AdjustableCLIP_Flickr \
--output_dir ./clf_output \
 --checkpoint ./storage/lilt_example/checkpoint_14.pth \ --evaluate \
  --overrides text_encoder=base vision_encoder=base
```
 The value of the `--config` flag is not a typo, we just reuse the same config for classification and retrieval.

## Retrieval
Follow the instructions in [codezakh/SIMLA#zero-shot-flickr](https://github.com/codezakh/SIMLA#zero-shot-flickr) to set up the data for Flickr.
The only change is that you should edit `configs-v2/Retrieval_AdjustableCLIP_Flickr.yaml` to point to the downloaded dataset rather than `configs/Retrieval_flickr.yaml`.


See `examples/evaluate_{clip, lilt, lit}.sh` for evaluation scripts.

## Multilingual Retrieval
Download the XD10 dataset from [official repo](https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10). 
Then, run the script `process_xd10.py`, making sure to edit the paths at the top of the file:
```python
XTD10_DIR = Path("/home/zkhan/Cross-lingual-Test-Dataset-XTD10/XTD10")
COCO_TRAIN_DIR = Path("./storage/10/coco2014/train2014")
OUTPUT_DIR = Path("./storage/10/multilingual_coco2014_xtd10")
```
so that `XD10_DIR` and `COCO_TRAIN_DIR` point to where you have downloaded the respective datasets.

You can then evaluate a pretrained model like this:
```bash
python -m torch.distributed.launch --master_port=40770 \
--nproc_per_node=1 \
--use_env zero_shot_retrieval.py \
--config Retrieval_AdjustableCLIP_COCO \
--output_dir ./eval_output \
--checkpoint ./path/to/checkpoint.pth \
--evaluate --overrides text_encoder=base_multilingual \
vision_encoder=base \
test_file=$XD10_OUTPUT_DIR/val.json"
```
where `XD10_OUTPUT_DIR` is the place you told the `process_xd10.py` script to put the preprocessed dataset.

To evaluate on a specific language as done in the paper, change `test_file=$XD10_OUTPUT_DIR/val.json` to `test_file=$XD10_OUTPUT_DIR/val_{lang_abbrv}.json` where `lang_abbrv` is one of the following:
```
LANGUAGES = ("es", "it", "ko", "pl", "ru", "tr", "zh")
```



# Citation
```
@inproceedings{ContrastiveAlignmentVisionFu2023,
  title = {Contrastive Alignment of Vision to Language Through Parameter-Efficient Transfer Learning},
  booktitle = {The Eleventh International Conference on Learning Representations},
  author = {Khan, Zaid and Fu, Yun},
  year = {2023},
}

```

# Acknowledgements
This code borrows heavily from https://github.com/salesforce/ALBEF.
