import utils
from models import build

import torch
import torch.nn.functional as F
from torch import nn

import random


class CLIP(nn.Module):
    def __init__(
        self, tokenizer=None, config=None,
        **kwargs,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        embed_dim = config["embed_dim"]

        self.visual_encoder = build.vision_encoder(
            config, config.vision_encoder, config.adapter_append
        )
        vision_width = self.visual_encoder.embed_dim
        self.text_encoder = build.text_encoder(
            config, config.text_encoder, config.adapter_append
        )

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config["temp"])

        if config.freeze_vision_encoder:
            utils.freeze_model(self.visual_encoder)

        if config.freeze_text_encoder:
            utils.freeze_model(self.text_encoder)

        if config.freeze_proj:
            utils.freeze_model(self.vision_proj)
            utils.freeze_model(self.text_proj)

        if config.unlock_layernorm:
            if config.unlock_layernorm in ("vision_only", True):
                for name, param in self.visual_encoder.named_parameters():
                    if "norm" in name.lower():
                        param.requires_grad = True
            if config.unlock_layernorm in ("language_only", True):
                for name, param in self.text_encoder.named_parameters():
                    if "LayerNorm" in name:
                        param.requires_grad = True

        if config.unlock_dense:
            for name, param in self.visual_encoder.named_parameters():
                if "mlp" in name.lower():
                    param.requires_grad = True
            for name, param in self.text_encoder.named_parameters():
                if "dense" in name:
                    param.requires_grad = True

        if config.unlock_attn:
            for name, param in self.visual_encoder.named_parameters():
                if "attn" in name.lower():
                    param.requires_grad = True
            for name, param in self.text_encoder.named_parameters():
                if "attention" in name:
                    param.requires_grad = True

        if config.unlock_random:
            bert_choices = (
                "query",
                "key",
                "value",
                "attention.output.dense",
                "intermediate.dense",
            )
            for block in self.text_encoder.encoder.layer:
                parameter_to_unlock = random.choice(bert_choices)
                for name, param in block.named_parameters():
                    if parameter_to_unlock in name.lower():
                        param.requires_grad = True

            vit_choices = (
                "proj",
                "fc1",
                "fc2",
            )
            for block in self.visual_encoder.blocks:
                parameter_to_unlock = random.choice(vit_choices)
                for name, param in block.named_parameters():
                    if parameter_to_unlock in name.lower():
                        param.requires_grad = True

        if config.add_adapter:
            last_lm_layer = self.text_encoder.encoder.layer[-1]
            for param in last_lm_layer.parameters():
                param.requires_grad = True

            last_vit_layer = self.visual_encoder.blocks[-1]
            for param in last_vit_layer.parameters():
                param.requires_grad = True

            for param in self.visual_encoder.norm.parameters():
                param.requires_grad = True

        if config.conventional_adapter.insert:
            if config.conventional_adapter.insert in ("vision_only", True):
                for name, param in self.visual_encoder.named_parameters():
                    if "adapter" in name:
                        param.requires_grad = True

            if config.conventional_adapter.insert in ("language_only", True):
                for name, param in self.text_encoder.encoder.named_parameters():
                    if "adapter" in name:
                        param.requires_grad = True

        if config.bitfit:
            if config.bitfit in ("vision_only", True):
                for name, param in self.visual_encoder.named_parameters():
                    if "bias" in name:
                        param.requires_grad = True
            if config.bitfit in ("language_only", True):
                for name, param in self.text_encoder.named_parameters():
                    if "bias" in name:
                        param.requires_grad = True

        if config.always_freeze:
            for idx_always_locked in config.always_freeze.visual_encoder:
                for block_idx, block in enumerate(self.visual_encoder.blocks):
                    if idx_always_locked == block_idx:
                        for name, param in block.named_parameters():
                            param.requires_grad = False

            for idx_always_locked in config.always_freeze.text_encoder:
                for block_idx, block in enumerate(self.text_encoder.encoder.layer):
                    if idx_always_locked == block_idx:
                        for name, param in block.named_parameters():
                            param.requires_grad = False

        trainable_params = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        total_params = sum(param.numel() for param in self.parameters())
        print(
            "percentage_trainable={}".format(
                round(trainable_params / total_params * 100, 2)
            )
        )
        print("num trainable={}".format(trainable_params))
        print("total params={}".format(total_params))

    def forward(self, image, text, return_dict=False):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds = self.visual_encoder(image)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text_output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        sim_i2t = image_feat @ text_feat.T / self.temp
        sim_t2i = text_feat @ image_feat.T / self.temp

        with torch.no_grad():
            sim_targets = torch.zeros(sim_i2t.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        if return_dict:
            return {
                "losses": {
                    "loss_ita": loss_ita,
                }
            }
        return loss_ita

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient



@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output