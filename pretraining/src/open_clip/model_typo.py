from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, Union, List
import json
import logging
import os.path as osp
from pathlib import Path
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch import TensorType
import torch.nn.functional as F

import transformers
from transformers import AutoTokenizer, AutoConfig, T5ForConditionalGeneration
from torchvision.ops import roi_align

from .hf_model import MeanPooler

from .model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict,\
    resize_pos_embed, get_cast_dtype, resize_text_pos_embed, set_model_preprocess_cfg
from .coca_model import CoCa
from .loss import ClipLoss, DistillClipLoss, CoCaLoss, SigLipLoss
from .openai import load_openai_model
from .pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained,\
    list_pretrained_tags_by_model, download_pretrained_from_hf
from .transform import image_transform_v2, AugmentationCfg, PreprocessCfg, merge_preprocess_dict, merge_preprocess_kwargs
from .tokenizer import HFTokenizer, SimpleTokenizer, DEFAULT_CONTEXT_LENGTH

huggingface_cache_dir = None

HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def create_model_open_clip(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        **model_kwargs,
):
    def get_model_config(model_name):
        if model_name in _MODEL_CONFIGS:
            return deepcopy(_MODEL_CONFIGS[model_name])
        else:
            return None

    def _get_hf_config(model_id, cache_dir=None):
        config_path = download_pretrained_from_hf(model_id, filename='open_clip_config.json', cache_dir=cache_dir)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config

    def load_state_dict(checkpoint_path: str, map_location='cpu'):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, torch.jit.ScriptModule):
            state_dict = checkpoint.state_dict()
            for key in ["input_resolution", "context_length", "vocab_size"]:
                state_dict.pop(key, None)
        else:
            state_dict = checkpoint
        if next(iter(state_dict.items()))[0].startswith('module'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        return state_dict

    def load_checkpoint(model, checkpoint_path, strict=True):
        if Path(checkpoint_path).suffix in ('.npz', '.npy'):
            from .big_vision import load_big_vision_weights
            load_big_vision_weights(model, checkpoint_path)
            return {}

        state_dict = load_state_dict(checkpoint_path)
        # detect old format and make compatible with new format
        if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
            state_dict = convert_to_custom_text_state_dict(state_dict)
        # If loading a non-SigLIP model for SigLIP training. See https://github.com/mlfoundations/open_clip/issues/712
        if 'logit_bias' not in state_dict and model.logit_bias is not None:
            state_dict["logit_bias"] = torch.zeros_like(state_dict["logit_scale"])
        # Certain text transformers no longer expect position_ids after transformers==4.31
        position_id_key = 'text.transformer.embeddings.position_ids'
        if position_id_key in state_dict and not hasattr(model, position_id_key):
            del state_dict[position_id_key]
        resize_pos_embed(state_dict, model)
        resize_text_pos_embed(state_dict, model)
        incompatible_keys = model.load_state_dict(state_dict, strict=strict)
        return incompatible_keys

    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix:
        model_id = model_name[len(HF_HUB_PREFIX):]
        checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
        config = _get_hf_config(model_id, cache_dir)
        preprocess_cfg = merge_preprocess_dict(preprocess_cfg, config['preprocess_cfg'])
        model_cfg = config['model_cfg']
        pretrained_hf = False  # override, no need to load original HF text weights
    else:
        model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
        checkpoint_path = None
        model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(
            model_name,
            precision=precision,
            device=device,
            cache_dir=cache_dir,
        )
    else:
        model_cfg = model_cfg or get_model_config(model_name)
        if model_cfg is not None:
            logging.info(f'Loaded {model_name} model config.')
        else:
            logging.error(f'Model config for {model_name} not found.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if force_image_size is not None:
            # override model config's image size
            model_cfg["vision_cfg"]["image_size"] = force_image_size

        is_timm_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})
        if pretrained_image:
            if is_timm_model:
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
        cast_dtype = get_cast_dtype(precision)
        is_hf_model = 'hf_model_name' in model_cfg.get('text_cfg', {})
        if is_hf_model:
            # load pretrained weights for HF text model IFF no CLIP weights being loaded
            model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf and not pretrained
        custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_model

        model_cfg = dict(model_cfg, **model_kwargs)  # merge cfg dict w/ kwargs (kwargs overrides cfg)
        if custom_text:
            if "multimodal_cfg" in model_cfg:
                model = CoCa(**model_cfg, cast_dtype=cast_dtype)
            else:
                model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype)

        if precision in ("fp16", "bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            # manual mixed precision that matches original OpenAI behaviour
            if is_timm_model:
                # FIXME this is a bit janky, create timm based model in low-precision and
                # then cast only LayerNormFp32 instances back to float32 so they don't break.
                # Why? The convert_weights_to_lp fn only works with native models.
                model.to(device=device, dtype=dtype)
                from .transformer import LayerNormFp32

                def _convert_ln(m):
                    if isinstance(m, LayerNormFp32):
                        m.weight.data = m.weight.data.to(torch.float32)
                        m.bias.data = m.bias.data.to(torch.float32)
                model.apply(_convert_ln)
            else:
                model.to(device=device)
                convert_weights_to_lp(model, dtype=dtype)
        elif precision in ("pure_fp16", "pure_bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            model.to(device=device, dtype=dtype)
        else:
            model.to(device=device)

        pretrained_loaded = False
        if pretrained:
            checkpoint_path = ''
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
                preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg)
            elif osp.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                load_checkpoint(model, checkpoint_path)
            else:
                error_str = (
                    f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                    f' Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
                logging.warning(error_str)
                raise RuntimeError(error_str)
            pretrained_loaded = True
        elif has_hf_hub_prefix:
            logging.info(f'Loading pretrained {model_name} weights ({checkpoint_path}).')
            load_checkpoint(model, checkpoint_path)
            pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
                f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, 'image_size', None) is not None:
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg['size'] = model.visual.image_size
    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))

    return model

@dataclass
class TypoCLIPVisionCfg:
    image_size: int = 224

    vision_encoder_name: str = None

@dataclass
class TypoCLIPTextCfg:
    text_encoder_name: str = None
    proj_type: str = None
    color_special_token: bool = False,
    font_special_token: bool = False,
    font_ann_path: str = None,
    color_ann_path: str = None,

class TypoTextEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int,
        proj_type: str = None,
        color_special_token: bool = False,
        font_special_token: bool = False,
        font_ann_path: str = None,
        color_ann_path: str = None,
    ):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=huggingface_cache_dir,
        )
        self.encoder = T5ForConditionalGeneration.from_pretrained(
            model_name_or_path, cache_dir=huggingface_cache_dir,
        ).get_encoder()
        d_model = getattr(self.config, "d_model")

        self.pooler = MeanPooler()

        if (d_model == output_dim) and (proj_type is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=False),
            )

        with open(font_ann_path, 'r') as f:
            idx_font_dict = json.load(f)
        with open(color_ann_path, 'r') as f:
            idx_color_dict = json.load(f)

        if color_special_token or font_special_token:
            font_token = [f'<font-{i}>' for i in range(len(idx_font_dict))]
            color_token = [f'<color-{i}>' for i in range(len(idx_color_dict))]
            additional_special_tokens = []
            if color_special_token:
                additional_special_tokens = color_token
            if font_special_token:
                additional_special_tokens += font_token
            self.tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
            print(f"add tokens! currently model contains {len(self.tokenizer)} tokens")
            self.encoder.resize_token_embeddings(len(self.tokenizer))

    def lock(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, prompts: List[str], max_length=512, bbox_dict=None, text_attn_mask=None):
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        device = self.encoder.device
        attention_mask = text_inputs.attention_mask.to(device) if text_attn_mask is None else text_attn_mask.to(device, dtype=text_inputs.attention_mask.dtype)

        prompt_embeds = self.encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )

        # box level contrastive loss
        '''
            bbox_dict: {
                'bbox_index': [N],
                'bbox_list': [N, max_length]
            }
            
            bbox_features: [N, max_length, dim]
        '''
        bbox_index = bbox_dict['bbox_index']
        bbox_list = bbox_dict['bbox_list'].view(*bbox_dict['bbox_list'].shape, 1)
        bbox_features = prompt_embeds[0][bbox_index]
        pooled_out = (bbox_features * bbox_list).sum(dim=1) / torch.max(bbox_list.sum(dim=1), torch.tensor(1).to(bbox_list.device))
        projected = self.proj(pooled_out)

        return projected

class TypoVisionEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int,
    ):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.encoder = torch.hub.load('facebookresearch/dinov2', model_name_or_path)
        d_model = self.encoder.patch_embed.proj.out_channels

        self.encoder.mask_token.requires_grad = False
        if d_model != output_dim:
            self.encoder.head = nn.Linear(d_model, output_dim, bias=False)
        self.d_model = d_model
        self.output_dim = output_dim


    def forward(self, x: torch.FloatTensor, bbox_dict=None):
        # box level contrastive loss
        '''
            bbox_dict: {
                'bbox_index': [N],
                'bbox_list': [N, max_length]
            }
            
            bbox_features: [N, max_length, dim]
        '''
        bbox_index = bbox_dict['bbox_index']
        bbox_list = bbox_dict['bbox_list'].view(*bbox_dict['bbox_list'].shape, 1)

        outputs = self.encoder(x, is_training=True)
        outputs = outputs['x_norm_patchtokens']

        bbox_features = outputs[bbox_index]
        pooled_out = (bbox_features * bbox_list).sum(dim=1) / torch.max(bbox_list.sum(dim=1), torch.tensor(1).to(bbox_list.device))

        return pooled_out

    def lock(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        if self.d_model != self.output_dim:
            for param in self.encoder.head.parameters():
                param.requires_grad = True

    def unlock_layer(self, num_layers):
        num_blocks = len(self.encoder.blocks)
        for i in range(num_blocks - num_layers, num_blocks):
            for name, param in self.encoder.blocks[i].named_parameters():
                param.requires_grad = True
                print(f"unlocking param {name}")


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: TypoCLIPVisionCfg,
):
    vision_tower = TypoVisionEncoder(
        vision_cfg.vision_encoder_name,
        output_dim=embed_dim,
    )
    return vision_tower

def _build_text_tower(
        embed_dim: int,
        text_cfg: TypoCLIPTextCfg,
):
    text_tower = TypoTextEncoder(
        text_cfg.text_encoder_name,
        output_dim=embed_dim,
        proj_type=text_cfg.proj_type,
        color_special_token=text_cfg.color_special_token,
        font_special_token=text_cfg.font_special_token,
        font_ann_path=text_cfg.font_ann_path,
        color_ann_path=text_cfg.color_ann_path,
    )
    return text_tower

class TypoCLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: TypoCLIPVisionCfg,
            text_cfg: TypoCLIPTextCfg,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            multi_logit_scale: int = None,
            multi_logit_scale_v: float = np.log(1 / 0.01),
    ):
        super().__init__()

        self.vision_tower = _build_vision_tower(embed_dim, vision_cfg)
        self.text_tower = _build_text_tower(embed_dim, text_cfg)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

        if multi_logit_scale is not None:
            self.multi_logit_scale = nn.Parameter(torch.ones([multi_logit_scale]) * multi_logit_scale_v)
        else:
            self.multi_logit_scale = None

    def set_grad_checkpointing(self):
        try:
            self.text_tower.encoder.gradient_checkpointing_enable()
        except:
            print("Text encoder gradient checkpointing not enabled!")

    def lock_image_tower(self):
        self.vision_tower.lock()

    def unlock_image_layer(self, num_layers):
        self.vision_tower.unlock_layer(num_layers)

    def lock_text_tower(self):
        self.text_tower.lock()

    def encode_image(self, image, bbox_dict: Optional[Dict] = None, normalize: bool = False):
        features = self.vision_tower(image, bbox_dict=bbox_dict)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, bbox_dict: Optional[Dict] = None, text_attn_mask = None, normalize: bool = False):
        features = self.text_tower(text, bbox_dict=bbox_dict, text_attn_mask=text_attn_mask)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(
        self,
        image: torch.FloatTensor = None,
        text: List[str] = None,
        vision_bbox_dict: Optional[Dict] = None,
        text_bbox_dict: Optional[Dict] = None,
        text_attn_mask = None,
    ):
        image_features = self.encode_image(image, bbox_dict=vision_bbox_dict, normalize=True) if image is not None else None
        text_features = self.encode_text(text, bbox_dict=text_bbox_dict, text_attn_mask=text_attn_mask, normalize=True) if text is not None else None

        out_dict = {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.logit_scale.exp(),
            "multi_logit_scale": [logit.exp() for logit in self.multi_logit_scale] if self.multi_logit_scale is not None else None,
        }
        if self.logit_bias is not None:
            out_dict['logit_bias'] = self.logit_bias
        return out_dict
