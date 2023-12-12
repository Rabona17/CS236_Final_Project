import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from functools import partial
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import boto3
import webdataset as wds
import json
from PIL import Image
# from pudb.remote import set_trace
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel#, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
import io
from datasets import load_from_disk
# from save_datasets import load_and_concatenate_datasets
if is_wandb_available():
    import wandb
import torch.nn as nn
pretrained_model_name_or_path = 'CompVis/stable-diffusion-v1-4'
noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer",
)
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)

vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae"
)
unet = UNet2DConditionModel.from_pretrained(
    '/table_efs/users/rabonagy/cs236_final/sd-768-model-default-vae/checkpoint-120500/', subfolder="unet"
)

# Freeze vae and text_encoder and set unet to trainable
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
pipeline = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    safety_checker=None,

    # torch_dtype="fp16",
)
pipeline = pipeline.to(torch.device('cuda:1'))
ds  = load_from_disk('/table_efs/users/rabonagy/cs236_final/train_dataset/').select(range(999000,1000000))
generator = torch.Generator(device=torch.device('cuda:1')).manual_seed(0)
with torch.no_grad():
    for idx in range(175,500):
        original_words_data=json.loads(ds[idx]['annotations'])['original_words_data']
        title = original_words_data[0]['content']
        inputs_ids = tokenizer(title, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(text_encoder.device)
        
        prompt_emb = text_encoder(inputs_ids)[0]
        max_length = prompt_emb.shape[1]
        uncond_input = tokenizer(
            [''],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        attention_mask = uncond_input.attention_mask.to(text_encoder.device)
        negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(text_encoder.device),
                attention_mask=None,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]
        image = pipeline(prompt_embeds=prompt_emb, num_inference_steps=20, generator=generator,negative_prompt_embeds=negative_prompt_embeds,height=768,width=768).images[0]
        # image = pipeline(title, num_inference_steps=20, generator=generator).images[0]
        image.save(f'sd_vanilla/train_{idx}_prompt_emb.png')