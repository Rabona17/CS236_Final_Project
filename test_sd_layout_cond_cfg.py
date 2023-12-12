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
from train_sd import *
def concat_images(image1, image2, save_path):
    """
    Concatenate two images side by side and save the result.

    :param image1: The first PIL image.
    :param image2: The second PIL image.
    :param save_path: Path to save the concatenated image.
    """
    # Get dimensions of the first image
    width1, height1 = image1.size

    # Get dimensions of the second image
    width2, height2 = image2.size

    # Create a new image with a width that is the sum of both images and the height of the tallest image
    new_width = width1 + width2
    new_height = max(height1, height2)

    # Create a new blank image with the calculated dimensions
    new_image = Image.new('RGB', (new_width, new_height))

    # Paste the first image onto the new image
    new_image.paste(image1, (0, 0))

    # Paste the second image onto the new image, right next to the first image
    new_image.paste(image2, (width1, 0))

    # Save the concatenated image
    new_image.save(save_path)
vl_encoder = Pix2StructForS4.from_pretrained(f"/table_efs/users/rabonagy/WebRenderStrongSupervision/donut/result/train_s4_pix2struct_layout_ocr/debug/ckpt_9").encoder
vl_encoder.requires_grad_(False)
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-base",is_vqa=False)
processor.is_vqa=False
tokens_to_add = set()
S4_special_tokens = ["<sep/>"]
bbox_unique_tokens = [f"<{k}>" for k in
                      range(1281)]  # splits to 1k positions  not needed if using bbox_head
tokens_to_add.update(S4_special_tokens + bbox_unique_tokens)
processor.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(tokens_to_add))})
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
    '/table_efs/users/rabonagy/cs236_final/sd-768-model-default-vae-cond-on-layout-2048-cfg/checkpoint-83500', subfolder="unet"
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
vl_encoder = vl_encoder.to(torch.device('cuda:0'))
pipeline = pipeline.to(torch.device('cuda:0'))

ds  = load_from_disk('/table_efs/users/rabonagy/cs236_final/train_dataset/').select(range(99900,1000000))
generator = torch.Generator(device=torch.device('cuda:0')).manual_seed(0)
with torch.no_grad():
    for idx in range(10):
        original_words_data=json.loads(ds[idx]['annotations'])['original_words_data']
        title = original_words_data[0]['content']
        inputs_ids = tokenizer(title, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(text_encoder.device)
        layout=ds[idx]['layout']
        image_cond = render_layout_with_text(layout,original_words_data)
        image_cond = render_header(image_cond, title, font_path='/table_efs/users/rabonagy/WebRenderStrongSupervision/donut/Arial.TTF',text_size=32)
        inputs = processor(images=image_cond, return_tensors="pt")
        inputs = {name: tensor.to(vl_encoder.device) for name, tensor in inputs.items()}
        prompt_layout = vl_encoder(**inputs)[0]
        prompt_emb = text_encoder(inputs_ids)[0]
        prompt = torch.cat([prompt_emb,prompt_layout],dim=1)
        max_length = prompt_emb.shape[1]
        uncond_input = tokenizer(
            [''],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(text_encoder.device),
                attention_mask=None,
        )
        image_cond_neg = Image.new('RGB', (1280, 1280), color='white')
        image_cond_neg = render_header(image_cond_neg, '', font_path='/table_efs/users/rabonagy/WebRenderStrongSupervision/donut/Arial.TTF',text_size=32)
        negative_prompt_embeds = negative_prompt_embeds[0]
        
        neg_inputs = processor(images=image_cond_neg, return_tensors="pt")
        neg_inputs = {name: tensor.to(vl_encoder.device) for name, tensor in neg_inputs.items()}
        neg_prompt_layout = vl_encoder(**neg_inputs)[0]
        negative_prompt_embeds = torch.cat([negative_prompt_embeds,neg_prompt_layout],dim=1)
        image = pipeline(prompt_embeds=prompt, num_inference_steps=20, generator=generator,negative_prompt_embeds=negative_prompt_embeds,height=768,width=768).images[0]
        concat_images(image, image_cond.resize((768,768)),f'sd_cond_layout_train_with_cfg/train_{idx}_prompt_emb.png')
        # image.save(f'sd_cond_layout_cfg/train_{idx}_prompt_emb.png')