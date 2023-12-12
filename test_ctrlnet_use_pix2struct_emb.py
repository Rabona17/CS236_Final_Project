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
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionControlNetPipeline, ControlNetModel,UniPCMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel#, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
import io
import re
import json
from datasets import load_from_disk
# from save_datasets import load_and_concatenate_datasets
if is_wandb_available():
    import wandb
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from PIL import Image
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


def extract_boxes_with_labels(data):
    all_boxes_with_labels = []
    
    # Regular expression to match the pattern 'label<x><y><x><y>'
    pattern = re.compile(r'([a-zA-Z]+)(<\d+><\d+><\d+><\d+>)')

    # Find all matches
    matches = pattern.findall(data)

    for match in matches:
        label = match[0]
        numbers = [int(x) for x in re.findall(r'<(\d+)>', match[1])]

        if numbers and len(numbers) == 4:  # Check if the numbers list is not empty and contains exactly four numbers
            all_boxes_with_labels.append((label, numbers))
        else:  # If not, append the label with a default box of [0, 0, 0, 0]
            all_boxes_with_labels.append((label, [0, 0, 0, 0]))
    
    return all_boxes_with_labels

if __name__=='__main__':
    device=torch.device('cuda:1')
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
        '/table_efs/users/rabonagy/cs236_final/sd-768-model-default-vae-cond-on-layout-1024/checkpoint-77000/', subfolder="unet"
    )
    controlnet = ControlNetModel.from_pretrained('/table_efs/users/rabonagy/cs236_final/ctrlnet-768-model-cond-on-layout-2048-77000/checkpoint-675500', subfolder='controlnet')
    # Freeze vae and text_encoder and set unet to trainable
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
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,

        # torch_dtype="fp16",
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    vl_encoder = vl_encoder.to(device)
    ds  = load_from_disk('/table_efs/users/rabonagy/cs236_final/train_dataset/').select(range(99900,1000000))
    with torch.no_grad():
        for idx in range(0,500):
            original_words_data=json.loads(ds[idx]['annotations'])['original_words_data']
            title = original_words_data[0]['content']
            inputs_ids = tokenizer(title, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(text_encoder.device)
            layout=ds[idx]['layout']
            image_cond = render_layout_with_text(layout,original_words_data)
            image_cond_no_text = render_layout_with_text(layout,[])
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
            generator = torch.Generator(device).manual_seed(0)
            image_cond = image_cond.resize((768,768))

            image = pipeline(prompt_embeds=prompt,image=image_cond,num_inference_steps=20, generator=generator,negative_prompt_embeds=negative_prompt_embeds).images[0]
            concat_images(image,image_cond,f'ctrlnet_vis_use_pix2struct/{idx}.png')
            image_cond_no_text.save(f'ctrlnet_vis_use_pix2struct/layout_only_{idx}.png')