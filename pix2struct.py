from PIL import Image
import requests
import transformers
from transformers import AutoProcessor, Pix2StructForConditionalGeneration, Pix2StructConfig, XLMRobertaTokenizer
import torch
import re
import os
from transformers import Pix2StructForConditionalGeneration, Pix2StructConfig, AutoProcessor
from typing import Any, List, Optional, Union, Dict
import requests
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from datasets import load_dataset
import collections
import itertools
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
import editdistance
from PIL import Image, ImageDraw
from torch import tensor
import argparse
from transformers.models.pix2struct.image_processing_pix2struct import render_text
# import gradio as gr 

class Pix2StructForS4(Pix2StructForConditionalGeneration):

    config_class = Pix2StructConfig
    base_model_prefix = "pix2struct"

    def __init__(self, config: Pix2StructConfig):
        super().__init__(config)
        self.processor = AutoProcessor.from_pretrained(self.config.name_or_path if self.config.name_or_path !='' else 'google/pix2struct-base')
        # self.add_special_tokens(["<sep/>"])
        # self.tokenizer = self.processor.tokenizer

    def add_special_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.processor.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(list_of_tokens))})
        if newly_added_num > 0:
            self.resize_token_embeddings(len(self.processor.tokenizer))

    def save_pretrained(self, save_directory):
        # First, save the original model.
        super().save_pretrained(save_directory)
        # Now let's save the processor
        self.processor.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load the model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        # Load the processor
        model.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)
        return model

def render_header(image: np.ndarray, header: str, **kwargs):
    """
    Renders the input text as a header on the input image.

    Args:
        image (`np.ndarray`):
            The image to render the header on.
        header (`str`):
            The header text.
        data_format (`Union[ChannelDimension, str]`, *optional*):
            The data format of the image. Can be either "ChannelDimension.channels_first" or
            "ChannelDimension.channels_last".

    Returns:
        `np.ndarray`: The image with the header rendered.
    """

    # Convert to PIL image if necessary
    header_image = render_text(header, **kwargs)
    new_width = image.width

    new_height = image.height
    if header_image.width>new_width:
        header_image = header_image.resize((new_width, int(header_image.height*new_width/header_image.width)))
    new_header_height = header_image.height

    new_image = Image.new("RGB", (new_width, new_height + new_header_height), "white")
    new_image.paste(header_image, (0, 0))
    new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))

    # Convert back to the original framework if necessary
    

    return new_image