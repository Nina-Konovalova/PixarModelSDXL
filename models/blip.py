from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn.functional as nnf
import numpy as np

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration




class BlipModel(torch.nn.Module):
    def __init__(self, processor: str, model: str, text: str):
        """
        Initialize the BLIP model for conditional generation.

        Args:
            processor (str): The name or path of the BLIP processor.
            model (str): The name or path of the BLIP model.
            text (str): The text prompt for image captioning.
        """
        super().__init__()
        self.processor = BlipProcessor.from_pretrained(processor)
        self.model = BlipForConditionalGeneration.from_pretrained(model)
        self.text = text

    def forward(self, image: torch.Tensor, text: Optional[str] = None) -> str:
        """
        Forward pass to generate image captions.

        Args:
            image (torch.Tensor): The input image tensor.
            text (Optional[str]): The text prompt for conditional generation.

        Returns:
            str: The generated caption.
        """
        text = [self.text] * image.shape[0]
        if image.max() <= 1:
            image = image * 255.
        inputs = self.processor(image, text, return_tensors="pt").to(image.device)
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
