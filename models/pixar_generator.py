
from typing import Optional, Union, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
import torch
from rembg import remove
from models.utils import *
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, AutoPipelineForImage2Image
import torchvision




class PixarGenerator(torch.nn.Module):
    def __init__(self, controlnet_type: str,
                 pixar_weights: str = 'https://huggingface.co/martyn/sdxl-turbo-mario-merge-top-rated/blob/main/topRatedTurboxlLCM_v10.safetensors',
                 lora_weights: str = 'ntc-ai/SDXL-LoRA-slider.pixar-style',
                 adapter_weights: float = 2.0,
                 description_model: Optional[torch.nn.Module] = None,
                 seed: int = None,
                 scheduler: str = 'ddim'):
        """
        Initialize the PixarGenerator.

        Args:
            controlnet_type (str): Type of controlnet ('pose', 'canny', 'scribble').
            pixar_weights (str): URL or path to the pixar weights.
            lora_weights (str): Path to the LoRA weights.
            adapter_weights (float): Weight of the adapter.
            description_model (Optional[torch.nn.Module]): Description model for additional processing.
            seed (int): Random seed for reproducibility.
        """
        super().__init__()
        
        if controlnet_type != 'no':
            controlnet, self.processor = prepare_control(controlnet_type)
        else:
            controlnet, self.processor = None, None
        
        self.controlnet_type = controlnet_type

        # Load the pipeline with pixar weights
        pipe = StableDiffusionXLPipeline.from_single_file(pixar_weights)

        # Load and activate the LoRA
        pipe.load_lora_weights(lora_weights, weight_name='pixar-style.safetensors', adapter_name="pixar-style")
        pipe.set_adapters(["pixar-style"], adapter_weights=[adapter_weights])

        # Set controlnet
        if controlnet is not None:
            pipe.controlnet = controlnet

        self.pipeline_im2im = AutoPipelineForImage2Image.from_pipe(pipe)

        if scheduler == 'euler':
            self.pipeline_im2im.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline_im2im.scheduler.config)



        self.description_model = description_model

        # Set the generator based on the seed
        self.generator = torch.Generator(seed) if seed is not None else None

    def set_device(self, device):
        self.pipeline_im2im.to(device)
        #self.generator.to(device)
        if self.description_model is not None:
            self.description_model.to(device)
        if device == torch.device('cuda'):
            try:
                import xformers
                self.pipeline_im2im.enable_xformers_memory_efficient_attention()
            except ImportError:
                pass  # Run without xformers if not available




    @torch.no_grad()
    def create_control(self, images: Union[np.ndarray, Image.Image], no_background: bool = True) -> Optional[Image.Image]:
        """
        Create a control image based on the input image and controlnet type.

        Args:
            image (Union[np.ndarray, Image.Image]): The input image.
            no_background (bool): Whether to remove the background from the image.

        Returns:
            Optional[Image.Image]: The control image.
        """
        if no_background:
            for image in images:
                image = torchvision.transforms.functional.to_pil_image(image)
                image = remove(image)
                image = torch.from_numpy(np.array(image))
        
        if self.controlnet_type == 'pose':
            control_image = openpose_control(image, self.processor)
        elif self.controlnet_type == 'canny':
            control_image = canny_control(image)
        elif self.controlnet_type == 'scribble':
            control_image = scribble_control(image, self.processor)
        else:
            control_image = None
        
        return control_image

    def forward(self, image: Union[np.ndarray, Image.Image], prompt: str=None, 
                control_image: Optional[Image.Image] = None, 
                negative_prompt: str = '',
                strength: float = 0.6,
                num_inference_steps: int = 25,
                guidance_scale: float = 8) -> Image.Image:
        """
        Generate an image based on the input image, prompt, and control image.

        Args:
            image (Union[np.ndarray, Image.Image]): The input image.
            prompt (str): The text prompt for the generation.
            control_image (Optional[Image.Image]): The control image.
            negative_prompt (str): The negative text prompt for the generation.
            strength (float): The strength of the transformation.
            num_inference_steps (int): The number of inference steps.
            guidance_scale (float): The guidance scale for generation.
        Returns:
            PIL.Image: The generated image.
        """
        if control_image is None:
            control_image = self.create_control(image)
        
        prompt = self.description_model(image, prompt) + prompt

        generated_image = self.pipeline_im2im(prompt, 
                                              negative_prompt=negative_prompt, 
                                              control_image=control_image, 
                                              image=image, 
                                              strength=strength, 
                                              num_inference_steps=num_inference_steps, 
                                              guidance_scale=guidance_scale,
                                              generator=self.generator).images[0]
        return generated_image




        

        


