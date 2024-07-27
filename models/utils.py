from controlnet_aux import HEDdetector
import numpy as np
import cv2
from PIL import Image
from diffusers import ControlNetModel
from controlnet_aux import OpenposeDetector
import torch


control_dict = {
                'model': {'pose': "thibaud/controlnet-openpose-sdxl-1.0",
                          'canny': "diffusers/controlnet-canny-sdxl-1.0",
                          'scribble': "xinsir/controlnet-scribble-sdxl-1.0"
                },
                'processor': {'pose': "lllyasviel/ControlNet",
                              'canny': None,
                              'scribble': "lllyasviel/Annotators"
                }

}

def nms(x: np.ndarray, t: float, s: float) -> np.ndarray:
    """
    Apply Non-Maximum Suppression (NMS) on the input array.

    Args:
        x (np.ndarray): The input array.
        t (float): The threshold value.
        s (float): The standard deviation for Gaussian blur.

    Returns:
        np.ndarray: The output array after NMS.
    """
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z



def prepare_control(controlnet_type):
    """
    Create controlnet model and processor

    Args:
        controlnet_type (str): pose, canny or scribble type of controlnet

    Returns:
        _type_: _description_
    """
    controlnet = ControlNetModel.from_pretrained(control_dict['model'][controlnet_type], torch_dtype=torch.float16)
    if controlnet_type == 'pose':
        processor = OpenposeDetector.from_pretrained(control_dict['processor'][controlnet_type])
    elif controlnet_type == 'scribble':
        processor = HEDdetector.from_pretrained(control_dict['processor'][controlnet_type]) 
    else:
        processor = None

    return controlnet, processor



def openpose_control(image, processor=None):
    """
    Apply the OpenPose processor to the input image.

    Args:
        image (numpy.ndarray or PIL.Image): The input image.
        processor (callable): The processor function to apply.

    Returns:
        PIL.Image: The processed image with OpenPose applied.
    """
    assert processor is not None, "You should use openpose processor"
    
    openpose_image = processor(image)
    return openpose_image


def canny_control(image, processor=None):
    """
    Apply Canny edge detection to the input image.

    Args:
        image (numpy.ndarray or PIL.Image): The input image.
        processor (callable): The processor function to apply.

    Returns:
        PIL.Image: The image with Canny edge detection applied.
    """
    np_image = np.array(image)
    edges = cv2.Canny(np_image, 50, 140)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    canny_image = Image.fromarray(edges)
    return canny_image


def scribble_control(image, processor=None):
    """
    Apply a scribble effect to the input image using the HED detector.

    Args:
    image (numpy.ndarray or PIL.Image): The input image.
        path (str): Path to the pretrained HED detector model.
        processor (callable): The processor function to apply.

    Returns:
        PIL.Image: The image with the scribble effect applied.
    """
    assert processor is not None, "You should use scribble processor"
    
    scribble_img = processor(image, scribble=False)
    
    # Simulate human sketch draw with additional processing
    scribble_img = np.array(scribble_img)
    scribble_img = nms(scribble_img, 127, 3)
    scribble_img = cv2.GaussianBlur(scribble_img, (0, 0), 3)

    # Apply threshold to generate thinner lines
    threshold = 10 
    scribble_img[scribble_img > threshold] = 255
    scribble_img[scribble_img < 255] = 0
    scribble_img = Image.fromarray(scribble_img)
    return scribble_img

