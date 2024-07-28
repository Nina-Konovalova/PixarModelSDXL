import hydra
from omegaconf import OmegaConf
from utils import *
import os.path
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms



def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to image or image dir')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Path to save images')
    parser.add_argument('--text_model', type=str, default='blip', choices=['blip', 'llava'],
                        help='Text model to process image')
    parser.add_argument('--config', type=str, default='configs/eval.yaml',
                        help='Path to config file')
    parser.add_argument('--strength', type=float, default=0.6,
                        help='Strength for Im2Im pipeline')
    parser.add_argument('--guidance_scale', type=float, default=8, 
                        help='Guidance_scale')
    parser.add_argument('--gradio', type=bool, default=False,
                        help='Run Gradio interface if set to True')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def main():
    """
    Main function to run the image processing pipeline.
    """
    args = parse_args()

    config = OmegaConf.load(args.config)
    model = hydra.utils.instantiate(config.model)

    if not args.gradio:
        accelerator = hydra.utils.instantiate(config.accelerator)

        if os.path.isdir(args.image_path):
            images = get_images_from_directory(args.image_path)
        else:
            assert is_image_file(args.image_path), f"You should use one of the extensions: {['.jpg', '.jpeg', '.png', '.bmp', '.gif']}"
            images = [args.image_path]

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = ImageDataset(images, transform=transform)
        assert args.batch_size == 1, "We can work only with batch size 1 now"
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        model, dataloader = accelerator.prepare(model, dataloader)
        model.set_device(accelerator.device)

        model.eval()
        all_outputs = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['image'].to(accelerator.device)
                prompt = ", pixar-style, pixar art style, highly detailed"
                negative_prompt = 'cropped head, black and white, slanted eyes, deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, ((((mutated hands and fingers)))), watermark, watermarked, oversaturated, censored, distorted hands, amputation, missing hands, obese, doubled face, double hands'
                outputs = model(inputs, prompt=prompt, negative_prompt=negative_prompt, strength=args.strength, guidance_scale=args.guidance_scale)
                all_outputs.append(outputs)
                save_images([outputs], args.output_dir, batch['name'])
        return all_outputs


if __name__ == "__main__":
    main()
