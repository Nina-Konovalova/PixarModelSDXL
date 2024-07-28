import gradio as gr
from PIL import Image
import hydra
from omegaconf import OmegaConf
import numpy as np
import torch


def process_image(image, control_type, strength, guidance_scale, prompt, negative_prompt, seed):
    # Placeholder for the actual image processing pipeline
    # Replace this part with the actual model inference logic
    image = torch.from_numpy(np.array(image) / 255.).permute(2, 0, 1)[None]
    
    prompt = prompt + ", pixar-style, pixar art style, highly detailed"
    negative_prompt = negative_prompt + 'cropped head, black and white, slanted eyes, deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, ((((mutated hands and fingers)))), watermark, watermarked, oversaturated, censored, distorted hands, amputation, missing hands, obese, doubled face, double hands'
    outputs = model(image, prompt=prompt, negative_prompt=negative_prompt, strength=strength, guidance_scale=guidance_scale)

    return outputs

css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 15rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 15rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
            margin-top: 10px;
            margin-left: auto;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;right:0;
        }
        #share-btn * {
            all: unset;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
        
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #prompt-text-input, #negative-prompt-text-input{padding: .45rem 0.625rem}
        #component-16{border-top-width: 1px!important;margin-top: 1em}
        .image_duplication{position: absolute; width: 100px; left: 50px}
        img {
            max-height: 400px;
            width: 100%; /* Add this line */
            height: 100%; /* Add this line */
            object-fit: contain;
            border: none; /* Remove the border */
        }
"""

examples = [
    ["images/nina.jpg", "no", 0.6, 8.0, "", ""],
    ["images/nina_2.jpg", "canny", 0.6, 8.0, "", ""],
    ["images/photo_2024-07-27_14-16-25.jpg", "scribble", 0.6, 8.0, "", ""],
    ["images/photo_2024-07-27_14-16-26.jpg", "pose", 0.6, 8.0, "", ""]
]


with gr.Blocks(css=css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Pixar portrait SDXL</h1>")
    gr.Markdown("This model can transform your portrait image into a pixar-like portrait. It is based on [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [LoRA weights](https://huggingface.co/ntc-ai/SDXL-LoRA-slider.pixar-style).")
    config = OmegaConf.load("configs/eval.yaml")
    model = hydra.utils.instantiate(config.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.set_device(device)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload your image")
            control_type = gr.Radio(["canny", "scribble", "pose", "no"], label="Type of Control", value="no")
            run_button = gr.Button("Run")
            
            strength = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.6, label="Strength")
            guidance_scale = gr.Slider(minimum=0, maximum=20, step=0.5, value=8, label="Guidance Scale")
            prompt = gr.Textbox(value="", label="Additional Prompt")
            negative_prompt = gr.Textbox(value="", label="Additional Negative Prompt")
            # seed = gr.Number(value=42, label="Seed", precision=0)
            
        with gr.Column():
            output_image = gr.Image(type="pil", label="Output Image")

    gr.Markdown("### Example Images")
    examples_component = gr.Examples(examples=examples, inputs=[input_image, control_type, strength, guidance_scale], outputs=output_image)

    run_button.click(fn=process_image, inputs=[input_image, control_type, strength, guidance_scale, prompt, negative_prompt], outputs=[output_image])

demo.launch()