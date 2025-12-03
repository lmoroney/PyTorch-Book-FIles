import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login

login(token="PUT TOKEN HERE")

# Set your seed value
seed = 123456  # You can use any integer value you want

# Create a generator with the seed
generator = torch.Generator("cpu").manual_seed(seed)

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
image = pipe(
    "A photo of a group of teddy bears eating pizza on the surface of mars",
    num_inference_steps=40,
    generator=generator  # Add the generator here
).images[0]
image.save("teddies.png")
