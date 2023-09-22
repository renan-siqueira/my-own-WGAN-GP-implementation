import os
import json
import numpy as np

import torch
import torchvision.utils as vutils

from PIL import Image
from tqdm import tqdm

from src.app.generator import Generator
from src.app.utils import check_if_gpu_available


def generate_latent_points(latent_dimension, num_samples, device):
    points = torch.randn(num_samples, latent_dimension, device=device)
    return points


def generate_images(model, latent_dimension, num_samples, device):
    points = generate_latent_points(latent_dimension, num_samples, device)
    points = points.view(num_samples, latent_dimension, 1, 1)
    with torch.no_grad():
        images = model(points)
    return images


def tensor_to_PIL_image(img_tensor):
    img_array = img_tensor.clone().detach().cpu().numpy()
    img_array = img_array.transpose(1, 2, 0)
    img_array = (img_array * 255).round().astype(np.uint8)
    return Image.fromarray(img_array)


def main(path_data, path_train_params, path_images_params, path_images_generated):
    
    with open(path_train_params, 'r') as f:
        train_params = json.load(f)

    with open(path_images_params, 'r') as f:
        images_params = json.load(f)
    
    num_samples = images_params['num_samples']
    output_directory = os.path.join(path_images_generated, images_params["train_version"])

    check_if_gpu_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_path = f'{path_data}/{images_params["train_version"]}/weights/checkpoint.pth'

    checkpoint = torch.load(checkpoint_path)

    generator = Generator(train_params["z_dim"], train_params["channels_img"], train_params["features_g"], img_size=train_params['image_size']).to(device)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    latent_dimension = train_params['z_dim']

    print("Generating images...")
    images = generate_images(generator, latent_dimension, num_samples, device)
    images = (images + 1) / 2.0

    os.makedirs(output_directory, exist_ok=True)

    print("Saving individual images...")
    for i in tqdm(range(num_samples)):
        individual_img = images[i].cpu().clamp(0, 1)
        img = tensor_to_PIL_image(individual_img)
        img_path = os.path.join(output_directory, f'image_{i}.jpg')
        img.save(img_path)

    print("Saving image grid...")
    grid_img = vutils.make_grid(images, nrow=int(num_samples**0.5), padding=2, normalize=True)
    img_grid = tensor_to_PIL_image(grid_img.cpu())
    img_grid_path = os.path.join(output_directory, 'grid.jpg')
    img_grid.save(img_grid_path)

    print(f"Images saved to {output_directory}")