import os

import numpy as np
import torch
import torchvision.utils as vutils
import cv2
from PIL import Image
from tqdm import tqdm

from src.app.generator import Generator
from src.app.utils import check_if_gpu_available
from .resize_image import process_and_resize_image


def generate_latent_points(latent_dimension, num_samples, device):
    points = torch.randn(num_samples, latent_dimension, device=device)
    return points


def generate_images(model, latent_dimension, num_samples, device):
    points = generate_latent_points(latent_dimension, num_samples, device)
    points = points.view(num_samples, latent_dimension, 1, 1)
    with torch.no_grad():
        images = model(points)
    return images


def tensor_to_PIL_image(img_tensor, post_processing):
    img_array = img_tensor.clone().detach().cpu().numpy()
    img_array = img_array.transpose(1, 2, 0)
    img_array = (img_array * 255).round().astype(np.uint8)

    # Post processing
    if post_processing:
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)

    return Image.fromarray(img_array)


def main(train_params, images_params, path_data, path_images_generated, upscale_width):

    seed_value = images_params.get("seed", None)

    print('Use seed:', seed_value)

    if seed_value:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

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
        img = tensor_to_PIL_image(individual_img, images_params['post_processing'])
        image_size_str = f"{train_params['image_size']}x{train_params['image_size']}_seed_{images_params['seed']}"
        
        if upscale_width:
            image_size_str = f"{upscale_width}x{upscale_width}"
            img = np.asarray(img)
            img = process_and_resize_image(img, new_width=upscale_width)
            img = Image.fromarray(img)

        img_path = os.path.join(output_directory, f'image_{image_size_str}_{i}.jpg')
        img.save(img_path)

    print("Saving image grid...")
    grid_img = vutils.make_grid(images, nrow=int(num_samples**0.5), padding=2, normalize=True)
    img_grid = tensor_to_PIL_image(grid_img.cpu(), images_params['post_processing'])

    if upscale_width:
        img_grid = np.asarray(img_grid)
        img_grid = process_and_resize_image(img_grid, new_width=upscale_width)
        img_grid = Image.fromarray(img_grid)

    img_grid_path = os.path.join(output_directory, f'grid_{image_size_str}.jpg')
    img_grid.save(img_grid_path)

    print(f"Images saved to {output_directory}")
