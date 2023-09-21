import os
import json
import numpy as np

import torch
import torchvision.utils as vutils

from PIL import Image

from app.generator import Generator
from app.utils import check_if_gpu_available


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
    img_array = img_tensor.clone().detach().cpu().numpy()  # Clonar e detach para evitar modificações no tensor original
    img_array = img_array.transpose(1, 2, 0)
    img_array = (img_array * 255).round().astype(np.uint8)  # Multiplicar por 255 e arredondar
    return Image.fromarray(img_array)


with open('parameters.json', 'r') as f:
    params = json.load(f)


check_if_gpu_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_version = 'v4'
checkpoint_path = f'data/{train_version}/weights/checkpoint.pth'

checkpoint = torch.load(checkpoint_path)

generator = Generator(params["z_dim"], params["channels_img"], params["features_g"], img_size=params['image_size']).to(device)

# generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

num_samples = 4
latent_dimension = params['z_dim']

images = generate_images(generator, latent_dimension, num_samples, device)
images = (images + 1) / 2.0

output_directory = 'generated_images/'
os.makedirs(output_directory, exist_ok=True)

# Salvar cada imagem individualmente
for i in range(num_samples):
    individual_img = images[i].cpu().clamp(0, 1)  # Garantir que esteja no intervalo [0, 1]
    img = tensor_to_PIL_image(individual_img)
    img_path = os.path.join(output_directory, f'image_{i}.jpg')
    img.save(img_path)

# Criar e salvar o grid com todas as imagens
grid_img = vutils.make_grid(images, nrow=int(num_samples**0.5), padding=2, normalize=True)
img_grid = tensor_to_PIL_image(grid_img.cpu())
img_grid_path = os.path.join(output_directory, 'grid.jpg')
img_grid.save(img_grid_path)

print(f"Images saved to {output_directory}")
