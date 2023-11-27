import os
import json
import shutil
import datetime

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms, datasets


def get_params(path_file):
    with open(path_file, 'r', encoding='utf-8') as f:
        params = json.load(f)

    return params


def create_next_version_directory(base_dir, continue_last_training, training_version):
    if training_version:
        return training_version
    
    versions = [d for d in os.listdir(base_dir) if d.startswith('v') and os.path.isdir(os.path.join(base_dir, d))]

    if not versions:
        next_version = 1
    else:
        if continue_last_training:
            return f"v{max(int(v[1:]) for v in versions)}"
        
        next_version = max(int(v[1:]) for v in versions) + 1

    new_dir_base = os.path.join(base_dir, f'v{next_version}')

    for sub_dir in ['', 'samples', 'weights', 'log']:
        os.makedirs(os.path.join(new_dir_base, sub_dir), exist_ok=True)

    return f"v{next_version}"


def print_datetime(label="Current Date and Time"):
    data_hora_atual = datetime.datetime.now()
    data_hora_formatada = data_hora_atual.strftime("%d/%m/%Y %H:%M:%S")
    print(f'\n{label}: {data_hora_formatada}')


def check_if_gpu_available():
    print('Use GPU:', torch.cuda.is_available())

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"GPUs available: {device_count}")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {device_name}")
    else:
        print("No GPU available.")


def check_if_set_seed(seed=None):
    if seed:
        torch.manual_seed(seed)
        print(f'Using the Seed: {seed}')
    else:
        print('Using random seed.')


def dataloader(directory, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(directory, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

def load_checkpoint(path, generator, discriminator=None, optim_g=None, optim_d=None):
    if not os.path.exists(path):
        print("No checkpoint found.")
        return 1, [], []

    checkpoint = torch.load(path)

    generator.load_state_dict(checkpoint['generator_state_dict'])

    if discriminator:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    if optim_g:
        optim_g.load_state_dict(checkpoint['optimizer_g_state_dict'])

    if optim_d:
        optim_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

    epoch = checkpoint['epoch'] + 1
    losses_g = checkpoint['losses_g']
    losses_d = checkpoint['losses_d']

    print(f'Checkpoint loaded, starting from epoch {epoch}')
    return epoch, losses_g, losses_d


def plot_losses(losses_g, losses_d, save_plot_image):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(losses_g, label="G")
    plt.plot(losses_d, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_plot_image, bbox_inches='tight')
    plt.show()


def safe_copy(src, dest_path):
    dest_dir, filename = os.path.split(dest_path)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if os.path.exists(dest_path):
        base_name, ext = os.path.splitext(filename)
        counter = 1

        while os.path.exists(os.path.join(dest_dir, f"{base_name}_{counter}{ext}")):
            counter += 1

        dest_path = os.path.join(dest_dir, f"{base_name}_{counter}{ext}")

    shutil.copy(src, dest_path)
    return dest_path
