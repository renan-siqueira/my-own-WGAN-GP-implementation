import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms, datasets
import datetime


def print_datetime(label="Current Date and Time"):
    data_hora_atual = datetime.datetime.now()
    data_hora_formatada = data_hora_atual.strftime("%d/%m/%Y %H:%M:%S")
    print(label + ":", data_hora_formatada)


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
        print(f'Using random seed.')


def create_dirs(directories):
    for directory in directories:
            os.makedirs(directory, exist_ok=True)


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
        

def load_checkpoint(path, generator, discriminator, optim_g, optim_d):
    if not os.path.exists(path):
        print("No checkpoint found.")
        return 1, [], []

    checkpoint = torch.load(path)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optim_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optim_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    
    epoch = checkpoint['epoch']
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
