import time
import os

import torch
import torch.optim as optim
from torchvision import models

from src.app.generator import Generator
from src.app.discriminator import Discriminator
from src.app.training import train_model
from src.app.utils import print_datetime, check_if_gpu_available, check_if_set_seed, create_next_version_directory, weights_init, dataloader, load_checkpoint, plot_losses, safe_copy


def main(params, path_data, path_dataset, path_train_params):
    time_start = time.time()
    print_datetime()

    check_if_gpu_available()
    check_if_set_seed(params["seed"])

    print('Number of repetitions for the discriminator:', params['n_critic'])
    print(f'Image size: {params["image_size"]}x{params["image_size"]}\n')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Frechet Inception Distance (FID)
    inception_model = models.inception_v3(weights='Inception_V3_Weights.DEFAULT', transform_input=False, init_weights=False).to(device)
    inception_model = inception_model.eval()

    generator = Generator(params["z_dim"], params["channels_img"], params["features_g"], img_size=params['image_size']).to(device)
    generator.apply(weights_init)

    discriminator = Discriminator(params["channels_img"], params["features_d"], params["alpha"], img_size=params['image_size']).to(device)
    discriminator.apply(weights_init)

    data_loader = dataloader(path_dataset, params["image_size"], params["batch_size"])

    optim_g = optim.Adam(generator.parameters(), lr=params["lr_g"], betas=(params['g_beta_min'], params['g_beta_max']))
    optim_d = optim.Adam(discriminator.parameters(), lr=params["lr_d"], betas=(params['d_beta_min'], params['d_beta_max']))

    training_version = create_next_version_directory(path_data, params['continue_last_training'])

    data_dir = os.path.join(path_data, training_version)
    print('Training version:', training_version)

    # Create a copy of parameters in training version folder
    safe_copy(path_train_params, os.path.join(data_dir, path_train_params.split('/')[-1]))

    last_epoch, losses_g, losses_d = load_checkpoint(os.path.join(data_dir, 'weights', 'checkpoint.pth'), generator, discriminator, optim_g, optim_d)

    losses_g, losses_d = train_model(
        inception_model=inception_model,
        generator=generator,
        discriminator=discriminator,
        weights_path= os.path.join(data_dir, 'weights'),
        n_critic=params["n_critic"],
        sample_size=params["sample_size"],
        sample_dir= os.path.join(data_dir, 'samples'),
        optim_g=optim_g,
        optim_d=optim_d,
        data_loader=data_loader,
        device=device,
        z_dim=params["z_dim"],
        lambda_gp=params["lambda_gp"],
        num_epochs=params["num_epochs"],
        last_epoch=last_epoch,
        save_model_at=params['save_model_at'],
        log_dir = os.path.join(data_dir, 'log'),
        losses_g=losses_g,
        losses_d=losses_d,
    )

    time_end = time.time()
    time_total = (time_end - time_start) / 60

    print(f"The code took {round(time_total, 1)} minutes to execute.")
    print_datetime()

    plot_losses(losses_g, losses_d, save_plot_image=os.path.join(data_dir, f"{training_version}.jpg"))
