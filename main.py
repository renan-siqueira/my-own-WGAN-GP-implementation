import json
import torch
import torch.optim as optim
import time

from app.generator import Generator
from app.discriminator import Discriminator
from app.training import train_model
from app.utils import print_datetime, check_if_gpu_available, check_if_set_seed, create_next_version_directory, weights_init, dataloader, load_checkpoint, plot_losses


def main():
    time_start = time.time()
    print_datetime()

    with open('parameters.json', 'r') as f:
        params = json.load(f)

    check_if_gpu_available()
    check_if_set_seed(params["seed"])
    create_next_version_directory(params["directories"]),

    print('Number of repetitions for the discriminator:', params['n_critic'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generator = Generator(params["z_dim"], params["channels_img"], params["features_g"]).to(device)
    generator.apply(weights_init)

    discriminator = Discriminator(params["channels_img"], params["features_d"], params["alpha"]).to(device)
    discriminator.apply(weights_init)

    data_loader = dataloader(params["dataset_dir"], params["image_size"], params["batch_size"])

    optim_g = optim.Adam(generator.parameters(), lr=params["lr_g"], betas=(params['g_beta_min'], params['g_beta_max']))
    optim_d = optim.Adam(discriminator.parameters(), lr=params["lr_d"], betas=(params['d_beta_min'], params['d_beta_max']))

    last_epoch, losses_g, losses_d = load_checkpoint(f'{params["directories"][2]}/checkpoint.pth', generator, discriminator, optim_g, optim_d)

    losses_g, losses_d = train_model(
        generator=generator,
        discriminator=discriminator,
        weights_path=params["directories"][2],
        n_critic=params["n_critic"],
        sample_size=params["sample_size"],
        sample_dir=params["directories"][1],
        optim_g=optim_g,
        optim_d=optim_d,
        data_loader=data_loader,
        device=device,
        z_dim=params["z_dim"],
        num_epochs=params["num_epochs"],
        last_epoch=last_epoch,
        save_model_at=params['save_model_at'],
        log_dir =params['directories'][3],
        losses_g=losses_g,
        losses_d=losses_d,
    )

    time_end = time.time()
    time_total = (time_end - time_start) / 60

    print(f"The code took {round(time_total, 1)} minutes to execute.")
    print_datetime()

    plot_losses(losses_g, losses_d, save_plot_image=params["directories"][0])


if __name__ == '__main__':
    main()
