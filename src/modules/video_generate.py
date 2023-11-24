import os
import torch
import numpy as np
import cv2
from tqdm import tqdm

from src.app.generator import Generator
from src.app.utils import check_if_gpu_available
from .resize_image import process_and_resize_image


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos(torch.clamp(torch.matmul(low_norm, high_norm.t()), -1, 1))
    so = torch.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high
    interpolation = (torch.sin((1.0 - val) * omega) / so * low) + (torch.sin(val * omega) / so * high)
    return interpolation


def interpolate(z1, z2, alpha):
    return slerp(alpha, z1, z2)


def multi_interpolate(generator, z_list, steps_between):
    generated_images = []
    for i in range(len(z_list) - 1):
        z1 = z_list[i]
        z2 = z_list[i + 1]
        alphas = np.linspace(0, 1, steps_between)
        for alpha in alphas:
            z_interp = interpolate(z1, z2, alpha)
            z_interp = z_interp.view(z_interp.size(0), z_interp.size(1), 1, 1)
            with torch.no_grad():
                generated_image = generator(z_interp)
            generated_images.append(generated_image)
    return generated_images


def main(train_params, video_params, path_data, path_videos_generated, upscale_width):

    seed_value = video_params.get("seed", None)
    print('Use seed:', seed_value)

    if seed_value:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

    output_directory = os.path.join(path_videos_generated, video_params['train_version'])

    check_if_gpu_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_path = f'{path_data}/{video_params["train_version"]}/weights/checkpoint.pth'
    checkpoint = torch.load(checkpoint_path)

    generator = Generator(train_params["z_dim"], train_params["channels_img"], train_params["features_g"], img_size=train_params['image_size']).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    z_points = [torch.randn(1, train_params["z_dim"]).to(device) for _ in range(video_params['interpolate_points'])]

    print("Generating interpolated images...")
    generated_images = multi_interpolate(generator, z_points, video_params['steps_between'])

    os.makedirs(output_directory, exist_ok=True)

    if upscale_width:
        frame_size = (upscale_width, upscale_width)
    else:
        frame_size = (train_params["image_size"], train_params["image_size"])

    out = cv2.VideoWriter(
        os.path.join(
            output_directory,
            f'video_{frame_size[0]}x{frame_size[1]}_{video_params["fps"]}fps_{video_params["steps_between"]}steps_between_{video_params["interpolate_points"]}_interpolate_points_seed_{video_params["seed"]}.mp4'
        ),
        cv2.VideoWriter_fourcc(*'mp4v'),
        video_params["fps"],
        frame_size
    )

    print("Writing images to video...")
    for image in tqdm(generated_images):
        image_np = image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        image_np = (image_np + 1) / 2
        frame = (image_np * 255).astype(np.uint8)
        
        if upscale_width:
            frame = process_and_resize_image(frame, new_width=upscale_width)
        
        # Post processing
        if video_params['post_processing']:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
