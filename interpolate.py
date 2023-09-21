import json
import torch
import numpy as np
import cv2
from tqdm import tqdm

from app.generator import Generator
from app.utils import check_if_gpu_available


def interpolate(z1, z2, alpha):
    return alpha * z1 + (1 - alpha) * z2


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


def main(train_version, interpolate_points, steps_between, fps, video_name):
    with open('parameters.json', 'r') as f:
        params = json.load(f)

    check_if_gpu_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_path = f'data/{train_version}/weights/checkpoint.pth'
    checkpoint = torch.load(checkpoint_path)

    generator = Generator(params["z_dim"], params["channels_img"], params["features_g"], img_size=params['image_size']).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    z_points = [torch.randn(1, params["z_dim"]).to(device) for _ in range(interpolate_points)]

    print("Generating interpolated images...")
    generated_images = multi_interpolate(generator, z_points, steps_between)

    frame_size = (params["image_size"], params["image_size"])
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, frame_size)

    print("Writing images to video...")
    for image in tqdm(generated_images):
        image_np = image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        image_np = (image_np + 1) / 2
        frame = (image_np * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()


if __name__ == '__main__':

    train_version = 'v8'

    interpolate_points = 10
    steps_between = 30
    fps = 30

    video_name = 'video.avi'

    main(train_version, interpolate_points, steps_between, fps, video_name)