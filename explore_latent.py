import os
import numpy as np
import torch
from PIL import Image
from src.config import settings
from src.app.generator import Generator
from src.app.utils import load_checkpoint, check_if_gpu_available, get_params
from src.modules.image_generate import tensor_to_PIL_image, process_and_resize_image
from src.modules.video_generate import multi_interpolate, generate_latent_vectors


def main(params, path_data):
    check_if_gpu_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    training_params = get_params(
        os.path.join(
            settings.PATH_DATA,
            params['train_version'],
            'training_params.json'
        )
    )

    generator = Generator(training_params["z_dim"], training_params["channels_img"], training_params["features_g"], img_size=training_params['image_size'])
    generator.to(device)

    checkpoint_path = os.path.join(
        settings.PATH_DATA,
        params['train_version'],
        settings.PATH_INSIDE_DATA_WEIGHTS,
        params['checkpoint_file']
    )
    
    print('Checkpoint Path:', checkpoint_path.replace('\\', '/'))
    _, _, _ = load_checkpoint(checkpoint_path, generator, None)

    generator.eval()

    if not os.path.exists(path_data):
        os.makedirs(path_data)

    z_points = generate_latent_vectors(training_params["z_dim"], device, points=[], num_variations=params['num_variations'], step=params['step'])

    generated_images = multi_interpolate(generator, z_points, params['steps_between'])
    print(len(generated_images))

    # Salvando as imagens
    for i, img_tensor in enumerate(generated_images):
        img = tensor_to_PIL_image(img_tensor[0].cpu(), post_processing=params['post_processing'], explore_mode=params['explore_mode'])
        if params['upscale']:
            img_array = np.asarray(img)
            img_array = process_and_resize_image(img_array, new_width=params['upscale'])
            img = Image.fromarray(img_array)
        img_path = os.path.join(path_data, f'generated_image_{i}.jpg')
        img.save(img_path)
        print(f'Image {img_path} saved.')


if __name__ == '__main__':
    params = get_params(settings.PATH_EXPLORE_PARAMS_FILE)
    path_data = os.path.join(settings.PATH_EXPLORE_LATENT_DATA, params['train_version'])

    main(params, path_data)
