import os
import time
import threading
import random

import torch
import numpy as np
import cv2

from src.app import utils
from src.config import settings
from src.modules import video_generate, resize_image
from src.app.generator import Generator


# GLOBAL
params = {}
training_params = {}
device = None
generator = None
wait_generated_images = None
upscale = None
post_processing = None
window_size = (256, 256)

def get_points_for_command(command):
    global params

    try:
        commands_to_points = utils.get_params(
            os.path.join(
                settings.PATH_INTERATIVE_DATA, params['train_version'],
                settings.COMMANDS_FILENAME
            )
        )
    except FileNotFoundError as e:
        print('No such file or directory', e)
        commands_to_points = {}

    if command == 'random' or 'esperar' not in commands_to_points:
        return generate_random_points()

    return commands_to_points.get(command, commands_to_points["esperar"])


def generate_random_points():
    global params, training_params
    num_points = random.randint(3, 8)
    random_points = [random.randint(0, training_params['z_dim'] - 1) for _ in range(num_points)]
    print('Random:', random_points)
    return random_points


def play_video(generated_images, stop_event):
    global params, window_size

    while True:
        try:
            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Video', window_size[0], window_size[1])
            break
        except:
            time.sleep(1)

    while not stop_event.is_set():
        for image in generated_images:
            if stop_event.is_set():
                break

            image_np = image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            image_np = (image_np + 1) / 2
            frame = (image_np * 255).astype(np.uint8)

            if params['upscale']:
                frame = resize_image.process_and_resize_image(frame, new_width=window_size[0])

            # Post processing
            if params['post_processing']:
                frame = cv2.GaussianBlur(frame, (5, 5), 0)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame, window_size)

            # Verifique se a janela ainda existe antes de exibir o frame
            if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) >= 1:
                cv2.imshow('Video', frame_resized)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    stop_event.set()
                    break
            else:
                stop_event.set()
                break

    cv2.destroyAllWindows()


def configure():
    global params, training_params, device, generator, wait_generated_images

    print('Start project configurations...')

    params = utils.get_params(settings.PATH_INTERATIVE_PARAMS_FILE)
    print('--Parameters configured!')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('--Device configured:', device)

    checkpoint_path = f'{settings.PATH_DATA}/{params["train_version"]}/weights/{params["checkpoint_file"]}'
    checkpoint = torch.load(checkpoint_path)
    print('--Checkpoint loaded:', checkpoint_path)

    training_params = utils.get_params(f'{settings.PATH_DATA}/{params["train_version"]}/training_params.json')
    generator = Generator(training_params["z_dim"], training_params["channels_img"], training_params["features_g"], img_size=training_params['image_size']).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    print('--Generator initialized!')

    print('...Loading images to command "esperar"')
    wait_points = get_points_for_command("esperar")
    wait_z_points = video_generate.generate_latent_vectors(training_params["z_dim"], device, wait_points)
    wait_generated_images = video_generate.multi_interpolate(generator, wait_z_points, params['steps_between'])
    print('Done!\n')


def main():
    configure()

    stop_event = threading.Event()

    while True:
        stop_event.clear()
        video_thread = threading.Thread(target=play_video, args=(wait_generated_images, stop_event))
        video_thread.start()

        command = input("Digite um comando: ")

        if command and command != "esperar":
            stop_event.set()
            video_thread.join()

            points = get_points_for_command(command)
            z_points = video_generate.generate_latent_vectors(training_params["z_dim"], device, points)
            generated_images = video_generate.multi_interpolate(generator, z_points, params['steps_between'])

            command_stop_event = threading.Event()
            threading.Thread(target=play_video, args=(generated_images, command_stop_event)).start()
            time.sleep(3)
            command_stop_event.set()


if __name__ == "__main__":
    main()
