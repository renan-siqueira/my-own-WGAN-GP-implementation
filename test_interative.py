import os
import json
import cv2
import numpy as np
import torch

from src.app import utils
from src.config import settings
from src.modules import video_generate, resize_image
from src.app.generator import Generator


# GLOBAL
params = {}
training_params = {}
device = None
generator = None
upscale = None
post_processing = None
window_size = (320, 320)


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

    return commands_to_points.get(command, [0])


def play_video(generated_images):
    global params, window_size

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', window_size[0], window_size[1])

    for image in generated_images:
        image_np = image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        image_np = (image_np + 1) / 2
        frame = (image_np * 255).astype(np.uint8)

        if params['upscale']:
            frame = resize_image.process_and_resize_image(frame, new_width=window_size[0])

        if params['post_processing']:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame, window_size)

        cv2.imshow('Video', frame_resized)
        cv2.waitKey(25)


def save_command(command_name, points):
    data_interative_path_version = os.path.join(settings.PATH_INTERATIVE_DATA, params['train_version'])
    os.makedirs(data_interative_path_version, exist_ok=True)

    commands_file = os.path.join(settings.PATH_INTERATIVE_DATA, params['train_version'], settings.COMMANDS_FILENAME)


    if not os.path.isfile(commands_file) or os.path.getsize(commands_file) == 0:
        commands = {}
    else:
        with open(commands_file, 'r') as file:
            try:
                commands = json.load(file)
            except json.JSONDecodeError:
                commands = {}

    commands[command_name] = points

    with open(commands_file, 'w') as file:
        json.dump(commands, file, indent=4)


def delete_command(command_name):
    commands_file = os.path.join(settings.PATH_INTERATIVE_DATA, params['train_version'], settings.COMMANDS_FILENAME)

    if not os.path.isfile(commands_file) or os.path.getsize(commands_file) == 0:
        print("Arquivo de comandos vazio ou não encontrado.")
        return

    with open(commands_file, 'r') as file:
        try:
            commands = json.load(file)
        except json.JSONDecodeError:
            print("Erro na leitura do arquivo JSON.")
            return

    if command_name in commands:
        del commands[command_name]
        with open(commands_file, 'w') as file:
            json.dump(commands, file, indent=4)
        print(f"Comando '{command_name}' removido com sucesso.")
    else:
        print(f"Comando '{command_name}' não encontrado.")


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

    print('Done!\n')


def main():
    global params, training_params, device, generator

    configure()

    initial_points = [0, 0]
    z_points = video_generate.generate_latent_vectors(training_params["z_dim"], device, initial_points)
    generated_images = video_generate.multi_interpolate(generator, z_points, params['steps_between'])
    play_video(generated_images)

    while True:
        command_input = input("Digite um comando (números separados por vírgulas, um nome de comando salvo ou 'exit' para sair): ")
        if command_input == 'exit':
            break

        # Verifica se o comando é numérico ou um comando salvo
        if command_input.replace(',', '').isnumeric():
            points = list(map(int, command_input.split(',')))
            save_or_delete = "salvar"
        else:
            points = get_points_for_command(command_input)
            save_or_delete = "deletar" if points else None

        z_points = video_generate.generate_latent_vectors(training_params["z_dim"], device, points)
        generated_images = video_generate.multi_interpolate(generator, z_points, params['steps_between'])
        play_video(generated_images)
        
        if save_or_delete:
            response = input(f"Deseja {save_or_delete} esta ação? (y/n): ").lower()
            if response == 'y':
                if save_or_delete == "salvar":
                    action_name = input("Digite o nome da ação: ")
                    save_command(action_name, points)
                elif save_or_delete == "deletar":
                    delete_command(command_input)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
