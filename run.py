import json
import argparse
import os

from src.config import settings
from src.modules import run_training, image_generate, video_generate


def get_params(path_file):
    with open(path_file, 'r') as f:
        params = json.load(f)
    
    return params


def main(args):    

    if args.training:
        training_params = get_params(settings.PATH_TRAIN_PARAMS)
        run_training.main(training_params, settings.PATH_DATA, settings.PATH_DATASET, settings.PATH_TRAIN_PARAMS)

    if args.image:
        image_params = get_params(settings.PATH_IMAGE_PARAMS)
        training_params = get_params(os.path.join(settings.PATH_DATA, image_params['train_version'], os.path.basename(settings.PATH_TRAIN_PARAMS)))

        image_generate.main(training_params, image_params, settings.PATH_DATA, settings.PATH_IMAGES_GENERATED, upscale_width=args.upscale)

    if args.video:
        video_params = get_params(settings.PATH_VIDEO_PARAMS)
        training_params = get_params(os.path.join(settings.PATH_DATA, video_params['train_version'], os.path.basename(settings.PATH_TRAIN_PARAMS)))

        video_generate.main(training_params, video_params, settings.PATH_DATA, settings.PATH_VIDEOS_GENERATED, upscale_width=args.upscale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train the generator, generate images and/or videos from a trained generator")

    parser.add_argument('--training', action='store_true', help='If true, executes the training')
    parser.add_argument('--image', action='store_true', help='If true, generates images')
    parser.add_argument('--video', action='store_true', help='If true, generates videos')
    parser.add_argument('--upscale', type=int, default=None, help='Sets the upscale width. Can be None or an integer value.')

    args = parser.parse_args()

    main(args)
