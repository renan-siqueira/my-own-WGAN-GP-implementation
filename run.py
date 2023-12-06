import argparse
import os

from src.app import utils
from src.config import settings
from src.modules import run_training, image_generate, video_generate


def main(args):

    if args.training:
        if args.version:
            training_params = utils.get_params(os.path.join(settings.PATH_DATA, args.version, settings.JSON_TRAIN_PARAMS_FILENAME))
            print(f'--Use {args.version} training params')
        else:
            training_params = utils.get_params(settings.PATH_TRAIN_PARAMS)
            print('--Use main training params')

        training_params['train_version'] = args.version if args.version else training_params['train_version']
        path_dataset = os.path.join(settings.PATH_DATASET, training_params['path_dataset'])

        run_training.main(training_params, settings.PATH_DATA, path_dataset, settings.PATH_TRAIN_PARAMS)

    if args.image:
        image_params = utils.get_params(settings.PATH_IMAGE_PARAMS)
    
        image_params['train_version'] = args.version if args.version else image_params['train_version']
        image_params['checkpoint'] = args.checkpoint if args.checkpoint else image_params['checkpoint']

        training_params = utils.get_params(os.path.join(settings.PATH_DATA, image_params['train_version'], os.path.basename(settings.PATH_TRAIN_PARAMS)))

        image_generate.main(training_params, image_params, settings.PATH_DATA, settings.PATH_IMAGES_GENERATED, upscale_width=args.upscale)

    if args.video:
        video_params = utils.get_params(settings.PATH_VIDEO_PARAMS)

        video_params['train_version'] = args.version if args.version else video_params['train_version']
        video_params['checkpoint'] = args.checkpoint if args.checkpoint else video_params['checkpoint']

        training_params = utils.get_params(os.path.join(settings.PATH_DATA, video_params['train_version'], os.path.basename(settings.PATH_TRAIN_PARAMS)))

        video_generate.main(training_params, video_params, settings.PATH_DATA, settings.PATH_VIDEOS_GENERATED, upscale_width=args.upscale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train the generator, validate the traning, generate images and/or videos from a trained generator")

    parser.add_argument('--training', action='store_true', help='If true, executes the training')
    parser.add_argument('--image', action='store_true', help='If true, generates images')
    parser.add_argument('--video', action='store_true', help='If true, generates videos')

    parser.add_argument('--upscale', type=int, default=None, help='Sets the upscale width. Can be None or an integer value.')
    parser.add_argument('--version', type=str, default=None, help='Sets the version of training. Can be None or string value.')
    parser.add_argument('--checkpoint-epoch', type=str, default=None, help='Sets the checkpoint epoch file of training. Can be None or string value.')
    
    args = parser.parse_args()

    main(args)
