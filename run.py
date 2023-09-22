import json

from src.config.settings import PATH_MAIN_PARAMS, PATH_DATA, PATH_DATASET, PATH_IMAGES_GENERATED, PATH_VIDEOS_GENERATED, PATH_ORIGINAL_FILES, PATH_TRAIN_PARAMS, PATH_IMAGE_PARAMS, PATH_VIDEO_PARAMS
from src.modules import copy_files_to_dataset, run_training, image_generate, video_generate


def main():
    with open(PATH_MAIN_PARAMS, 'r') as f:
        main_params = json.load(f)


    if main_params['copy_files_to_dataset']:
        copy_files_to_dataset.main(PATH_ORIGINAL_FILES, PATH_DATASET, interval=1)

    if main_params['run_training']:
        run_training.main(PATH_DATA, PATH_DATASET, PATH_TRAIN_PARAMS)

    if main_params['generate_image']:
        image_generate.main(PATH_DATA, PATH_TRAIN_PARAMS, PATH_IMAGE_PARAMS, PATH_IMAGES_GENERATED)

    if main_params['generate_video']:
        video_generate.main(PATH_DATA, PATH_TRAIN_PARAMS, PATH_VIDEO_PARAMS, PATH_VIDEOS_GENERATED, upscale_width=None)


if __name__ == '__main__':
    main()
