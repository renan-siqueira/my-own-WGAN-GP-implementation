import cv2
import os
import json
from tqdm import tqdm
from src.config import settings

def create_video_from_images(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    
    folder_path = config['input_folder']
    output_file = config['output_file']
    frame_rate = float(config.get('frame_rate', 30.0))

    images = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".jpg") or file.endswith(".png"):
            images.append(os.path.join(folder_path, file))

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    for image in tqdm(images, desc="Creating Video"):
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    create_video_from_images(settings.PATH_CREATE_VIDEO_FROM_FOLDER_FILE)