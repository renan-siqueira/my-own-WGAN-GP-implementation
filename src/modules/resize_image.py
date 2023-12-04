from PIL import Image, ImageFilter, ImageEnhance
import numpy as np


def gamma_correction(image, gamma):
    image = np.array(image)
    image = 255 * (image / 255) ** (1 / gamma)
    return Image.fromarray(np.uint8(image))


def process_and_resize_image(image_np, new_width=256, apply_filter=None, gamma=1.1):
    original_image = Image.fromarray(image_np)
    width, height = original_image.size

    new_height = int(new_width * height / width)

    resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)

    if apply_filter:
        resized_image = resized_image.filter(ImageFilter.SHARPEN)

        resized_image = ImageEnhance.Contrast(resized_image)
        resized_image = resized_image.enhance(1.25)

        resized_image = gamma_correction(resized_image, gamma)
        
    final_image_np = np.array(resized_image)

    return final_image_np