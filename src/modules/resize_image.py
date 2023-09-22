from PIL import Image, ImageFilter, ImageEnhance
import numpy as np


def gamma_correction(image, gamma):
    image = np.array(image)
    image = 255 * (image / 255) ** (1 / gamma)
    return Image.fromarray(np.uint8(image))


def process_and_resize_image(image_np, new_width=256, gamma=1.1):
    original_image = Image.fromarray(image_np)
    width, height = original_image.size

    new_height = int(new_width * height / width)

    resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
    resized_image = resized_image.filter(ImageFilter.SHARPEN)

    enhancer = ImageEnhance.Contrast(resized_image)
    enhanced_image = enhancer.enhance(1.25)

    gamma_corrected_image = gamma_correction(enhanced_image, gamma)
    final_image_np = np.array(gamma_corrected_image)

    return final_image_np