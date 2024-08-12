import os
import gc
import cv2
import concurrent.futures

from utils import time_decorator


def read_image(filepath):
    return filepath, cv2.imread(filepath)


@time_decorator
def read_images_multithreading(directory, **kwargs):
    images = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=kwargs['max_workers']) as executor:
        futures = [executor.submit(read_image, os.path.join(directory, filename)) for filename in os.listdir(directory) if filename.endswith(".jpg") or filename.endswith(".png")]
        for future in concurrent.futures.as_completed(futures):
            img_path, img = future.result()
            if img is not None:
                images[img_path] = img
    gc.collect()
    return images
