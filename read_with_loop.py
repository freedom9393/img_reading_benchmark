import os
import gc
import cv2

from utils import time_decorator

extensions = ('.jpg', '.jpeg', '.png')


@time_decorator
def read_images_iteratively(directory, **kwargs):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(extensions):
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:
                images.append(img)
    gc.collect()
    return len(images)
