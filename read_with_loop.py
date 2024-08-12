import os
import cv2

def read_images_iteratively(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:
                images.append(img)
    return images

images = read_images_iteratively('path/to/images')
