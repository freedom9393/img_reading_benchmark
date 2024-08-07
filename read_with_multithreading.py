import os
import cv2
import concurrent.futures


max_workers = 4  # number of threads


def read_image(filename):
    return cv2.imread(filename)

def read_images_multithreaded(directory):
    images = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(read_image, os.path.join(directory, filename)) for filename in os.listdir(directory) if filename.endswith(".jpg") or filename.endswith(".png")]
        for future in concurrent.futures.as_completed(futures):
            img = future.result()
            if img is not None:
                images.append(img)
    return images

images = read_images_multithreaded('path/to/images')
