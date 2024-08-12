import os
import gc
import cv2
import asyncio
import aiofiles
import numpy as np

from utils import time_decorator_async


async def read_image_async(filename, semaphore):
    async with semaphore:
        async with aiofiles.open(filename, mode='rb') as f:
            content = await f.read()
            img_array = np.asarray(bytearray(content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img


@time_decorator_async
async def read_images_async(directory, **kwargs):
    images = []
    tasks = []
    semaphore = asyncio.Semaphore(10)  # Start with a limit of 10 concurrent tasks
    # print(f"Reading images from directory: {directory}")
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            tasks.append(read_image_async(filepath, semaphore))
            # print(f"Scheduled task for {filepath}")
    results = await asyncio.gather(*tasks)
    for img in results:
        if img is not None:
            images.append(img)

    gc.collect()
    return len(images)
