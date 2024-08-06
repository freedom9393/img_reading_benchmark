import os
import cv2
import asyncio
import aiofiles

async def read_image_async(filename):
    async with aiofiles.open(filename, mode='rb') as f:
        content = await f.read()
        img_array = np.asarray(bytearray(content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

async def read_images_async(directory):
    images = []
    tasks = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            tasks.append(read_image_async(os.path.join(directory, filename)))
    results = await asyncio.gather(*tasks)
    for img in results:
        if img is not None:
            images.append(img)
    return images

images = asyncio.run(read_images_async('path/to/images'))
