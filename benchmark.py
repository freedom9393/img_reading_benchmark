import gc
import asyncio

from read_with_tensorflow import generate_with_tf
from read_with_loop import read_images_iteratively
from read_with_multithreading import read_images_multithreading
from read_with_asynchronous import read_images_async


source_dir = "/home/real/example_imgs/example"
target_size = (640, 640)
batch_size = 16
threads = 4

generate_with_tf(source_dir, batch_size, target_size)
read_images_iteratively(source_dir, method='loop')
read_images_multithreading(source_dir, method='multithread', max_workers=threads)
asyncio.run(read_images_async(source_dir, method='async'))

gc.collect()
