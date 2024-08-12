import gc
from read_with_tensorflow import generate_with_tf


source_dir = "/home/real/example_imgs/example"
target_size = (640, 640)
batch_size = 16
threads = 4

generate_with_tf(source_dir, batch_size, target_size)

gc.collect()
