import gc
import os
import time
import tensorflow as tf


def decode_and_pad_image(image, target_size):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, target_size[0], target_size[1])
    return image


def read_images_tensorflow(directory, **kwargs):
    list_ds = tf.data.Dataset.list_files(os.path.join(directory, '*.jpg'))
    image_ds = list_ds.map(lambda x: tf.io.read_file(x))
    image_ds = image_ds.map(lambda x: decode_and_pad_image(x, kwargs['target_size']))

    # Batch and prefetch data
    image_ds = image_ds.batch(kwargs['batch_size']).prefetch(buffer_size=tf.data.AUTOTUNE)

    return image_ds


def generate_with_tf(source_dir, batch_size, target_size):
    start_time = time.time()
    image_dataset = read_images_tensorflow(
        source_dir,
        batch_size=batch_size,
        target_size=target_size
    )

    for batch in image_dataset:
        res = batch.shape[0]
    print(f'tensorflow: {time.time() - start_time}')
    del image_dataset
    gc.collect()
