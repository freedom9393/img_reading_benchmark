import tensorflow as tf

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    return image


def read_images_tensorflow(directory, batch_size=32):
    list_ds = tf.data.Dataset.list_files(os.path.join(directory, '*.jpg'))
    image_ds = list_ds.map(lambda x: tf.io.read_file(x))
    image_ds = image_ds.map(decode_image)

    # Batch and prefetch data
    image_ds = image_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return image_ds

images = read_images_tensorflow('path/to/images', 32)
