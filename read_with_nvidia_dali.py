from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import time


def run_with_pad(source_dir, batch_size, target_size, threads, total_images):
    @pipeline_def(batch_size=batch_size, num_threads=threads, device_id=0)
    def dali_image_reader_with_resize_and_pad():
        jpegs, labels = fn.readers.file(file_root=source_dir, random_shuffle=True)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.resize(images, resize_shorter=target_size[0], max_size=target_size)
        images = fn.pad(images, fill_value=0, axis_names="HW", shape=target_size)
        return images

    start_time = time.time()
    # Create the pipeline
    pipe = dali_image_reader_with_resize_and_pad()

    # Build the pipeline
    pipe.build()

    # Calculate the total number of batches needed
    num_batches = (total_images + batch_size - 1) // batch_size

    # Process all images
    for _ in range(num_batches):
        outputs = pipe.run()
        images_ = outputs[0].as_cpu().as_array()
        continue

    print(f'Spent time: {time.time() - start_time}')
