import nvidia.dali.pipeline as pipeline
from nvidia.dali import ops, types

class DaliPipeline(pipeline.Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir):
        super(DaliPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.FileReader(file_root=data_dir, random_shuffle=True)
        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        self.resize = ops.Resize(device='gpu', resize_x=224, resize_y=224)

    def define_graph(self):
        images = self.input()
        decoded_images = self.decode(images)
        resized_images = self.resize(decoded_images)
        return resized_images

pipe = DaliPipeline(batch_size=32, num_threads=4, device_id=0, data_dir='path/to/images')
pipe.build()
pipe_out = pipe.run()
images = pipe_out[0].as_cpu().as_array()
