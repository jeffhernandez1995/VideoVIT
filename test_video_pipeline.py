import ffmpeg
import numpy as np
import decord as de
from io import BytesIO
import cv2
import subprocess
from functools import partial

from torchvision.datasets import DatasetFolder
from torch.utils.data import Subset

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField

from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import IntDecoder

from dataclasses import replace
from typing import Callable, Optional, Tuple, Type

# to create a field
from ffcv.fields.base import Field, ARG_TYPE

# Operations in ffcv
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.libffcv import memcpy

Compiler.set_enabled(False)


class ImageFromVideoDecoder(Operation):
    def __init__(
        self
    ):
        super().__init__()

    def declare_state_and_memory(
        self,
        previous_state: State
    ) -> Tuple[State, AllocationQuery]:
        widths = self.metadata['width']
        heights = self.metadata['height']
        max_width = widths.max()
        max_height = heights.max()
        min_height = heights.min()
        min_width = widths.min()
        if min_width != max_width or max_height != min_height:
            msg = (
                'SimpleVideoDecoder ony supports constant shape videos, ',
                'consider RandomResizedCropVideoDecoder ',
                'or CenterCropVideoDecoder instead.'
            )
            raise TypeError(msg)

        biggest_shape = (max_height, max_width, 3)
        my_dtype = np.dtype('<u1')

        return (
            replace(previous_state, jit_mode=True,
                    shape=biggest_shape, dtype=my_dtype),
            AllocationQuery(biggest_shape, my_dtype)
        )

    def generate_code(self) -> Callable:
        mem_read = self.memory_read
        my_range = Compiler.get_iterator()
        # my_bytes2video = Compiler.compile(de.VideoReader)
        my_memcpy = Compiler.compile(memcpy)

        def decode(batch_indices, destination, metadata, storage_state):
            for dst_ix in my_range(len(batch_indices)):
                source_ix = batch_indices[dst_ix]
                field = metadata[source_ix]
                video_data = mem_read(field['data_ptr'], storage_state)
                height, width, fps, n_frames = field['height'], field['width'], field['fps'], field['n_frames']
                vidbytes = video_data.tobytes()
                # sample random frame
                frame_ix = np.random.randint(0, n_frames)
                video = de.VideoReader(BytesIO(vidbytes))

                vid_np = video[frame_ix].asnumpy()

                # resize to fit destination
                vid_np = vid_np.reshape((height, width, 3))

                my_memcpy(vid_np, destination[dst_ix])

            return destination[:len(batch_indices)]

        decode.is_parallel = True
        return decode


class VideoField(Field):

    def __init__(
        self,
        max_width: int = None,
    ) -> None:
        self.max_width = max_width

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('fps', '<u2'),
            ('n_frames', '<u2'),
            ('width', '<u2'),
            ('height', '<u2'),
            ('data_ptr', '<u8'),
        ])

    def get_decoder_class(self) -> Type[Operation]:
        return ImageFromVideoDecoder

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return VideoField()

    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]

    def encode(self, destination, video_path, malloc):

        width, height, n_frames, frame_rate = get_video_size(video_path)

        destination['width'] = width
        destination['height'] = height
        destination['n_frames'] = n_frames
        destination['fps'] = frame_rate

        with open(video_path, 'rb') as fid:
            as_video = fid.read()
            as_video = np.frombuffer(as_video, dtype=np.uint8)

        destination['data_ptr'], storage = malloc(as_video.nbytes)
        storage[:] = as_video


def get_video_size(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    try:
        n_frames = int(video_info['nb_frames'])
    except KeyError:
        n_frames = int(video_info['duration']) * eval(video_info['r_frame_rate'])
    frame_rate = eval(video_info['r_frame_rate'])
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height, n_frames, frame_rate


def video2bytes(filename, width=None, height=None):
    if width is None and height is None:
        width, height, n_frames, frame_rate = get_video_size(filename)
    elif width is not None and height is None:
        height = -1
    elif width is None and height is not None:
        width = -1
    args = (
        ffmpeg
        .input(filename)
        .filter('scale', width, height)
        .output('pipe:', format='avi')
        .compile()
    )
    procces = subprocess.Popen(args, stdout=subprocess.PIPE)
    return procces.stdout.read()


# video_name = "datasets/KTH_raw/boxing/person01_boxing_d1_uncomp.avi"
# with open(video_name, 'rb') as fid:
#     video_bytes = fid.read()
#     as_video = np.frombuffer(video_bytes, dtype=np.uint8)

# print(as_video.shape)
# vr = de.VideoReader(BytesIO(video_bytes))
# print(len(vr))
# assert 2 == 3

# dataset = DatasetFolder(
#     root='datasets/KTH_raw/',
#     loader=lambda x: x,
#     extensions=('.mp4', '.avi'),
# )
# dataset = Subset(dataset, range(10))

# writer = DatasetWriter('test.beton', {
#     'video': VideoField(max_width=160),
#     'label': IntField(),
# }, num_workers=2)

# writer.from_indexed_dataset(dataset, chunksize=100)
# assert 2 == 1

video_pipeline = [ImageFromVideoDecoder()]
label_pipeline = [IntDecoder()]
pipelines = {
    'video': video_pipeline,
    'label': label_pipeline
}

loader = Loader(
    'test.beton',
    batch_size=1,
    num_workers=4,
    order=OrderOption.SEQUENTIAL,
    pipelines=pipelines,
    custom_fields={
        'video': VideoField(max_width=160),
    }
)

for image, label in loader:
    print(type(image), type(label))
    assert 2 == 1
