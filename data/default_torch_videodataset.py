import torchvision.datasets.video_utils

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset

import numpy as np
import ffmpeg
import random


def get_video_size(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    try:
        n_frames = int(video_info['nb_frames'])
    except KeyError:
        n_frames = float(video_info['duration']) * eval(video_info['r_frame_rate'])
    frame_rate = eval(video_info['r_frame_rate'])
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height, int(n_frames), frame_rate, float(video_info['duration'])


class VideoDataset(VisionDataset):

    def __init__(
        self,
        root,
        extensions=('mp4', 'avi'),
        transform=None,
    ):
        super(VideoDataset, self).__init__(root)
        extensions = extensions

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get random sample
        success = False
        while not success:
            try:
                path, target = self.samples[idx]
                _, _, _, _, duration = get_video_size(path)
                start = random.uniform(0., duration)
                frame, _, _ = torchvision.io.read_video(
                    path,
                    start_pts=start,
                    end_pts=start,
                    pts_unit='sec',
                    output_format="TCHW"
                )
                success = True
            except Exception as e:
                print(e)
                print('skipped idx', idx)
                idx = np.random.randint(self.__len__())
        # Seek and return frames
        frame = self.transform(frame[0])
        return frame, target


# class VideoDataset(VisionDataset):
#     def __init__(
#         self,
#         root,
#         frames_per_clip,
#         step_between_clips=1,
#         frame_rate=None,
#         extensions=('mp4',),
#         transform=None,
#         _precomputed_metadata=None
#     ):
#         super(VideoDataset, self).__init__(root)
#         extensions = extensions

#         classes = list(sorted(list_dir(root)))
#         class_to_idx = {classes[i]: i for i in range(len(classes))}

#         self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
#         self.classes = classes
#         video_list = [x[0] for x in self.samples]
#         self.video_clips = VideoClips(
#             video_list,
#             frames_per_clip,
#             step_between_clips,
#             frame_rate,
#             _precomputed_metadata,
#         )
#         self.transform = transform

#     def __len__(self):
#         return self.video_clips.num_clips()

#     def __getitem__(self, idx):
#         success = False
#         while not success:
#             try:
#                 video, _, info, video_idx = self.video_clips.get_clip(idx)
#                 success = True
#             except:
#                 print('skipped idx', idx)
#                 idx = np.random.randint(self.__len__())

#         label = self.samples[video_idx][1]
#         if self.transform is not None:
#             video = self.transform(video)

#         return video, label
