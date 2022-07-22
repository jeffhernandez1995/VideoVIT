import torchvision.datasets.video_utils

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset

import random


class VideoDataset(VisionDataset):
    def __init__(
        self,
        root,
        extensions=('mp4', 'avi'),
        transform=None
    ):
        super(VideoDataset).__init__()

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
        path, target = self.samples[idx]
        # Get video object
        vid = torchvision.io.VideoReader(path, "video")
        metadata = vid.get_metadata()

        # Seek and return frames
        max_seek = metadata["video"]['duration'][0]
        start = random.uniform(0., max_seek)
        frame = next(vid.seek(start))['data']
        if self.transform:
            frame = self.transform(frame)

        return frame, target
