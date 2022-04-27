import numpy as np
import tensorflow as tf
import glob
import re

import random
import string
import ffmpeg
import difflib


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


PREFIX = f'datasets/KTH_raw'

CATEGORIES = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking"
]

# Dataset are divided according to the instruction at:
# http://www.nada.kth.se/cvap/actions/00sequences.txt
TRAIN_PEOPLE_ID = [11, 12, 13, 14, 15, 16, 17, 18]
DEV_PEOPLE_ID = [19, 20, 21, 23, 24, 25, 1, 4]
TEST_PEOPLE_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]
NUM_FRAMES = 15
tf_sizes = {
    'training': 1815,
    'validation': 2013,
    'testing': 2193
}
mode = 'testing'
tf_size = tf_sizes[mode]
TASK_ID = 'prediction'
OUT_PATH = f'datasets/KTH_tfrecords/{mode}'


def parse_sequence_file(filename):
    print(f"Parsing ..{filename}")

    # Read 00sequences.txt file.
    with open(f'{filename}', 'r') as content_file:
        content = content_file.readlines()

    # Remove top of file with information about splits
    content = content[20:]
    content = [f.replace('\n', '') for f in content]

    filenames = ['{}_uncomp.avi'.format(line.split("\t")[0]) for line in content]
    frames = [line.split("\t")[-1].replace(' ', '').split(",") for line in content]
    frames = [
        [tuple(map(int, val.split("-"))) for val in f] #
        for f in frames
    ]
    frames_idx = dict(zip(filenames, frames))
    return frames_idx


def make_raw_dataset(
    dataset="train",
    sequence_file=f'{PREFIX}/sequences.txt',
    categories=CATEGORIES,
):
    if dataset == "training":
        IDS = TRAIN_PEOPLE_ID
    elif dataset == "validation":
        IDS = DEV_PEOPLE_ID
    else:
        IDS = TEST_PEOPLE_ID

    input_vids = []
    output_img = []
    frames_idx = parse_sequence_file(sequence_file)
    for cat in categories:
        folder_name = f'{PREFIX}/{cat}'
        filenames = sorted(glob.glob(f'{folder_name}/*.avi'))
        for filename in filenames:
            # Get id of person in this video.
            person_id = int(re.findall(r'\d+', filename.split("_")[1])[0])
            if person_id not in IDS:
                continue
            probe = ffmpeg.probe(filename)
            video_info = next(
                s for s in probe['streams'] if s['codec_type'] == 'video'
            )
            width = int(video_info['width'])
            height = int(video_info['height'])
            in_file = ffmpeg.input(filename)
            if filename.split('/')[-1] not in frames_idx.keys():
                continue
            videos = []
            for (start_frame, end_frame) in frames_idx[filename.split('/')[-1]]:
                out, err = (
                    in_file
                    .trim(start_frame=start_frame, end_frame=end_frame)
                    .output('pipe:', format='rawvideo', pix_fmt='gray')
                    .run(capture_stdout=True)
                )
                video = (
                    np.frombuffer(out, np.uint8).reshape([-1, height, width])
                )
                size_ = (video.shape[0] // (NUM_FRAMES + 1)) * (NUM_FRAMES + 1)
                video = video[:size_, ...].reshape(
                    [-1, (NUM_FRAMES + 1), height, width]
                )
                videos.append(video)
            videos = np.concatenate(videos, axis=0)
            input_vids.append(videos[:, :NUM_FRAMES, ...])
            output_img.append(videos[:, -1, ...])
    inputs = np.concatenate(input_vids, axis=0)
    outputs = np.concatenate(output_img, axis=0)
    return inputs, outputs


inputs, outputs = (
    make_raw_dataset(mode, f'{PREFIX}/sequences.txt', CATEGORIES)
)
N, T, W, H = inputs.shape

max_index = (N // tf_size) * tf_size

indices = np.random.choice(
    np.arange(N),
    size=max_index,
    replace=False
)

inputs = inputs[indices].reshape([-1, tf_size, T, W, H])
outputs = outputs[indices].reshape([-1, tf_size, W, H])
print(inputs.shape)

for i in range(inputs.shape[0]):
    name = ''.join(random.choices(
        string.ascii_uppercase +
        string.digits, k=7
    ))
    filename = f'{OUT_PATH}/{TASK_ID}-{name}_{str(i).zfill(4)}.tfrecord'
    with tf.io.TFRecordWriter(filename) as writer:
        for j in range(tf_size):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'video': _bytes_feature(
                                inputs[i, j].tobytes()
                        ),
                        'frame': _bytes_feature(
                                outputs[i, j].tobytes()
                        )
                    }
                )
            )

            writer.write(example.SerializeToString())
    print(f"{tf_size} instances saved in: {filename}")
