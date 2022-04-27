from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow as tf
import glob

import matplotlib.pyplot as plt
import numpy as np
import random

# Setting seeds for reproducibility.
SEED = 42
keras.utils.set_random_seed(SEED)

# DATA
BUFFER_SIZE = 1024
BATCH_SIZE = 256
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (15, 120, 160)
OUTPUT_SHAPE = (120, 160)
NUM_CLASSES = 6

# OPTIMIZER
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4

# PRETRAINING
EPOCHS = 100

# AUGMENTATION
IMAGE_SIZE = 120  # We will resize input images to this size.
PATCH_SIZE = 12  # Size of the patches to be extracted from the input images.
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
MASK_PROPORTION = 0.75  # We have found 75% masking to give us the best results.

# ENCODER and DECODER
LAYER_NORM_EPS = 1e-6
ENC_PROJECTION_DIM = 128
DEC_PROJECTION_DIM = 64
ENC_NUM_HEADS = 4
ENC_LAYERS = 6
DEC_NUM_HEADS = 4
DEC_LAYERS = (
    2  # The decoder is lightweight but should be reasonably deep for reconstruction.
)
ENC_TRANSFORMER_UNITS = [
    ENC_PROJECTION_DIM * 2,
    ENC_PROJECTION_DIM,
]  # Size of the transformer layers.
DEC_TRANSFORMER_UNITS = [
    DEC_PROJECTION_DIM * 2,
    DEC_PROJECTION_DIM,
]

filenames = sorted(glob.glob("datasets/KTH_tfrecords/training/*.tfrecord"))
train_ds = tf.data.TFRecordDataset(filenames)
train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

filenames = sorted(glob.glob("datasets/KTH_tfrecords/validation/*.tfrecord"))
val_ds = tf.data.TFRecordDataset(filenames)
val_ds = val_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

filenames = sorted(glob.glob("datasets/KTH_tfrecords/testing/*.tfrecord"))
test_ds = tf.data.TFRecordDataset(filenames)
test_ds = test_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)
