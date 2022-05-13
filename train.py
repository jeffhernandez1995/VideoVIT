from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow as tf
import glob

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
from itertools import product
import numpy as np
import random

# Setting seeds for reproducibility.
SEED = 42
keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# DATA
BUFFER_SIZE = 1024
BATCH_SIZE = 64
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (15, 120, 160, 3)
TIME_LEN = INPUT_SHAPE[0]
OUTPUT_SHAPE = (120, 160, 3)
NUM_CLASSES = 6

# OPTIMIZER
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4

# PRETRAINING
EPOCHS = 500

# AUGMENTATION
IMAGE_SIZE = 48  # We will resize input images to this size.
PATCH_SIZE = 6  # Size of the patches to be extracted from the input images.
CROP_SIZE = 100
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
MASK_PROPORTION = 0.75  # We have found 75% masking to give us the best results.

# ENCODER and DECODER
LAYER_NORM_EPS = 1e-6
ENC_PROJECTION_DIM = 128
DEC_PROJECTION_DIM = 64
ENC_NUM_HEADS = 4
ENC_LAYERS = 8
DEC_NUM_HEADS = 4
DEC_LAYERS = (
    4  # The decoder is lightweight but should be reasonably deep for reconstruction.
)
ENC_TRANSFORMER_UNITS = [
    ENC_PROJECTION_DIM * 2,
    ENC_PROJECTION_DIM,
]  # Size of the transformer layers.
DEC_TRANSFORMER_UNITS = [
    DEC_PROJECTION_DIM * 2,
    DEC_PROJECTION_DIM,
]

"""Returns a Dataset for reading from a SageMaker PipeMode channel."""
features = {
    'video': tf.io.FixedLenFeature([], tf.string),
    'frame': tf.io.FixedLenFeature([], tf.string),
}


def parse(record):

    parsed = tf.io.parse_single_example(
        serialized=record,
        features=features
    )
    video_raw = parsed['video']
    video_raw = tf.io.decode_raw(video_raw, tf.uint8)
    video_raw = tf.cast(video_raw, tf.float32)

    frame_raw = parsed['frame']
    frame_raw = tf.io.decode_raw(frame_raw, tf.uint8)
    frame_raw = tf.cast(frame_raw, tf.float32)

    video_raw = tf.reshape(video_raw, INPUT_SHAPE)
    frame_raw = tf.reshape(frame_raw, OUTPUT_SHAPE)

    video_raw = tf.concat([video_raw, video_raw, video_raw], axis=-1)
    frame_raw = tf.concat([frame_raw, frame_raw, frame_raw], axis=-1)
    return video_raw, frame_raw


files = tf.data.Dataset.list_files("datasets/KTH_tfrecords/training/*.tfrecord")
train_ds = files.interleave(
    lambda x: tf.data.TFRecordDataset(x).prefetch(100),
    cycle_length=8
)
train_ds = train_ds.map(parse, num_parallel_calls=AUTO)
train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)


files = tf.data.Dataset.list_files("datasets/KTH_tfrecords/validation/*.tfrecord")
val_ds = files.interleave(
    lambda x: tf.data.TFRecordDataset(x).prefetch(100),
    cycle_length=8
)
val_ds = val_ds.map(parse, num_parallel_calls=AUTO)
val_ds = val_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)


def left_right_flip(video, frame):
    '''
    Performs tf.image.flip_left_right on entire list of video frames.
    Work around since the random selection must be consistent for entire video
    :param video: Tensor constaining video frames (N,H,W,3)
    :return: video: Tensor constaining video frames left-right flipped (N,H,W,3)
    '''
    video_list = tf.unstack(video, axis=1)
    for i in range(len(video_list)):
        video_list[i] = tf.image.flip_left_right(video_list[i])
    video = tf.stack(video_list, axis=1)
    frame = tf.image.flip_left_right(frame)
    return video, frame


def random_crop(video, frame, size):
    # (T, H, W, 3)
    shape = tf.shape(video)
    size = tf.convert_to_tensor(size, dtype=shape.dtype)
    h_diff = shape[2] - size[1]
    w_diff = shape[3] - size[0]

    dtype = shape.dtype
    rands = tf.random.uniform(shape=[2], minval=0, maxval=dtype.max, dtype=dtype)
    h_start = tf.cast(rands[0] % (h_diff + 1), dtype)
    w_start = tf.cast(rands[1] % (w_diff + 1), dtype)
    size = tf.cast(size, tf.int32)
    video_list = tf.unstack(video, axis=1)
    for i in range(len(video_list)):
        video_list[i] = tf.image.crop_to_bounding_box(
            video_list[i],
            h_start, w_start,
            size[1], size[0]
        )
    video = tf.stack(video_list, axis=1)
    frame = tf.image.crop_to_bounding_box(
        frame,
        h_start, w_start,
        size[1], size[0]
    )
    return video, frame


def resize(video, frame, size):
    video_list = tf.unstack(video, axis=1)
    for i in range(len(video_list)):
        video_list[i] = tf.image.resize(
            video_list[i],
            size
        )
    video = tf.stack(video_list, axis=1)
    frame = tf.image.resize(frame, size)
    return video, frame


class TrainingPreprocessing(tf.keras.layers.Layer):
    def __init__(
        self,
        crop_size=(CROP_SIZE, CROP_SIZE),
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        **kwargs
    ):
        self.crop_size = crop_size
        self.image_size = image_size
        super(TrainingPreprocessing, self).__init__()

    def call(self, data):
        video, frame = data
        video, frame = random_crop(video, frame, self.crop_size)
        video, frame = resize(video, frame, self.image_size)
        sample = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
        option = tf.less(sample, 0.5)
        video, frame = tf.cond(
            option,
            lambda: left_right_flip(video, frame),
            lambda: (video, frame)
        )
        video = tf.cast(video, tf.float32) * (1 / 255.)
        frame = tf.cast(frame, tf.float32) * (1 / 255.)
        return video, frame


class TestingPreprocessing(tf.keras.layers.Layer):
    def __init__(self, size=(IMAGE_SIZE, IMAGE_SIZE), **kwargs):
        self.size = size
        super(TestingPreprocessing, self).__init__()

    def call(self, data):
        video, frame = data
        video, frame = resize(video, frame, self.size)
        video = tf.cast(video, tf.float32) * (1 / 255.)
        frame = tf.cast(frame, tf.float32) * (1 / 255.)
        return video, frame


def get_train_augmentation_model(
    input_shape=INPUT_SHAPE,
    output_shape=OUTPUT_SHAPE,
    crop_size=(CROP_SIZE, CROP_SIZE),
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
):
    inputs = keras.Input(
        shape=input_shape,
        name="Original Video"
    )
    output = keras.Input(
        shape=output_shape,
        name="Next frame"
    )
    [aug, out] = TrainingPreprocessing(
        crop_size=crop_size,
        image_size=image_size
    )([inputs, output])
    return keras.Model(inputs=[inputs, output], outputs=[aug, out], name="train_data_augmentation")


def get_test_augmentation_model(
    input_shape=INPUT_SHAPE,
    output_shape=OUTPUT_SHAPE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
):
    inputs = keras.Input(
        shape=input_shape,
        name="Original Video"
    )
    output = keras.Input(
        shape=output_shape,
        name="Next frame"
    )
    [aug, out] = TestingPreprocessing(
        image_size=image_size
    )([inputs, output])
    return keras.Model(inputs=[inputs, output], outputs=[aug, out], name="test_data_augmentation")


class Patches(layers.Layer):
    def __init__(self, patch_size=PATCH_SIZE, time_len=TIME_LEN, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.time_len = time_len
        self.resize = layers.Reshape((-1, patch_size * patch_size * 3))

    def call(self, data):
        video, frame = data
        # Create patches from the input images
        video_list = tf.unstack(video, axis=1)
        for i in range(len(video_list)):
            patches = tf.image.extract_patches(
                images=video_list[i],
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )

            # Reshape the patches to (batch, num_patches, patch_area) and return it.
            video_list[i] = self.resize(patches)
        video = tf.stack(video_list, axis=1)
        frame = tf.image.extract_patches(
            images=frame,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        return video, frame

    def show_patched_image(self, video, patches):
        # This is a utility function which accepts a batch of images and its
        # corresponding patches and help visualize one image and its patches
        # side by side.
        idx = np.random.choice(patches.shape[0])
        n = int(np.sqrt(patches.shape[-2]))
        print(f"Index selected: {idx}.")

        fig = plt.figure()
        gs = gridspec.GridSpec(n, 2 * n, figure=fig, hspace=0.08, wspace=0.1)
        ax = fig.add_subplot(gs[:n, n: 2*n])
        big_im = ax.imshow(video[idx, 0, ...])
        ax.set_axis_off()

        grid = list(product(range(n), range(n)))
        patch_list = []
        for i in range(patches.shape[-2]):
            ax = fig.add_subplot(gs[grid[i][0], grid[i][1]])
            patch_img = tf.reshape(
                patches[idx, 0, i, :],
                (self.patch_size, self.patch_size, 3)
            )
            im = ax.imshow(patch_img)
            patch_list.append(im)
            ax.set_axis_off()
        plt.close()
        def init():
            big_im.set_data(video[idx,0,...])
            for i, im in enumerate(patch_list):
                patch_img = tf.reshape(
                    patches[idx, 0, i, :],
                    (self.patch_size, self.patch_size, 3)
                )
                im.set_data(patch_img)
        def animate(j):
            big_im.set_data(video[idx,j,...])
            for i, im in enumerate(patch_list):
                patch_img = tf.reshape(
                    patches[idx, j, i, :],
                    (self.patch_size, self.patch_size, 3)
                )
                im.set_data(patch_img)
        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=video.shape[1],
            interval=50
        )
        return anim, idx

    # taken from https://stackoverflow.com/a/58082878/10319735
    def reconstruct_from_patch(self, patch):
        # This utility function takes patches from a *single* image and
        # reconstructs it back into the image. This is useful for the train
        # monitor callback.
        num_patches = patch.shape[-2]
        n = int(np.sqrt(num_patches))
        patch = tf.reshape(patch, (self.time_len, num_patches, self.patch_size, self.patch_size, 3))
        video = []
        for i in range(self.time_len):
            rows = tf.split(patch[i], n, axis=0)
            rows = [tf.concat(tf.unstack(x), axis=1) for x in rows]
            reconstructed = tf.concat(rows, axis=0)
            video.append(reconstructed)
        return tf.stack(video, axis=0)


class PatchEncoder(layers.Layer):
    def __init__(
        self,
        patch_size=PATCH_SIZE,
        time_len=TIME_LEN,
        projection_dim=ENC_PROJECTION_DIM,
        mask_proportion=MASK_PROPORTION,
        downstream=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.time_len = time_len
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream

        # This is a trainable mask token initialized randomly from a normal
        # distribution.
        self.mask_token = tf.Variable(
            tf.random.normal([self.time_len, 1, patch_size * patch_size * 3]), trainable=True
        )

    def build(self, input_shape):
        (_, self.num_frames, self.num_patches, self.patch_area) = input_shape

        # Create the projection layer for the patches.
        self.projection = layers.GRU(units=self.projection_dim)
        # Create the positional embedding layer.
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )

        # Number of patches that will be masked.
        self.num_mask = int(self.mask_proportion * self.num_patches)

    def call(self, patches):
        # patches: (B, T, N, ps*ps)
        # Get the positional embeddings.
        batch_size = tf.shape(patches)[0]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1, 1]
        )  # (B, num_patches, projection_dim)

        # Embed the patches. (GRU)
        projection = tf.unstack(patches, axis=-2)
        for i in range(len(projection)):
            projection[i] = self.projection(projection[i])
        # (B, num_patches, projection_dim)
        projection = tf.stack(projection, axis=1)

        patch_embeddings = (
            projection + pos_embeddings
        )  # (B, num_patches, projection_dim)

        if self.downstream:
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            # The encoder input is the unmasked patch embeddings. Here we gather
            # all the patches that should be unmasked.
            unmasked_embeddings = tf.gather(
                patch_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)

            # Get the unmasked and masked position embeddings. We will need them
            # for the decoder.
            unmasked_positions = tf.gather(
                pos_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)
            masked_positions = tf.gather(
                pos_embeddings, mask_indices, axis=1, batch_dims=1
            )  # (B, mask_numbers, projection_dim)

            # Repeat the mask token number of mask times.
            # Mask tokens replace the masks of the image.
            # mask_tokens: (T, ps*ps)
            mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=1)
            # mask_tokens = (mask_numbers, projection_dim) 
            mask_tokens = tf.repeat(
                mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0
            )
            # Embed the tokens (GRU)
            mask_tokens = tf.unstack(mask_tokens, axis=-2)
            for i in range(len(mask_tokens)):
                mask_tokens[i] = self.projection(mask_tokens[i])
            # (B, num_patches, projection_dim)
            mask_tokens = tf.stack(mask_tokens, axis=1)d
            # Get the masked embeddings for the tokens.
            masked_embeddings = mask_tokens + masked_positions
            return (
                unmasked_embeddings,  # Input to the encoder.
                masked_embeddings,  # First part of input to the decoder.
                unmasked_positions,  # Added to the encoder outputs.
                mask_indices,  # The indices that were masked.
                unmask_indices,  # The indices that were unmaksed.
            )

    def get_random_indices(self, batch_size):
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices

    def generate_masked_image(self, patches, unmask_indices):
        # Choose a random patch and it corresponding unmask index.
        idx = np.random.choice(patches.shape[0])
        patch = patches[idx]
        unmask_index = unmask_indices[idx]

        # Build a numpy array of same shape as patch.
        new_patch = np.zeros_like(patch)

        # Iterate of the new_patch and plug the unmasked patches.
        for i in range(unmask_index.shape[0]):
            new_patch[:, unmask_index[i], ...] = patch[:, unmask_index[i], ...]
        return new_patch, idx


def mlp(x, dropout_rate, hidden_units):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_encoder(
    num_heads=ENC_NUM_HEADS,
    num_layers=ENC_LAYERS,
    projection_dim=ENC_PROJECTION_DIM,
    transformer_units=ENC_TRANSFORMER_UNITS,
    epsilon=LAYER_NORM_EPS,
):
    inputs = layers.Input(
        (None, projection_dim)
    )
    x = inputs

    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=epsilon)(x)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=epsilon)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x = layers.Add()([x3, x2])

    outputs = layers.LayerNormalization(epsilon=epsilon)(x)
    return keras.Model(inputs, outputs, name="mae_encoder")


def create_decoder(
    num_layers=DEC_LAYERS,
    num_heads=DEC_NUM_HEADS,
    num_patches=NUM_PATCHES,
    enc_projection_dim=ENC_PROJECTION_DIM,
    dec_projection_dim=DEC_PROJECTION_DIM,
    epsilon=LAYER_NORM_EPS,
    transformer_units=DEC_TRANSFORMER_UNITS,
    image_size=IMAGE_SIZE
):
    inputs = layers.Input(
        (num_patches, enc_projection_dim)
    )
    x = layers.Dense(dec_projection_dim)(inputs)

    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=epsilon)(x)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dec_projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=epsilon)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=epsilon)(x)
    x = layers.Flatten()(x)
    pre_final = layers.Dense(units=image_size * image_size * 3, activation='sigmoid')(x) # tanh sigmoid
    outputs = layers.Reshape((image_size, image_size, 3))(pre_final)

    return keras.Model(inputs, outputs, name="mae_decoder")


class MaskedAutoencoder(keras.Model):
    def __init__(
        self,
        input_shape=INPUT_SHAPE,
        output_shape=OUTPUT_SHAPE,
        crop_size=(CROP_SIZE, CROP_SIZE),
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        time_len=TIME_LEN,
        mask_proportion=MASK_PROPORTION,
        enc_projection_dim=ENC_PROJECTION_DIM,
        enc_transformer_units=ENC_TRANSFORMER_UNITS,
        num_enc_heads=ENC_NUM_HEADS,
        num_enc_layers=ENC_LAYERS,
        num_dec_layers=DEC_LAYERS,
        num_dec_heads=DEC_NUM_HEADS,
        dec_projection_dim=DEC_PROJECTION_DIM,
        dec_transformer_units=DEC_TRANSFORMER_UNITS,
        epsilon=LAYER_NORM_EPS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_augmentation_model = get_train_augmentation_model(
            input_shape=input_shape,
            output_shape=output_shape,
            crop_size=crop_size,
            image_size=image_size,
        )
        self.test_augmentation_model = get_test_augmentation_model(
            input_shape=input_shape,
            output_shape=output_shape,
            image_size=image_size,
        )
        self.patch_layer = Patches(
            patch_size=patch_size,
            time_len=time_len,
        )
        self.patch_encoder = PatchEncoder(
            patch_size=patch_size,
            time_len=time_len,
            projection_dim=enc_projection_dim,
            mask_proportion=mask_proportion
            downstream=self.training
        )
        self.encoder = create_encoder(
            num_heads=num_enc_heads,
            num_layers=num_enc_layers,
            projection_dim=enc_projection_dim,
            transformer_units=enc_transformer_units,
            epsilon=epsilon,
        )
        self.decoder = create_decoder(
            num_layers=num_dec_layers,
            num_heads=num_dec_heads,
            num_patches=num_patches,
            enc_projection_dim=enc_projection_dim,
            dec_projection_dim=dec_projection_dim,
            epsilon=epsilon,
            transformer_units=dec_transformer_units,
            image_size=image_size[0],
        )
        self.resize = layers.Reshape((-1, self.patch_layer.patch_size * self.patch_layer.patch_size * 1))

    def calculate_loss(self, data, test=False):
        videos, next_frames = data
        # Augment the input images.
        if test:
            aug_videos, next_frame = self.test_augmentation_model([videos, next_frames])
        else:
            aug_videos, next_frame = self.train_augmentation_model([videos, next_frames])

        # Patch the augmented images.
        # patches = self.patch_layer(aug_videos)
        vid_patches, frame_patches =  self.patch_layer([aug_videos, next_frame])

        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(vid_patches)

        # Pass the unmaksed patche to the encoder.
        encoder_outputs = self.encoder(unmasked_embeddings)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)

        # Decode the inputs.
        decoder_outputs = self.decoder(decoder_inputs)
        decoder_patches = tf.image.extract_patches(
            images=decoder_outputs,
            sizes=[1, self.patch_layer.patch_size, self.patch_layer.patch_size, 1],
            strides=[1, self.patch_layer.patch_size, self.patch_layer.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # Calculate loss on all patches.
        loss_output = self.resize(decoder_patches)
        loss_patch = self.resize(frame_patches)
        # Calculate loss on masked patches.
        # loss_patch = tf.gather(
        #     loss_patch,
        #     mask_indices,
        #     axis=1,
        #     batch_dims=1
        # )
        # loss_output = tf.gather(
        #     loss_output,
        #     mask_indices,
        #     axis=1,
        #     batch_dims=1
        # )
        # Compute the total loss.
        # Calculate loss on masked patches
        # total_loss = self.compiled_loss(loss_patch, loss_output)
        # # Calculate loss on all outputs
        total_loss = self.compiled_loss(frame_patches, decoder_patches)

        return total_loss, loss_patch, loss_output

    def train_step(self, data):
        videos, next_frames = data
        with tf.GradientTape() as tape:
            total_loss, loss_patch, loss_output = self.calculate_loss([videos, next_frames])
        # Apply gradients.
        train_vars = [
            self.train_augmentation_model.trainable_variables,
            self.patch_layer.trainable_variables,
            self.patch_encoder.trainable_variables,
            self.encoder.trainable_variables,
            self.decoder.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        # import pdb
        # pdb.set_trace()
        tv_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)

        # Report progress.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        videos, next_frames = data
        total_loss, loss_patch, loss_output = self.calculate_loss([videos, next_frames], test=True)
        # Update the trackers.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}


mae_model = MaskedAutoencoder(
    input_shape=INPUT_SHAPE,
    output_shape=OUTPUT_SHAPE,
    crop_size=(CROP_SIZE, CROP_SIZE),
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    patch_size=PATCH_SIZE,
    num_patches=NUM_PATCHES,
    time_len=TIME_LEN,
    mask_proportion=MASK_PROPORTION,
    enc_projection_dim=ENC_PROJECTION_DIM,
    enc_transformer_units=ENC_TRANSFORMER_UNITS,
    num_enc_heads=ENC_NUM_HEADS,
    num_enc_layers=ENC_LAYERS,
    num_dec_layers=DEC_LAYERS,
    num_dec_heads=DEC_NUM_HEADS,
    dec_projection_dim=DEC_PROJECTION_DIM,
    dec_transformer_units=DEC_TRANSFORMER_UNITS,
    epsilon=LAYER_NORM_EPS,
)

class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


total_steps = int((1815 / BATCH_SIZE) * EPOCHS)
warmup_epoch_percentage = 0.15
warmup_steps = int(total_steps * warmup_epoch_percentage)
scheduled_lrs = WarmUpCosine(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=0.0,
    warmup_steps=warmup_steps,
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='models/KTH/',
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

train_callbacks = [
    model_checkpoint_callback
]

optimizer = tfa.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)
# optimizer = tf.keras.optimizers.RMSprop(
#     learning_rate=scheduled_lrs
# )

# Compile and pretrain the model.
mae_model.compile(
    optimizer=optimizer,
    loss=keras.losses.MeanSquaredError(),
    metrics=["mae"],
    # run_eagerly=True
)
history = mae_model.fit(
    train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=train_callbacks,
)
