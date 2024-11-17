import argparse
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from dataset_configuration import DatasetConfiguration

from cnn_models.simple_model import simple_model
# from cnn_models.multi_branch_model import multi_branch_model
from cnn_models.multi_branch_binary_model import multi_branch_binary_model

C_LOGGER_NAME = "explanation"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def make_gradcam_heatmap(profile, model, last_conv_idx, pred_index=None):
    """
    source: https://keras.io/examples/vision/grad_cam/

    last_conv_layer_name -> last_FC_layer
    """
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(index=last_conv_idx).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    profile_t = tf.convert_to_tensor(profile)

    with tf.GradientTape() as tape1:
        A, S = grad_model(profile_t)
        if pred_index is None:
            pred_index = tf.argmax(S[0])
        Y_c = S[:, pred_index]
        S_c = tf.exp(Y_c)

    S_deriv1 = tape1.gradient(S_c, A)

    S_deriv1_2 = S_deriv1 ** 2
    S_deriv1_3 = S_deriv1_2 * S_deriv1

    A_sum = tf.reduce_sum(A, 1)

    a = S_deriv1_2 / ((2 * S_deriv1_2) + (A_sum * S_deriv1_3))

    S_deriv_relu = tf.maximum(S_deriv1, 0)

    w = a @ tf.transpose(S_deriv_relu, (0, 2, 1))

    L = w @ A
    L_finel = tf.reduce_sum(L, axis=2)
    L_finel = tf.squeeze(L_finel)
    heatmap = tf.maximum(L_finel, 0)
    old_range = tf.math.reduce_max(heatmap) - tf.math.reduce_min(heatmap)
    if old_range == 0:
        logger.info("Old range == 0... %s", heatmap)
        heatmap = tf.fill(tf.shape(heatmap), 0)
    else:
        heatmap = (heatmap - tf.math.reduce_min(heatmap)) / (
                tf.math.reduce_max(heatmap) - tf.math.reduce_min(heatmap)
        )

    return heatmap.numpy(), pred_index


def read_data(paths):
    logger.info("Loading data...")

    # Loading array with profiles
    profiles = np.load(paths.profiles_npy)
    logger.info("%s", paths.profiles_npy)

    model = load_model(paths.model)
    logger.info("%s", paths.model)

    logger.info("Loading data... Done!")

    return profiles, model


def process(config, paths):
    profiles, model = read_data(paths)

    profile_len = config("profile_length")

    if profile_len != profiles.shape[1]:
        logger.error(
            "Profile length in %s don't match with data in %s",
            paths.config,
            paths.profiles_npy,
        )

    x_test = profiles.reshape((profiles.shape[0], profiles.shape[1], 1))

    logger.info("%s", model.summary())

    layer_index = 5
    logger.info("Last conv layer: %s", model.layers[layer_index].name)

    heatmaps_output = np.zeros(x_test.shape[:-1])
    n_profiles = x_test.shape[0]

    for x_i in range(n_profiles):
        logger.info("Validation of %s/%s profile", x_i, n_profiles)

        data = x_test[x_i]
        data = np.expand_dims(data, axis=0)
        heatmap, pred = make_gradcam_heatmap(data, model, layer_index)
        n_hm = heatmap.shape[0]
        x_hm = np.arange(0, n_hm, n_hm / profile_len)
        xp_hm = np.arange(n_hm)
        heatmap_interp = np.interp(x_hm, xp_hm, heatmap)
        heatmaps_output[x_i, :] = heatmap_interp

    del model
    hm_path = os.path.join(paths.output, f"heatmap.npy")
    np.save(hm_path, heatmaps_output)
    logger.info("Heatmap has been saved... %s", hm_path)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config-fname",
        required=True,
        dest="config_fname",
        type=str,
        metavar="FILENAME",
        help="Path to file with configuration",
    )

    parser.add_argument(
        "-x",
        "--profiles",
        required=True,
        dest="profiles_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        dest="model",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        dest="output",
        type=str,
        metavar="FILENAME",
        help="Path to output directory",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.33
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    input_options = parse_args()
    data_settings = DatasetConfiguration(input_options.config_fname)

    process(data_settings, input_options)
