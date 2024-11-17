import argparse
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from dataset_configuration import DatasetConfiguration

from cnn_models.simple_model import simple_model

C_LOGGER_NAME = "explanation"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def make_saliency_map(profile, model, pred_index=None):
    """

    """

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    profile_t = tf.convert_to_tensor(profile)

    # --------

    with tf.GradientTape() as tape:
        tape.watch(profile_t)
        predictions = model(profile_t)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        target_score = predictions[:, pred_index]
    gradients = tape.gradient(target_score, profile_t)
    saliency = tf.abs(gradients).numpy()

    return saliency, pred_index


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

    heatmaps_output = np.zeros(x_test.shape[:-1])
    n_profiles = x_test.shape[0]

    for x_i in range(n_profiles):
        logger.info("Validation of %s/%s profile", x_i, n_profiles)

        data = x_test[x_i]
        data = np.expand_dims(data, axis=0)
        heatmap, pred = make_saliency_map(data, model)
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
