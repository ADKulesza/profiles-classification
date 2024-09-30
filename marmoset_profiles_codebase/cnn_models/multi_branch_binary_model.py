from tensorflow import expand_dims
from tensorflow.experimental import numpy
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.regularizers import l2


def conv_block(conv_layer, output_shape, kernel_size, input_shape=None):
    if input_shape:
        conv_layer = layers.Conv1D(
            output_shape,
            (kernel_size,),
            activation="relu",
            bias_regularizer=l2(0.00001),
            kernel_initializer="he_uniform",
            padding="same",
            input_shape=input_shape,
        )(conv_layer)
    else:
        conv_layer = layers.Conv1D(
            output_shape,
            (kernel_size,),
            activation="relu",
            bias_regularizer=l2(0.00001),
            kernel_initializer="he_uniform",
            padding="same",
        )(conv_layer)

    conv_layer = layers.Normalization(axis=1)(conv_layer)
    conv_layer = layers.AvgPool1D((2,))(conv_layer)
    conv_layer = layers.Dropout(0.2)(conv_layer)
    return conv_layer


def multi_branch_binary_model(n_class, profile_length, learning_rate):
    inputs = Input(shape=(profile_length, 1))

    # ---
    diff_layer = layers.Concatenate(axis=1)(
        [inputs, expand_dims(inputs[:, 0, :], axis=-1)]
    )
    diff_layer = layers.Lambda(lambda x: numpy.diff(x, axis=1))(diff_layer)

    diff_layer = conv_block(diff_layer, 256, 3, input_shape=(profile_length, 1))
    diff_layer = conv_block(diff_layer, 128, 3)
    diff_layer = conv_block(diff_layer, 64, 3)

    # ---
    conv = conv_block(inputs, 256, 3, input_shape=(profile_length, 1))
    conv = conv_block(conv, 128, 3)
    conv = conv_block(conv, 64, 3)

    conv = layers.Concatenate(axis=-1)([conv, diff_layer])
    conv = layers.BatchNormalization(axis=-1)(conv)

    conv = layers.Conv1D(
        8,
        (2,),
        activation="relu",
        bias_regularizer=l2(0.00001),
        kernel_initializer="he_uniform",
        padding="same",
    )(conv)
    conv = layers.Conv1D(
        n_class,
        (8,),
        activation="relu",
        bias_regularizer=l2(0.00001),
        kernel_initializer="he_uniform",
        padding="same",
    )(conv)
    conv = layers.Normalization(axis=1)(conv)

    conv = layers.Flatten()(conv)
    conv = layers.Dropout(0.2)(conv)
    conv = layers.Dense(
        2 * n_class, activation="relu", kernel_initializer="he_uniform"
    )(conv)
    conv = layers.Dense(n_class, activation="softmax")(conv)

    # Define the model
    model = Model(inputs, conv)

    #
    decay_steps = 2048

    cosine_decay_scheduler = CosineDecay(
        initial_learning_rate=learning_rate, decay_steps=decay_steps, alpha=0.5
    )

    opt = Adam(learning_rate=cosine_decay_scheduler)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=["accuracy", Recall(), Precision()],
    )

    return model
