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
    conv_layer = layers.MaxPool1D((2,))(conv_layer)
    conv_layer = layers.Dropout(0.2)(conv_layer)
    return conv_layer


def simple_model(n_class, profile_length, learning_rate):
    inputs = Input(shape=(profile_length, 1))


    conv = conv_block(inputs, 64, 3, input_shape=(profile_length, 1))
    conv = conv_block(conv, 128, 3)

    # conv = layers.Dropout(0.1)(conv)
    conv = layers.Flatten()(conv)

    conv = layers.Dense(
        256, activation="relu", kernel_initializer="he_uniform"
    )(conv)

    conv = layers.Dense(n_class, activation="softmax")(conv)

    # Define the model
    model = Model(inputs, conv)

    decay_steps = 512

    cosine_decay_scheduler = CosineDecay(
        initial_learning_rate=learning_rate, decay_steps=decay_steps, alpha=0.5
    )

    opt = Adam(learning_rate=cosine_decay_scheduler)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy", Recall(), Precision()],
    )

    return model
