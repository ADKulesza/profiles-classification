from tensorflow.keras.layers import (AvgPool1D, BatchNormalization, Conv1D,
                                     Dense, Dropout, Flatten, MaxPool1D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.regularizers import l2


def flatten_conv_model(n_class, profile_length, learning_rate):
    model = Sequential()

    model.add(
        Conv1D(
            256,
            (4,),
            activation="relu",
            bias_regularizer=l2(0.00001),
            kernel_initializer="he_uniform",
            padding="same",
            input_shape=(profile_length, 1),
        )
    )
    model.add(
        Conv1D(
            256,
            (2,),
            activation="relu",
            bias_regularizer=l2(0.00001),
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))
    model.add(AvgPool1D((2,)))

    model.add(
        Conv1D(
            128,
            (4,),
            activation="relu",
            bias_regularizer=l2(0.00001),
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    model.add(
        Conv1D(
            128,
            (2,),
            activation="relu",
            bias_regularizer=l2(0.00001),
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))
    model.add(MaxPool1D((2,)))

    model.add(
        Conv1D(
            64,
            (4,),
            activation="relu",
            bias_regularizer=l2(0.00001),
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    model.add(
        Conv1D(
            64,
            (2,),
            activation="relu",
            bias_regularizer=l2(0.00001),
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))
    model.add(MaxPool1D((2,)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(2 * n_class, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(n_class, activation="softmax"))

    opt = Nadam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model
