from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense,
                                     Dropout, Flatten, MaxPooling1D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def simple_model(n_class, profile_length, learning_rate):
    # logger.info('Train simple model...\n')
    model = Sequential()

    model.add(
        Conv1D(
            64,
            (3,),
            activation="relu",
            kernel_initializer="he_uniform",
            input_shape=(profile_length, 1),
        )
    )
    model.add(MaxPooling1D((2,)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))

    model.add(
        Conv1D(
            32, (3,), activation="relu", kernel_initializer="he_uniform", padding="same"
        )
    )
    model.add(MaxPooling1D((2,)))
    model.add(BatchNormalization(axis=-1))

    model.add(Flatten())
    model.add(Dense(n_class * 4, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dropout(0.2))
    model.add(Dense(n_class, activation="softmax"))

    opt = Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model
