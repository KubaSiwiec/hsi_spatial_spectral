import tensorflow as tf
from tensorflow import keras

# model definition function
def create_model(l2_loss_lambda = None, original_dim = (3, 3, 103)):
    keras.backend.clear_session()

    target_size = (32, 32)
    l2 = None if l2_loss_lambda is None else keras.regularizers.l2(l2_loss_lambda)
    if l2 is not None:
        print('Using L2 regularization - l2_loss_lambda = %.4f' % l2_loss_lambda)

    model = keras.Sequential(
        [
            keras.layers.Lambda(lambda image: tf.image.resize(image, target_size)),
            keras.layers.Conv2D(256, 3, activation=tf.nn.relu, input_shape=(32, 32, 103)),
            keras.layers.MaxPool2D(2),
            keras.layers.Conv2D(512, 3, activation=tf.nn.relu),
            keras.layers.MaxPool2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=l2),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation=tf.nn.softmax),
        ]
    )

    # model compiling

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model