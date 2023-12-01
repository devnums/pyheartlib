# Architecture of the example model.
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPool1D,
)


def print_layer(layer):
    """Prints layer output dim and its name or class name"""
    l_out_shape = tf.keras.backend.shape(layer)._inferred_value
    l_name = layer.name
    print(
        "\nLayer: {} -->  Output shape: {}".format(
            str(l_name).upper(), str(l_out_shape)
        )
    )


def reg():
    return tf.keras.regularizers.l2(l=0.01)


def conv1d_block(
    inp,
    name=None,
    filters=64,
    kernel_size=64,
    bn=True,
    drate=0.30,
    pool_size=0,
    flatten=True,
    regularizer=None,
):
    output = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )(inp)
    if bn:
        output = BatchNormalization(axis=-1)(output)
    output = Activation("relu")(output)
    if drate > 0:
        output = Dropout(drate)(output)
    if pool_size > 0:
        output = MaxPool1D(pool_size=pool_size)(output)
    if flatten:
        output = Flatten()(output)

    print_layer(output)
    return output


def model_arch(params_model):
    x_input_dim = int(params_model["x_input_dim"])
    num_classes = int(params_model["num_classes"])
    input1_layer = Input(shape=(None, x_input_dim), name="x_input_dim")
    out = tf.gather(input1_layer, tf.constant([0]), axis=1)
    out = tf.squeeze(input=out, axis=1)
    out = tf.expand_dims(out, axis=-1)
    out = BatchNormalization()(out)
    print_layer(out)
    out = conv1d_block(
        out,
        name="block1",
        filters=16,
        kernel_size=6,
        bn=True,
        drate=0.2,
        pool_size=6,
        flatten=False,
    )
    out = Activation("relu")(out)
    out = Bidirectional(LSTM(32, return_sequences=True))(out)
    out = Dropout(0.2)(out)
    print_layer(out)
    out = Bidirectional(LSTM(32, return_sequences=True))(out)
    out = Dropout(0.2)(out)
    print_layer(out)
    out = Dense(64, activation="relu")(out)
    out = Dropout(0.2)(out)
    out = Dense(num_classes, activation="softmax")(out)
    print_layer(out)
    return tf.keras.Model(inputs=input1_layer, outputs=out, name="model_rpeak")
