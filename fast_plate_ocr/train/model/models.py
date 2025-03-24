"""
Model definitions for the FastLP OCR.
"""

from typing import Literal

from keras.src.activations import softmax
from keras.src.layers import (
    Activation,
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Rescaling,
    Reshape,
    Softmax,
    AveragePooling2D,
    Conv2D
)
from keras.src.models import Model

from fast_plate_ocr.train.model.layer_blocks import (
    block_average_conv_down,
    block_bn,
    block_max_conv_down,
    block_no_activation,
)


def cnn_ocr_model_improved(
    h: int,
    w: int,
    max_plate_slots: int,
    vocabulary_size: int,
    activation: str = "relu",
    pool_layer: Literal["avg", "max"] = "max",
) -> Model:
    """
    Improved OCR model for license plate recognition with a spatial head to better distinguish similar characters.
    
    Args:
        h (int): Input height.
        w (int): Input width.
        max_plate_slots (int): Maximum number of characters in the license plate.
        vocabulary_size (int): Number of unique characters in the vocabulary.
        activation (str): Activation function for convolutional layers (default: "relu").
        pool_layer (Literal["avg", "max"]): Type of pooling layer (default: "max").
    
    Returns:
        Model: Compiled Keras model with spatial head.
    """
    input_tensor = Input((h, w, 1))
    x = Rescaling(1.0 / 255)(input_tensor)

    # Select pooling-convolution block
    if pool_layer == "avg":
        block_pool_conv = block_average_conv_down
    elif pool_layer == "max":
        block_pool_conv = block_max_conv_down

    # Backbone (unchanged)
    x = block_pool_conv(x, n_c=32, padding="same", activation=activation)
    x, _ = block_bn(x, k=3, n_c=64, s=1, padding="same", activation=activation)
    x, _ = block_bn(x, k=1, n_c=64, s=1, padding="same", activation=activation)
    x = block_pool_conv(x, n_c=64, padding="same", activation=activation)
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding="same", activation=activation)
    x, _ = block_bn(x, k=1, n_c=128, s=1, padding="same", activation=activation)
    x = block_pool_conv(x, n_c=128, padding="same", activation=activation)
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding="same", activation=activation)
    x, _ = block_bn(x, k=1, n_c=256, s=1, padding="same", activation=activation)
    x = block_pool_conv(x, n_c=256, padding="same", activation=activation)
    x, _ = block_bn(x, k=1, n_c=512, s=1, padding="same", activation=activation)
    x, _ = block_bn(x, k=1, n_c=1024, s=1, padding="same", activation=activation)

    # Spatial Head: Adjust spatial dimensions and predict per slot
    # After backbone: x has shape (batch_size, h/16, w/16, 1024)
    # e.g., for h=32, w=224: (batch_size, 2, 14, 1024)
    # Pool to (1, max_plate_slots), e.g., (1, 7)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # Shape: (batch_size, 1, max_plate_slots, 1024)
    
    x = Conv2D(filters=vocabulary_size, kernel_size=1)(x)
    # Shape: (batch_size, 1, max_plate_slots, vocabulary_size)
    
    x = Reshape((max_plate_slots, vocabulary_size))(x)
    # Shape: (batch_size, max_plate_slots, vocabulary_size)
    
    x = Activation('softmax')(x)
    # Softmax over vocabulary_size dimension for per-slot probabilities

    return Model(inputs=input_tensor, outputs=x)


def head(x, max_plate_slots: int, vocabulary_size: int):
    """
    Model's head with Fully Connected (FC) layers.
    """
    x = GlobalAveragePooling2D()(x)
    # dropout for more robust learning
    x = Dropout(0.5)(x)
    dense_outputs = [
        Activation(softmax)(Dense(units=vocabulary_size)(x)) for _ in range(max_plate_slots)
    ]
    # concat all the dense outputs
    x = Concatenate()(dense_outputs)
    return x


def head_no_fc(x, max_plate_slots: int, vocabulary_size: int):
    """
    Model's head without Fully Connected (FC) layers.
    """
    x = block_no_activation(x, k=1, n_c=max_plate_slots * vocabulary_size, s=1, padding="same")
    x = GlobalAveragePooling2D()(x)
    x = Reshape((max_plate_slots, vocabulary_size, 1))(x)
    x = Softmax(axis=-2)(x)
    return x
