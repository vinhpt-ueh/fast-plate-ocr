"""
Model definitions for the FastLP OCR.
"""

from typing import Literal
import kimm

from keras.activations import softmax
from keras.layers import (
    Activation,
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Reshape,
    Softmax,
)
from keras.models import Model

from fast_plate_ocr.train.model.layer_blocks import (
    block_no_activation,
)


def cnn_ocr_model(
    h: int,
    w: int,
    max_plate_slots: int,
    vocabulary_size: int,
    dense: bool = True,
    activation: str = "relu",
    pool_layer: Literal["avg", "max"] = "max",
) -> Model:
    input_tensor = Input((h, w, 1))  # Define the input tensor
    backbone = kimm.models.MobileViTV2W050(
        input_tensor=input_tensor,
        include_top=False,
        weights=None,
    )
    backbone_output = backbone.output
    x = (
        head(backbone_output, max_plate_slots, vocabulary_size)
        if dense
        else head_no_fc(backbone_output, max_plate_slots, vocabulary_size)
    )
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
