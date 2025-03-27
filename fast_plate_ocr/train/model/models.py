"""
Model definitions for the FastLP OCR.
"""

import kimm
import kimm.models
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
    Conv2D,
)
from keras.models import Model


from fast_plate_ocr.train.model.layer_blocks import (
    block_no_activation,
)
from fast_plate_ocr.train.model.stn import SpatialTransformer, create_localization_net


def cnn_ocr_model(
    h: int,
    w: int,
    max_plate_slots: int,
    vocabulary_size: int,
    dense: bool = True,
    use_stn: bool = False,
) -> Model:
    input_tensor = Input((h, w, 1))
    if use_stn:
        print('use_stn')
        localization_net = create_localization_net((h, w, 1))
        stn_layer = SpatialTransformer(localization_net=localization_net, output_size=(h, w))
        backbone_input = stn_layer(input_tensor)
        backbone_base = kimm.models.MobileViTV2W050(
            include_top=False,
            weights=None,
            input_shape=[h, w, 1],
        )
        backbone_output = backbone_base(backbone_input)
    else:
        print('not use_stn')
        backbone = kimm.models.MobileViTV2W050(
            input_tensor=input_tensor,
            include_top=False,
            weights=None,
        )
        #freeze backbone
        # backbone.trainable = False
        # x = backbone(input_tensor, training=False)
        backbone_output = backbone.output

    x = (
        head(backbone_output, max_plate_slots, vocabulary_size)
        if dense
        else head_no_fc(backbone_output, max_plate_slots, vocabulary_size)
    )
    model = Model(inputs=input_tensor, outputs=x)
    model.summary(show_trainable=True)
    return model


def head(x, max_plate_slots: int, vocabulary_size: int):
    """
    Model's head with Fully Connected (FC) layers.
    """
    # x = Conv2D(256, (3, 3), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    # dropout for more robust learning
    # x = Dense(256, activation='relu')(x)
    x = Dropout(0.6)(x)
    # x = Reshape((max_plate_slots, vocabulary_size, 1))(x)
    # x = Conv2D(256, kernel_size=1, padding="same", activation="relu")(x)
    
    # conv_layers = [
    #     Activation(x)(Conv2D(128, kernel_size=1, padding="same")(x)) for _ in range(max_plate_slots)
    # ]
    dense_outputs = [
        Activation(softmax)(Dense(units=vocabulary_size)(Dense(units=vocabulary_size)(x))) for _ in range(max_plate_slots)
    ]
    # concat all the dense outputs
    # x = Concatenate()(conv_layers)
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
