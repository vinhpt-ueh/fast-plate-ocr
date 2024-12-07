import keras
from keras import layers, models, ops

# ruff: noqa
# pylint: skip-file
# type: ignore


class SpatialTransformer(layers.Layer):
    def __init__(self, localization_net, output_size=None, **kwargs):
        super().__init__(**kwargs)
        self.localization_net = localization_net
        self.output_size = output_size

    def call(self, inputs):
        # Get input shape
        batch_size = ops.shape(inputs)[0]
        height = ops.shape(inputs)[1]
        width = ops.shape(inputs)[2]

        # Predict transformation parameters theta from the localization network
        theta = self.localization_net(inputs)
        theta = ops.reshape(theta, [batch_size, 2, 3])  # Assuming affine transformation

        # Generate grid
        if self.output_size is None:
            out_height, out_width = height, width
        else:
            out_height, out_width = self.output_size

        grid = self._affine_grid_generator(theta, out_height, out_width)
        x_transformed = self._bilinear_sampler(inputs, grid)

        return x_transformed

    def _affine_grid_generator(self, theta, out_height, out_width):
        batch_size = ops.shape(theta)[0]

        # Create normalized grid coordinates
        x = ops.linspace(-1.0, 1.0, out_width)
        y = ops.linspace(-1.0, 1.0, out_height)
        x_t, y_t = ops.meshgrid(x, y)
        ones = ops.ones_like(x_t)
        sampling_grid = ops.stack([x_t, y_t, ones], axis=2)  # Shape: [H, W, 3]
        sampling_grid = ops.expand_dims(sampling_grid, axis=0)  # Shape: [1, H, W, 3]
        sampling_grid = ops.broadcast_to(sampling_grid, [batch_size, out_height, out_width, 3])

        # Transform the sampling grid using theta
        sampling_grid = ops.reshape(sampling_grid, [batch_size, out_height * out_width, 3])
        theta = ops.cast(theta, "float32")
        sampling_grid = ops.cast(sampling_grid, "float32")
        grid_transformed = ops.matmul(sampling_grid, ops.transpose(theta, [0, 2, 1]))
        grid_transformed = ops.reshape(grid_transformed, [batch_size, out_height, out_width, 2])

        return grid_transformed

    def _bilinear_sampler(self, img, grid):
        height = ops.shape(img)[1]
        width = ops.shape(img)[2]

        x = grid[..., 0]
        y = grid[..., 1]

        # Scale grid to pixel coordinates
        max_y = ops.cast(height - 1, "float32")
        max_x = ops.cast(width - 1, "float32")
        x = ((x + 1.0) * 0.5) * max_x
        y = ((y + 1.0) * 0.5) * max_y

        x0 = ops.floor(x)
        y0 = ops.floor(y)
        x1 = x0 + 1
        y1 = y0 + 1

        x0 = ops.clip(x0, 0, max_x)
        x1 = ops.clip(x1, 0, max_x)
        y0 = ops.clip(y0, 0, max_y)
        y1 = ops.clip(y1, 0, max_y)

        # Calculate interpolation weights
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # Expand dimensions for broadcasting
        wa = ops.expand_dims(wa, axis=-1)
        wb = ops.expand_dims(wb, axis=-1)
        wc = ops.expand_dims(wc, axis=-1)
        wd = ops.expand_dims(wd, axis=-1)

        # Gather pixel values from image
        Ia = self._get_pixel_value(img, x0, y0)
        Ib = self._get_pixel_value(img, x0, y1)
        Ic = self._get_pixel_value(img, x1, y0)
        Id = self._get_pixel_value(img, x1, y1)

        # Compute output
        out = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return out

    def _get_pixel_value(self, img, x, y):
        batch_size = ops.shape(img)[0]
        height = ops.shape(img)[1]
        width = ops.shape(img)[2]
        channels = ops.shape(img)[3]

        x = ops.cast(x, "int32")
        y = ops.cast(y, "int32")

        # Clip indices to be within the boundaries of the image
        x = ops.clip(x, 0, width - 1)
        y = ops.clip(y, 0, height - 1)

        # Since advanced indexing is not available in keras.ops,
        # we need to flatten the indices and use ops.take
        batch_offsets = ops.reshape(ops.arange(batch_size) * height * width, (batch_size, 1, 1))
        y_offsets = y * width
        indices = batch_offsets + y_offsets + x  # Shape: [batch_size, out_height, out_width]

        # Flatten the image and indices
        flat_img = ops.reshape(img, [batch_size * height * width, channels])
        flat_indices = ops.reshape(indices, [-1])

        # Gather pixel values
        pixel_values = ops.take(flat_img, flat_indices, axis=0)
        pixel_values = ops.reshape(
            pixel_values, [batch_size, ops.shape(x)[1], ops.shape(x)[2], channels]
        )

        return pixel_values

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "localization_net": keras.layers.serialize(self.localization_net),
                "output_size": self.output_size,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["localization_net"] = keras.layers.deserialize(config["localization_net"])
        return cls(**config)


# Define the localization network
def create_localization_net(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(8, kernel_size=7, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(10, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    # Initialize the transformation parameters to identity
    initializer = keras.initializers.Zeros()
    theta = layers.Dense(6, activation="linear", kernel_initializer=initializer)(x)
    model = models.Model(inputs, theta)
    return model
