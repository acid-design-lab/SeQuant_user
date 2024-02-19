import tensorflow as tf
from keras.layers import (
    Input,
    Dense,
    Conv2D,
    AveragePooling2D,
    BatchNormalization,
    Flatten,
    Reshape,
    Conv2DTranspose,
    LeakyReLU,
    Activation
)
from keras.models import Model
from keras.engine.keras_tensor import KerasTensor


def encoder(x: KerasTensor, height: int) -> tf.Tensor:
    x = Conv2D(121, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = AveragePooling2D((1, 3))(x)

    x = Conv2D(121, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = AveragePooling2D((1, 2))(x)

    x = Conv2D(height, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = AveragePooling2D((1, 2))(x)

    x = Conv2D(height, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = AveragePooling2D((1, 2))(x)

    x = Conv2D(1, (1, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = AveragePooling2D((1, 2))(x)
    return x


def latent_space(x: KerasTensor, latent_dim: int) -> tf.Tensor:
    x = Flatten()(x)
    units = x.shape[1]
    x = Dense(units)(x)
    x = Dense(latent_dim, name='Latent')(x)
    x = Dense(units)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((latent_dim, 1, 1))(x)
    return x


def decoder(x: KerasTensor, height: int) -> tf.Tensor:
    x = Conv2DTranspose(1, (1, 4), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(height, (1, 4), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(height, (1, 4), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(121, (1, 4), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(121, (1, 4), strides=(1, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    return x


def autoencoder_model(height: int, width: int, channels: int, latent_dim: int, learning_rate: float) -> tf.keras.Model:
    inputs = Input(shape=(height, width, channels))
    x = inputs
    x = encoder(x, height)
    x = latent_space(x, latent_dim)
    output = decoder(x, height)
    autoencoder = Model(inputs, output, name='Conv_Autoencoder')
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    return autoencoder
