import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from data import load_data

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class CVAE(keras.Model):
    def __init__(self, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.latent_dim = 2
        self.input_size = 2
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def build_encoder(self):    
        encoder_inputs = keras.Input(shape=(self.input_size+1, ))
        encoder_hidden = layers.Dense(16, activation="relu")(encoder_inputs)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(encoder_hidden)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(encoder_hidden)
        z = Sampling()([z_mean, z_log_var])
        label = tf.expand_dims(encoder_inputs[:, 2], axis=-1)
        z_cond = keras.layers.concatenate([z, label], axis=1)
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z_cond], name="encoder")
        encoder.summary()
        return encoder

    def build_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim+1,))
        decoder_hidden = layers.Dense(16, activation="relu")(latent_inputs)
        decoder_outputs = layers.Dense(self.input_size, activation="sigmoid")(decoder_hidden)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()
        return decoder

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z_cond = self.encoder(data)
            reconstruction = self.decoder(z_cond)
            reconstruction = tf.expand_dims(reconstruction, axis=-1)
            x = data[:, 0:2]
            reconstruction_loss = keras.losses.binary_crossentropy(x, reconstruction)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss, axis=-1))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def plot_label_clusters(cvae, data):
    z_mean, _, _ = cvae.encoder.predict(data)
    labels = data[:, 2]
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.savefig('cvae_gen/label_clusters.png')
    # plt.show()

def plot_predict(cvae):
    sample_z = np.random.randn(1000, 2)
    sample_c = np.full([40, 1], 0)
    for c in range(1, 25):
        new_sample_c = np.full([40, 1], c)
        sample_c = np.concatenate([sample_c, new_sample_c], axis=0)
    sample_zc = np.concatenate([sample_z, sample_c], axis=1)
    output = cvae.decoder.predict(sample_zc)
    plt.scatter(output[:, 0], output[:, 1], c=sample_c)
    plt.savefig('cvae_gen/predict.png')
    # plt.show()

def train_cvae():
    print("Loading data for cVAE training...")
    train_data = load_data('data/train.txt')
    train_data = np.expand_dims(train_data, axis=-1) / 50
    print("Successfully loaded data.")

    cvae = CVAE()
    cvae.compile(optimizer=keras.optimizers.Adam())
    tb_callback = tf.keras.callbacks.TensorBoard('./cvae_gen', update_freq=1)
    cvae.fit(train_data, shuffle=True, epochs=40, batch_size=128, callbacks=[tb_callback])

    print("Generating...")
    plot_label_clusters(cvae, train_data)
    plot_predict(cvae)
    print("Samples have been generated and saved to cvae_gen/.")

train_cvae()
