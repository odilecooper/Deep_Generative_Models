import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from data import load_data
class ConditionalGAN(keras.Model):
    def __init__(self):
        super(ConditionalGAN, self).__init__()
        self.latent_dim = 2
        self.batch_size = 64
        self.num_channels = 2
        self.latent_dim = 2
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        self.generator_in_channels = self.latent_dim + 1
        self.discriminator_in_channels = self.num_channels + 1
        # print(self.generator_in_channels, self.discriminator_in_channels)

        self.discriminator = keras.Sequential([
            keras.layers.InputLayer((self.discriminator_in_channels, )),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ], name="discriminator")

        self.generator = keras.Sequential([
            keras.layers.InputLayer((self.generator_in_channels, )),
            layers.Dense(16, activation="relu"),
            layers.Dense(2, activation="sigmoid"),
        ], name="generator")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker, self.total_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        real_samples, labels = data
        labels = tf.expand_dims(labels, axis=1)

        batch_size = tf.shape(real_samples)[0]
        random_latent_vectors = tf.squeeze(tf.random.normal(shape=(batch_size, self.latent_dim)))
        random_vector_labels = tf.concat([random_latent_vectors, labels], axis=1)

        generated_samples = self.generator(random_vector_labels)

        fake_and_labels = tf.concat([generated_samples, labels], -1)
        real_and_labels = tf.concat([real_samples, labels], -1)
        combined_samples = tf.concat([fake_and_labels, real_and_labels], axis=0)
        combined_labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_samples)
            d_loss = self.loss_fn(combined_labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat([random_latent_vectors, labels], axis=1)

        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            fake_samples = self.generator(random_vector_labels)
            fake_and_labels = tf.concat([fake_samples, labels], -1)
            predictions = self.discriminator(fake_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        total_loss = d_loss + g_loss
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        self.total_loss_tracker.update_state(total_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result()
        }

def plot_predict(cgan):
    sample_z = np.random.randn(1000, 2)
    sample_c = np.full([40, 1], 0)
    for c in range(1, 25):
        new_sample_c = np.full([40, 1], c)
        sample_c = np.concatenate([sample_c, new_sample_c], axis=0)
    sample_zc = np.concatenate([sample_z, sample_c], axis=1)
    output = cgan.generator.predict(sample_zc)
    plt.scatter(output[:, 0], output[:, 1], c=sample_c)
    plt.savefig('cgan_gen/predict.png')
    # plt.show()

def train_cgan():
    print("Loading data for cGAN training...")
    train_data = load_data('data/train.txt')
    all_digits = train_data[:, 0:2] / 50
    all_labels = train_data[:, 2]
    print("Successfully loaded data.")

    # print(f"Shape of training samples: {all_digits.shape}")
    # print(f"Shape of training labels: {all_labels.shape}")

    cgan = ConditionalGAN()
    cgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    tb_callback = tf.keras.callbacks.TensorBoard('./cgan_gen', update_freq=1)
    cgan.fit(all_digits, all_labels, epochs=40, callbacks=[tb_callback])

    print("Generating...")
    plot_predict(cgan)
    print("Samples have been generated and saved to cgan_gen/.")

train_cgan()
