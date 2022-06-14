from tensorflow import keras
from tensorflow.keras import layers

# from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# import imageio

from data import load_data

'''
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

# Scale the pixel values to [0, 1] range, add a channel dimension to
# the images, and one-hot encode the labels.
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
all_labels = keras.utils.to_categorical(all_labels, 10)

# Create tf.data.Dataset.
dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
'''

latent_dim = 2

train_data = load_data('data/train.txt')
train_data = train_data / 50
# train_data = np.expand_dims(train_data, axis=-1) / 50
all_digits = train_data[:, 0:2]
all_labels = train_data[:, 2]

print(f"Shape of training samples: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")


class ConditionalGAN(keras.Model):
    def __init__(self):
        super(ConditionalGAN, self).__init__()
        self.latent_dim = 2
        self.batch_size = 64
        self.num_channels = 2
        self.latent_dim = 2
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

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
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        real_samples, labels = data
        labels = tf.expand_dims(labels, axis=1)
        # labels = tf.cast(labels, dtype=tf.int32)

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
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

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat([random_latent_vectors, labels], axis=1)

        # Assemble labels that say "all real samples".
        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            fake_samples = self.generator(random_vector_labels)
            fake_and_labels = tf.concat([fake_samples, labels], -1)
            predictions = self.discriminator(fake_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
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
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig('cgan_gen/predict.png')
    # plt.show()

def train_cgan():
    cgan = ConditionalGAN()
    cgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    cgan.fit(all_digits, all_labels, epochs=20)

    plot_predict(cgan)


train_cgan()
