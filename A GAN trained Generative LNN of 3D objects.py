import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

# Load and preprocess the dataset
def load_data(path):
    voxel_data = np.load(path)
    voxel_data = voxel_data.astype('float32') / 255.0
    return voxel_data

# Define the Generator model
def build_generator():
    model = keras.Sequential()
    model.add(layers.Dense(128 * 4 * 4 * 4, activation="relu", input_dim=100))
    model.add(layers.Reshape((4, 4, 4, 128)))
    model.add(layers.Conv3DTranspose(128, (4, 4, 4), strides=(2, 2, 2), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv3DTranspose(64, (4, 4, 4), strides=(2, 2, 2), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv3DTranspose(1, (4, 4, 4), activation="sigmoid", padding="same"))
    return model

# Define the Discriminator model
def build_discriminator():
    model = keras.Sequential()
    model.add(layers.Conv3D(64, (4, 4, 4), strides=(2, 2, 2), padding="same", input_shape=(32, 32, 32, 1)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv3D(128, (4, 4, 4), strides=(2, 2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Loss functions
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)

# Training step function
@tf.function
def train_step(generator, discriminator, real_voxels):
    noise = tf.random.normal([batch_size, 100])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_voxels = generator(noise, training=True)
        
        real_output = discriminator(real_voxels, training=True)
        fake_output = discriminator(generated_voxels, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Training function
def train(generator, discriminator, dataset, epochs, batch_size):
    for epoch in range(epochs):
        for real_voxels in dataset:
            train_step(generator, discriminator, real_voxels)
        print(f'Epoch {epoch + 1}/{epochs} completed')

# Visualization function
def plot_voxel(voxel_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_data, edgecolor='k')
    plt.show()

# Main function
if __name__ == "__main__":
    # Load the dataset
    voxel_data = load_data('path_to_voxel_dataset.npy')
    batch_size = 32
    buffer_size = 60000

    # Prepare the dataset
    dataset = tf.data.Dataset.from_tensor_slices(voxel_data).shuffle(buffer_size).batch(batch_size)

    # Build the models
    generator = build_generator()
    discriminator = build_discriminator()

    # Train the models
    epochs = 50
    train(generator, discriminator, dataset, epochs, batch_size)

    # Generate and visualize a 3D object
    noise = tf.random.normal([1, 100])
    generated_voxel = generator(noise, training=False)
    plot_voxel(generated_voxel.numpy().reshape((32, 32, 32)))
