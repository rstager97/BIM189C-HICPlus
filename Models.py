from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Add, Conv2D, MaxPooling2D, Flatten
#from tensorflow.keras.layers.convolutional import Conv2D
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def HICplus_Default(input_size = 40):
  conv2d1_filters_numbers = 8
  conv2d1_filters_size = 9
  conv2d2_filters_numbers = 8
  conv2d2_filters_size = 1
  conv2d3_filters_numbers = 1
  conv2d3_filters_size = 5

  input = Input(shape=(input_size,input_size, 1, ))
  conv1 = Conv2D(conv2d1_filters_numbers, kernel_size=conv2d1_filters_size, activation='relu', padding='same')(input)
  conv2 = Conv2D(conv2d2_filters_numbers, kernel_size=conv2d2_filters_size, activation='relu', padding='same')(conv1)
  conv3 = Conv2D(conv2d3_filters_numbers, kernel_size=conv2d3_filters_size, activation='relu', padding='same')(conv2)
  model = Model(inputs=input, outputs=conv3)
  print(model.summary())
  plot_model(model, to_file='HICplus_Default_network.png')

  return model

def LapSRN(input_size = 40):
  input = Input(shape=(input_size,input_size, 1, ))

  #splitting
  HalfSize = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (int(input_size/2), int(input_size/2)), method=tf.image.ResizeMethod.BILINEAR))(input)
  QuarterSize = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (int(input_size/4), int(input_size/4)), method=tf.image.ResizeMethod.BILINEAR))(input)

  # Full Size
  conv1Full = Conv2D(8, kernel_size=(3,3), activation='relu', padding='same')(input) # padding needed to keep shape
  conv2Full = Conv2D(8, kernel_size=(3,3), activation='relu', padding='same')(conv1Full)
  conv3Full = Conv2D(8, kernel_size=(3,3), activation='relu', padding='same')(conv2Full)

  # Half Size
  conv1Half = Conv2D(8, kernel_size=(3,3), activation='relu')(HalfSize)
  conv2Half = Conv2D(8, kernel_size=(3,3), activation='relu')(conv1Half)
  conv3Half = Conv2D(8, kernel_size=(3,3), activation='relu')(conv2Half)

  # Quarter Size
  conv1Quarter = Conv2D(8, kernel_size=(3,3), activation='relu')(QuarterSize)
  conv2Quarter = Conv2D(8, kernel_size=(3,3), activation='relu')(conv1Quarter)
  conv3Quarter= Conv2D(8, kernel_size=(3,3), activation='relu')(conv2Quarter)

  #Readd and convolve
  HalfUpscaled = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (input_size, input_size), method=tf.image.ResizeMethod.BILINEAR))(conv3Half)
  QuarterUpscaled = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (input_size, input_size), method=tf.image.ResizeMethod.BILINEAR))(conv3Quarter)

  Added = Add()([conv3Full, HalfUpscaled, QuarterUpscaled])
  conv4 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(Added)
  output = Conv2D(1, kernel_size=(1, 1), activation='linear', padding='same')(conv4)

  model = Model(inputs=input, outputs=output)
  print(model.summary())
  plot_model(model, to_file='HICplus_LapSRN_network.png')

  return model

def Discriminator(input_size = 40):
  input = Input(shape=(input_size,input_size, 1, ))
  conv1= Conv2D(16, kernel_size=(3,3), activation='relu',)(input) # padding needed to keep shape
  MaxPool1 = MaxPooling2D(pool_size=(3, 3), strides=(1,1))(conv1)
  Conv2 = Conv2D(1, kernel_size=(3,3), activation='relu',)(MaxPool1) # padding needed to keep shape
  MaxPool2 = MaxPooling2D(pool_size=(3, 3), strides=(1,1))(Conv2)
  Flat = Flatten()(MaxPool2)
  D1 = Dense(32, activation='relu')(Flat)
  output = Dense(1, activation='sigmoid')(D1)

  model = Model(inputs=input, outputs=output)
  print(model.summary())
  plot_model(model, to_file='HICplus_Descriminator_network.png')

  return model

# define a composite gan model for the generator and discriminator
def define_gan(generator, discriminator):
  discriminator.trainable = False
  model = Sequential()
  model.add(generator)
  model.add(discriminator)
  model.compile(loss='binary_crossentropy', optimizer='adam')
  return model

def TrainGAN(generator, discriminator, datasetX, datasetY, n_epochs, n_batch, Fname = 'Gan.h5'):
  batches_per_epoch = int(len(datasetX) / n_batch)
  n_steps = batches_per_epoch * n_epochs
  gan_model = define_gan(generator, discriminator)
  plt.ion()
  # gan training algorithm
  for i in range(n_steps):
    # generate points in the latent space
    ix = np.random.randint(0, len(datasetY), n_batch)
    z = datasetX[ix,:,:,:]
    # generate fake images
    fake = generator.predict(z)
    # select a batch of random real images
    ix = np.random.randint(0, len(datasetY), n_batch)
    # retrieve real images
    real = datasetY[ix,:,:,:]
    # update weights of the discriminator models
    y_fake = np.zeros((n_batch, 1))
    discriminator_loss = discriminator.train_on_batch(fake, y_fake)
    y_real = np.ones((n_batch, 1))
    # update the discriminator for real images
    discriminator_loss += discriminator.train_on_batch(real, y_real)/2

    # generate points in the latent space
    ix = np.random.randint(0, len(datasetY), n_batch)
    z = datasetX[ix,:,:,:]
    # define target labels for real images
    y_real = np.ones((n_batch, 1))
    # update generator model
    generator_loss = gan_model.train_on_batch(z, y_real)

    # tracking data
    print("epoch:%s  discriminator loss:%s    generator loss:%s     " % (
      i, discriminator_loss, generator_loss))
    plt.scatter(i, generator_loss, c="b")
    plt.scatter(i, discriminator_loss, c="r")
    plt.pause(0.05)

    if i%batches_per_epoch == 0:
      gan_model.save_weights(Fname)





