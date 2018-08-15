'''Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name)#, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap('RdBu', np.max(y_test)-np.min(y_test)+1)
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap=cmap)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    #plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    #digit_size = 28
    x_size = 23 #18
    y_size = 24 #12
    figure = np.zeros((x_size * n, y_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-10, 10, n)
    grid_y = np.linspace(-10, 10, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(x_size, y_size)
            figure[i * x_size: (i + 1) * x_size,
                   j * y_size: (j + 1) * y_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = x_size // 2
    end_range = n * x_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, x_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    #plt.show()


# load the signature dataset
if not os.path.exists('ATLAS_LC/full_data.h5'):
    df_mira = pd.read_csv('ATLAS_LC/MIRA_features_table.csv')#[signature_cols]
    df_mpulse = pd.read_csv('ATLAS_LC/MPULSE_features_table.csv')#[signature_cols]
    df_dbf = pd.read_csv('ATLAS_LC/DBF_features_table.csv')#[signature_cols]
    df_lpv = pd.read_csv('ATLAS_LC/LPV_features_table.csv')#[signature_cols]
    df_dbh = pd.read_csv('ATLAS_LC/DBH_features_table.csv')#[signature_cols]
    df_pulse = pd.read_csv('ATLAS_LC/PULSE_features_table.csv')#[signature_cols]
    df_nsine = pd.read_csv('ATLAS_LC/NSINE_features_table.csv')#[signature_cols]
    df_sine = pd.read_csv('ATLAS_LC/SINE_features_table.csv')#[signature_cols]
    df_msine = pd.read_csv('ATLAS_LC/MSINE_features_table.csv')#[signature_cols]
    df_cbh = pd.read_csv('ATLAS_LC/CBH_features_table.csv')#[signature_cols]
    df_cbf = pd.read_csv('ATLAS_LC/CBF_features_table.csv')#[signature_cols]
    df_irr = pd.read_csv('ATLAS_LC/IRR_features_table.csv')#[signature_cols]

    full_data = pd.concat([df_mira, df_mpulse, df_dbf, df_lpv, df_dbh, df_pulse, 
                           df_nsine, df_sine, df_msine, df_cbf, df_cbh], sort=False)
    del(df_mira)
    del(df_dbf)
    del(df_lpv)
    del(df_dbh)
    del(df_pulse)
    del(df_nsine)
    del(df_sine)
    del(df_msine)
    del(df_cbh)
    del(df_cbf)
    del(df_irr)
    
    signature_cols = [col for col in full_data.columns if 'Signature' in col]
    signature_cols += ['OBJID', 'filter', 'CLASS']
    
    full_data[signature_cols].to_hdf('ATLAS_LC/full_data.h5', key='sign', mode='w')
    
    dmdt_cols = [col for col in full_data.columns if 'Deltam' in col]
    dmdt_cols += ['OBJID', 'filter', 'CLASS']
    
    full_data[dmdt_cols].to_hdf('ATLAS_LC/full_data.h5', key='dmdt', mode='w')
else:
    full_data = pd.read_hdf('ATLAS_LC/full_data.h5', key='dmdt')

#full_data = full_data.sample(100000)
signature_cols = [col for col in full_data.columns if 'Deltam' in col]
X = full_data[signature_cols].values
Y = pd.Categorical(full_data['CLASS'].values).codes

#data = X.reshape(X.shape[0], 18, 12)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.

#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)

# MNIST dataset
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size #* image_size
#x_train = np.reshape(x_train, [-1, original_dim])
#x_test = np.reshape(x_test, [-1, original_dim])
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 2048
latent_dim = 2
epochs = 100

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_sign_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_sign_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp_sign')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp_dmdt.png',
               show_shapes=True)

    if args.weights:
        vae = vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_mlp_dmdt.h5')

    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp_dmdt")
    
