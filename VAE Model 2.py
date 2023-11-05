#!/usr/bin/env python
# coding: utf-8

# # Audio Data Augmentation & Generation with VAEs - 2nd Pipeline

# ## 1. Data

# For the 3rd pipeline, we will perform more operations on the dataset that we used on the first pipeline. As a refresher, below is the summarized description on my dataset from assignment one. 

# In the Piano dataset, I selected 30 tracks that come from at least 8 different pianos over the past 10 years. The Human dataset contains 30 tracks that is mainly human voice. 
# 
# Audio files in both datasets has varying length from 10 seconds to 10min. The files are also recorded in varying conditions. All audio files are converted to .wav format, and the two dataset are saved separately in two directories, Data/Piano and Data/Human.
# 
# Below is two selected examples of the audio data, one from Piano and one from Human. Piano29.wav is Chopin's Nocturne Op32 No.2, one of my favorite nocturnes; Human26.wav is a group of ladies singing that I recorded in Argentina. Enjoy!"



#Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa
import os




#Nocturne Op32 No.2
ipd.Audio('Data/Piano/Piano29.wav') 


# ## 2. Data Structure 

# In the second pipeline, we will still use the Librosa library for audio data processing. 
# 
# As mentioned in assignment 1, we use Librosa.load() function to load .wav file into a time series array, whose elements represents the amplitude of the audio signal at a time point. 



audio_data, sample_rate = librosa.load('Data/Piano/Piano29.wav')
print("audio data:", audio_data)
print("sample rate:", sample_rate)


# In our pipeline, we're also going to base our operations on Mel spectrograms. Mel spectrograms are a visual representation of the audio signal in a certain period of time. The spectrogram plots time on the x-axis, frequency on the y-axis, and color intensity representing the power(loudness) of each frequency bin. With the Mel spectrogram, we could conduct many operations on the data, such as classification, data augmentation, and apply generative model. The Mel spectrogram will be presented in the following section. 

# ## 3. Data Processing Add On (Data Augmentation)

# Upon the feedback for the first pipeline, I treid out data augmentation on audio data. The following code implements shifting the audio data's pitch by one octave up. The augmentation is conducted by: defining the steps (or notes) we want to shift up, converting it to frequency, and shift the frequency using Librosa's pitch_shift function. We then generate the Mel spectrogram of original track and shifted track to visually see the result. 



# Load the audio file and extract the Mel spectrogram
y, sr = librosa.load('Data/Piano/Piano29.wav')

# Shift pitch up by one octave
n_steps = 8

# Convert the amount of pitch shift from semitones to frequency ratio
pitch_shift = 2 ** (n_steps / 12)

# Shift the pitch of the audio signal
y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps, bins_per_octave=12)




# Compute the Mel spectrogram of the original audio
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=512, n_fft=2048, fmin=30, power=1.0)

# Compute the Mel spectrogram of the shifted audio
mel_spec_shifted = librosa.feature.melspectrogram(y=y_shifted, sr=sr, n_mels=128, fmax=8000, hop_length=512, n_fft=2048, fmin=30, power=1.0)

# Convert the Mel spectrogram to decibels
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
mel_spec_shifted_db = librosa.power_to_db(mel_spec_shifted, ref=np.max)




# Plot the original and shifted Mel spectrograms
librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, hop_length=512)
plt.colorbar(format='%+2.0f dB')
plt.title('Original Mel spectrogram')
plt.show()

librosa.display.specshow(mel_spec_shifted_db, x_axis='time', y_axis='mel', sr=sr, hop_length=512)
plt.colorbar(format='%+2.0f dB')
plt.title('Shifted Mel spectrogram')
plt.show()


# From the two spectrograms, we can see that, the shitfed Mel spectrogram shows the same data pattern as the Original Mel spectrogram horizontally, but it shifts the frequency upwards. If we stack the two audio tracks together, we will get an octave chord. We can produce harmoney by applying data augmentation on the same track with different scale (such as moving the note upward by three notes and five notes to produce a major third chord). 

# ## Train the Model.
# 
# 

# In the below model, we upgraded our VAE by adding convolution layers to capture more details of the mel-spectrogram instead of using the fully connected layers like above. 
# 
# Since we're using Mel-spectrogram as input, CNN VAEs should perform better. Each Mel-Spectrogram is convolved with a set of kernels that scan the image and produce a set of feature maps that highlight the presence of specific patterns in the image. These feature maps can then be passed through additional convolutional layers to identify more complex patterns or combined with pooling layers to reduce the dimensionality of the feature maps(Shakflat, 2018). Therefore, they are able to exploit the translation invariance property of images, which means that the same pattern can appear in different locations in an image. By using shared weights across the convolutional filters, convolutional layers can identify the same pattern regardless of where it appears in the image. In this case, the pattern that appeared in the Mel-Spectrogram is basically same pieces of audio(could be melody or chord) that appeared again and again. Therefore, the CNN VAE should capture the reapting audio pattern better. 
:


from tensorflow.keras.layers import Conv2D, Conv2DTranspose 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

:


#define sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#encoder
def encoder(input_shape, latent_dim):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(128, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(256, 3, padding='same', activation='relu', strides=(2, 2))(x)
    shape_before_flattening = K.int_shape(x)[1:]
    x = Flatten()(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder, shape_before_flattening

#decoder
def decoder(input_shape, latent_dim):
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(np.prod(input_shape))(latent_inputs)
    x = Reshape(input_shape)(x)
    x = Conv2DTranspose(256, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2DTranspose(128, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2DTranspose(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2DTranspose(32, 3, padding='same', activation='relu')(x)
    outputs = Conv2DTranspose(1, 3, padding='same', activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder

:


# Reshape audio data to include a channel dimension
audio_data_reshaped = audio_data.reshape(audio_data.shape[0], audio_data.shape[1], audio_data.shape[2], 1)

# Split audio_data into training and validation sets
train_data = audio_data_reshaped[:-10]
val_data = audio_data_reshaped[-10:]

input_shape = (n_mels, time_frames, 1)

# Instantiate the encoder and decoder models
enc_model, shape_before_flattening = encoder(input_shape, latent_dim)
dec_model = decoder(shape_before_flattening, latent_dim)

# Define the VAE loss function
def vae_loss_function(args):
    y_true, y_pred, z_mean, z_log_var = args
    reconstruction_loss = K.mean(K.square(y_true - y_pred))
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
    return reconstruction_loss + kl_loss

# Build the VAE model
inputs = Input(shape=input_shape)
vae_target = Input(shape=input_shape)
z_mean, z_log_var, z = enc_model(inputs)
decoder_output = dec_model(z)
loss = Lambda(vae_loss_function, output_shape=(1,), name='loss')([vae_target, decoder_output, z_mean, z_log_var])
cnn_vae = Model(inputs=[inputs, vae_target], outputs=[decoder_output, loss], name='vae')

# Add the VAE loss to the model
cnn_vae.add_loss(loss)

# Compile the model
cnn_vae.compile(optimizer='adam')

# Train the model
cnn_vae.fit([train_data, train_data], epochs=50, batch_size=batch_size, validation_data=([val_data, val_data], None))


# ## Performance Evaluation.
:


cnn_vae.summary()


# Compared to the fully connected VAE, our CNN VAE has almost six times the amount of parameters. Let's see the output of the model. 
:


# Reshape audio data to include a channel dimension
audio_data_reshaped = audio_data.reshape(audio_data.shape[0], audio_data.shape[1], audio_data.shape[2], 1)

# Compute the reconstructed data
reconstructions, _ = cnn_vae.predict([audio_data_reshaped, audio_data_reshaped])

# Reshape the output to match the original audio_data shape
reconstructions = reconstructions.reshape(audio_data.shape)

# Calculate the reconstruction error (Mean Squared Error)
mse = np.mean((audio_data - reconstructions) ** 2)
print(f"Reconstruction error (MSE): {mse}")

# Visualize the original and reconstructed spectrograms
def plot_spectrogram_comparison(index):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the original spectrogram
    librosa.display.specshow(audio_data[index], x_axis='time', y_axis='mel', ax=axes[0])
    axes[0].set_title("Original Spectrogram")
    
    # Plot the reconstructed spectrogram
    librosa.display.specshow(reconstructions[index], x_axis='time', y_axis='mel', ax=axes[1])
    axes[1].set_title("Reconstructed Spectrogram")
    
    plt.show()

# Plot the comparison for all indexes
num_spectrograms = audio_data.shape[0]
for index in range(num_spectrograms):
    plot_spectrogram_comparison(index)


# Unfortunately, from the generated mel_spectrogram we see that our CNN VAE model performs worse than than the fully connected VAE. The generated Mel_Spectrogram is way too fuzzy. From the loss functon, we already see that the reconstruction loss is much higher than the fully connected model. 
# 

# ## References
# Rocca, J. (2019). Understanding Variational Autoencoders (VAEs). Towards Data Science.
# 
# Shafkat, I. (2018, June 1). Intuitively Understanding Convolutions for Deep Learning. Towards Data Science.

# ## HC
# 
# \#algorithms: Througout the semester, I learned ML models and algorithms & implemneted them into code.
