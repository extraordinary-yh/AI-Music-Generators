#!/usr/bin/env python
# coding: utf-8

# # Audio Data Augmentation & Generation with VAE - 1st Pipeline 

# ## 1. Data

# For the 1st pipeline, we will perform more operations on the dataset that we used on the first pipeline. As a refresher, below is the summarized description on my dataset.

# In the Piano dataset, I selected 30 tracks that come from at least 8 different pianos over the past 10 years. The Human dataset contains 30 tracks that is mainly human voice. 
# 
# Audio files in both datasets has varying length from 10 seconds to 10min. The files are also recorded in varying conditions. All audio files are converted to .wav format, and the two dataset are saved separately in two directories, Data/Piano and Data/Human.
# 
# Below is two selected examples of the audio data, one from Piano and one from Human. Piano29.wav is Chopin's Nocturne Op32 No.2, one of my favorite nocturnes; Human26.wav is a group of ladies singing that I recorded in Argentina. Enjoy!"

# In[4]:


#Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa
import os


# In[5]:


#Nocturne Op32 No.2
ipd.Audio('Data/Piano/Piano29.wav') 


# ## 2. Data Structure 

# In the second pipeline, we will still use the Librosa library for audio data processing. 
# 
# We use Librosa.load() function to load .wav file into a time series array, whose elements represents the amplitude of the audio signal at a time point. 

# In[6]:


audio_data, sample_rate = librosa.load('Data/Piano/Piano29.wav')
print("audio data:", audio_data)
print("sample rate:", sample_rate)


# In our pipeline, we're also going to base our operations on Mel spectrograms. Mel spectrograms are a visual representation of the audio signal in a certain period of time. The spectrogram plots time on the x-axis, frequency on the y-axis, and color intensity representing the power(loudness) of each frequency bin. With the Mel spectrogram, we could conduct many operations on the data, such as classification, data augmentation, and apply generative model. The Mel spectrogram will be presented in the following section. 

# ## 3. Data Processing Add On (Data Augmentation)

# Upon the feedback for the first pipeline, I treid out data augmentation on audio data. The following code implements shifting the audio data's pitch by one octave up. The augmentation is conducted by: defining the steps (or notes) we want to shift up, converting it to frequency, and shift the frequency using Librosa's pitch_shift function. We then generate the Mel spectrogram of original track and shifted track to visually see the result. 

# In[7]:


# Load the audio file and extract the Mel spectrogram
y, sr = librosa.load('Data/Piano/Piano29.wav')

# Shift pitch up by one octave
n_steps = 8

# Convert the amount of pitch shift from semitones to frequency ratio
pitch_shift = 2 ** (n_steps / 12)

# Shift the pitch of the audio signal
y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps, bins_per_octave=12)


# In[8]:


# Compute the Mel spectrogram of the original audio
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=512, n_fft=2048, fmin=30, power=1.0)

# Compute the Mel spectrogram of the shifted audio
mel_spec_shifted = librosa.feature.melspectrogram(y=y_shifted, sr=sr, n_mels=128, fmax=8000, hop_length=512, n_fft=2048, fmin=30, power=1.0)

# Convert the Mel spectrogram to decibels
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
mel_spec_shifted_db = librosa.power_to_db(mel_spec_shifted, ref=np.max)


# In[9]:


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

# ## 4&5. Discussing Tasks & Model Selection (Intro of VAE)

# I am implementing a variational autoencoder to generate new piano music based on the 30 data files I compiled. 
# 
# Similar to an autoencoder we introduced in class, they have mostly the same structure, with encoder and decoder to conduct dimentionality reduction and data reconstruction. However, there are two key differences that make VAE "variational". 
# 
# The first big difference is the way we encode data in the latent space. Rather than encoding the data as pointsin the latent space, the VAE encodes the data as a continuous **probability distribution**, with a mean and log variance 
# 
# The second big difference is the way we decode data. Rather than directly feeding the data point into the decoder, we take advantage of our distribution structure and **sample** the points with reparameterization trick. 
# 
# Another small difference is the loss function. In addition to the reconstruction loss function we use in AE, VAE adds the Kullback-Leibler (KL) divergence loss, which measures how divergent is the learned latent distribution compared to a standard normla distribution. 
# 
# More detailed explaination on these differences are included beside the code implementation below. Now, on with the code!
# 

# ## 6. Train the Model 

# In[10]:


import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


# First, I'll explain the constants. 
# 
# n_mels is essentially the nnumber of Mel frequency bands we use for the Mel spectrogram. Using 128 Mel frequency bands is common, because it reduces the dimention of the data while retaining the most important parts. 
# 
# 
# time_frames: the number of time frames (columns) to be used for each Mel spectrogram. We use this to make sure input size for the VAE model is consistent. 
# 
# 
# latent_dim: the number of dimension of the latent space inside the VAE model. As latent dim goes smaller, it is more compressed representation of the input.
# 
# batch_size: the number of audio samples that are processed after weight update.  
# 
# epochs: the number of times the VAE model will iterate. 

# In[58]:


# Set the constants
n_mels = 128
time_frames = 216
latent_dim = 16
batch_size = 10
epochs = 100

# Load and process audio data
def load_audio_data(audio_files):
    audio_data = []

    for file in audio_files:
        audio, sampling_rate = librosa.load(file)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=n_mels)
        S_dB = librosa.power_to_db(mel_spectrogram)
        
        # If the time_frames is greater than the actual frames, pad with zeros
        if S_dB.shape[1] < time_frames:
            pad_width = time_frames - S_dB.shape[1]
            S_dB = np.pad(S_dB, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # If the time_frames is less than the actual frames, truncate the extra frames
        if S_dB.shape[1] > time_frames:
            S_dB = S_dB[:, :time_frames]
        
        #Normalize the data 
        S_dB = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB))
        audio_data.append(S_dB)
        
    return np.array(audio_data)

# Load audio data
directory = 'Data/Piano'
audio_files = []

#create the audiofiles array
for file in os.listdir(directory):
    audio_files.append(os.path.join(directory, file))
    
audio_data = load_audio_data(audio_files)

time_frames = audio_data.shape[2]


# We do data loading and preprossesing in the function load_audio_data(), so that our input for VAE is standardized and able to process. 
# 
# First, we load the all 30 of our piano data. Then we compute the Mel-spectrogram features from .wav using librosa melspectrogram function. 
# 
# Then, we pad the data. We check if the number of frames in the computed Mel-spectrogram is less than or greater than 216 (our decided time frame). If it's less, the function pads the Mel-spectrogram with zeros; if it's greater, it truncates the extra frames. 
# 
# Finally, the function normalizes the Mel-spectrogram features using min-max normalization. By subtracting the minimum value and dividing by the range, we scale them between 0 and 1. 
# 
# The processed data is all appended on audio_data. 

# In[60]:


# Encoder
encoder_input = Input(shape=(n_mels, time_frames))
encoder_flatten = Flatten()(encoder_input)
z_mean = Dense(latent_dim)(encoder_flatten)
z_log_var = Dense(latent_dim)(encoder_flatten)


# The above is the four layers of encoder. The first layer is input layer, where we put our mel_spectrogram data into the VAE. The flatten layer flattens the mel_spectrogram data into  a one-dimensional tensor. 
# Then, we have two Dense layers, where we calculate the mean and the variance of the input dat, which generates the latent space distribution.

# In[61]:


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = Model(encoder_input, [z_mean, z_log_var, z], name="encoder")


# The sampling function is where we perform the reparameteriztaion trick. 
# 
# First, we use a latent variable z using the reparameterization trick, using the mean and variance of the distirbution. Then, we extract the batch size and dimension of the latent space from the shape of z_mean. Epsilon is generated from a standard normal distribution to introduce stochasticity in the sampling process.
# 
# The sampled z is calculated using the reparameterization trick formula (z_mean + K.exp(0.5 * z_log_var) * epsilon). 
# 
# Then, we use the Lambda layer to apply the sampling function to the [z_mean, z_log_var] tensor list, resulting in the generation of the sampled z. 

# In[62]:


decoder_input = Input(shape=(latent_dim,))
decoder_dense = Dense(n_mels * time_frames, activation="sigmoid")(decoder_input)
decoder_reshaped = Reshape((n_mels, time_frames))(decoder_dense)
decoder = Model(decoder_input, decoder_reshaped, name="decoder")


# Here, we build our decoder. 
# 
# The first layer is the input layer, feeding the sampled latent variables z into the the decoder. 
# 
# Then, in the Dence layer, we map our latent variables to a higher dimention. We use a sigmoid function so the values fit the mel spectrogram format.
# 
# The Reshape layer further maps the data into the original dimention of mel-spectrogram. 

# In[63]:


# Define the VAE loss function
def vae_loss_function(args):
    y_true, y_pred, z_mean, z_log_var = args
    reconstruction_loss = K.mean(K.square(y_true - y_pred))
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
    return reconstruction_loss + kl_loss


# As mentioned above, the overall loss function consists of the reconstruction loss and the KL divergence loss. 
# 
# Similar to autoencoders, the reconstructoin loss is the mean squared error between the true mel spectrogram (y_true) and the reconstructed mel spectrogram (y_pred). 
# 
# Then, we have VAE specific KL-loss. It calculates the KL divergence between the approximate posterior distribution q(z|x) and the prior distribution p(z). Here, we assume p(z) as a standard normal distribution (mean 0, variance 1).
# 
# Mathematically, the KL loss is givven by:
# 
# KL(q(z|x) || p(z)) = -0.5 * ∑(1 + log(σ²) - μ² - σ²)
# 
# The μ and σ² represent the mean and variance of q(z|x). 

# In[64]:


# Create the VAE
vae_input = Input(shape=(n_mels, time_frames))
vae_target = Input(shape=(n_mels, time_frames))
encoder_output = encoder(vae_input)
decoder_output = decoder(encoder_output[2])


# In[65]:


# Add the loss function to the model using a Lambda layer
loss = Lambda(vae_loss_function, output_shape=(1,), name='loss')([vae_target, decoder_output, encoder_output[0], encoder_output[1]])
vae = Model(inputs=[vae_input, vae_target], outputs=[decoder_output, loss])

# Compile the VAE
vae.compile(optimizer=Adam(learning_rate=0.001), loss=['mse', None])

# Train the VAE
vae.fit([audio_data, audio_data], [audio_data, np.zeros_like(audio_data)], epochs=epochs, batch_size=batch_size)


# In[66]:


vae.summary()


# ## 7. Performance Metrics

# In the training output, we see that the loss in the final epoch is around 0.0077, where it started from 0.1364 in epoch 1. Purely looking at the loss numbers, it looks great! 
# 
# We also calculated the MSE error in the following cell, which is also around 0.0077, which means that most of the error is contributed by reconstruction rather than KL divergence.
# 
# However, we need to really evaluate the quality by looking at the output. Here are the original Mel-spectrogram and reconstructed Mel-spectrogram. 

# In[71]:


# Compute the reconstructed data
reconstructions, _ = vae.predict([audio_data, audio_data])

# Calculate the reconstruction error (Mean Squared Error)
mse = np.mean((audio_data - reconstructions) ** 2)
print(f"Reconstruction error (MSE): {mse}")

# Visualize the original and reconstructed spectrograms
def plot_spectrogram_comparison(index):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the original spectrogram
    axes[0].imshow(audio_data[index], origin='lower', aspect='auto', cmap='viridis')
    axes[0].set_title("Original Spectrogram")
    
    # Plot the reconstructed spectrogram
    axes[1].imshow(reconstructions[index], origin='lower', aspect='auto', cmap='viridis')
    axes[1].set_title("Reconstructed Spectrogram")
    
    plt.show()

# Plot the comparison for all indexes
for songs in range(30):
    plot_spectrogram_comparison(songs)


# Comparing the two mel-spectrograms, we can see that the original one has much more clearer bars and patterns, while the reconstructed spectrogram looks like the data is all over the place. This means that the reconstructed audio output is more bizzare and needs further improvement. 
