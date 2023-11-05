#!/usr/bin/env python
# coding: utf-8

# ## Classifying Piano Recordings from Human Voices - 1st Pipeline

# ## 1. Data

# In the Piano dataset, I selected 30 tracks that come from at least 8 different pianos over the past 10 years, and the sounds are quite different. Some recordings are even taken by pointing my phone to a headphone, when I was playing Toronto public library's keyboard. The Human dataset is messier: generally it contains 30 tracks that is mainly human voice. It includes tracks of me talking, me and my friends talking, me singing, my friends singsing, my friend singing, talking and laughing after he's drunk, and against all sorts of background noise. 
# 
# What makes things even messier is that I didn't do any pre-processing on these data to make it more standardized. Rather, both dataset have very short recrdings of a few seconds to very long recordings of 10 minutes. Their volumes vary drastically, ranging from almost a flat line on the sound wave graph to maxing out the graph from start to finish. But as prof once said, reality is messy, so let's keep it that way :) However, in order to process files better, all audio files are converted to .wav format, and the two dataset are saved separately in two directories, Data/Piano and Data/Human.
# 
# Below is two selected examples of the audio data, one from Piano and one from Human. Piano29.wav is Chopin's Nocturne Op32 No.2, one of my favorite nocturnes; Human26.wav is a group of ladies singing that I recorded in Argentina. Enjoy!



#Import packages
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import IPython.display as ipd
import librosa
import os


# In[15]:


#Nocturne Op32 No.2
ipd.Audio('Data/Piano/Piano29.wav') 


# In[16]:


#Singing in restaurant
ipd.Audio('Data/Human/Human26.wav') 


# ## 2. Data Structure 

# To read and process the the audio files into Python, we used the librosa library to load all the .wav files. The Librosa.load() function returns two values for each audio file: the sampled audio data, and the sample rate of the audio.
# 
# The sampled audio data is represented in a 1d NumPy array, where each element of the array represents the amplitude of the audio signal at a time point. Essentially, we take discrete points in the continuous audio, and  sample the amplitude for future processing. The sample rate is essentialy the number of samples taken per second. 
# 
# Below is a example of what librosa.load() returns.

# In[19]:


audio_data, sample_rate = librosa.load('Data/Piano/Piano29.wav')
print("audio data:", audio_data)
print("sample rate:", sample_rate)


# ## 3. Data Processing & Analysis

# Having the audio_data and sample_rate, we can extract the features and components from the audio file. To do this, we calculate the Mel-frequency cepstral coefficients (MFCCs) from the audio files by using the librosa.feature.mfcc() function.
# 
# Essentially, MFCCs are used to identify the content and discarding noise. We use MFCCs here because they can capture important characteristics of the audio signal such as timbre and pitch, and they are robust to changes in recording conditions, which makes it very applicable to my dataset.

# MFCCs are representation of the spectral shape of an audio signal. They are obtained by analyzing the short-term power spectrum of the signal. (Jameslyons, 2013). 
# 
# The over all process are broken down into a few steps:
# 
# 1. We divide the audio into 20-40ms short frames. Like calculus, we assume in a short frame the audio signal don't change much. 
# 2. We calculate the power spectrum of each frame. The power spectrum essentially tells us what is the frequency of each frame.
# 3. Then, we calculate the Mel Bands of each frame. The Mel Bank uses the Mel scale, which converts frequency number to a scale that mimics what human perceives. Since Humans are more sensitive to changes in lower frequencies and are not good at detecting high frequencies the Mel scale. The transofrmation formula is: $m=2595log_{10}(1+\frac{f}{100})$. 
# 4. The mel scale is divided into a set of triangular overlapping bands, and the power spectrum of each frame is then weighted by the filterbank coefficients corresponding to these bands. The log of the weighted sum of each band is then taken. 
# 5. We transform the rest of coefficients with the discrete cosine transform (DCT) to obtain a set of coefficients that capture the most important spectral features of the audio signal.
# 
# Here's an example code that shows the WFCC outputs.


mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
mfccs


#Plot MFCCs (Librosa, 2023)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.power_to_db(mfccs, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel spectrogram')
ax[0].label_outer()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')


# Combining Part 2 and Part 3, we use the following code process all of our audio files. We store the features into the features array, and label the data. 


#Extract features from audio files
features = []
labels = []

#List the file data
file_types = ['Piano', 'Human']

#Process each file type
for file_type in file_types:
    #create directory path 
    type_dir = os.path.join('Data/', file_type)
    
    for file in os.listdir(type_dir):
        # Load audio file with the Librosa library.  
        audio_data, sample_rate = librosa.load(os.path.join(type_dir, file))

        # Extract Mel Frequency Cepstral Coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
        features.append(np.mean(mfccs, axis=1))

        # Assign labels to audio files
        labels.append(file_type)


# ## 4. Discussing Tasks

# Since we're trying to classify the audio into piano recordings and human voice recordings, the task we're doing is classifications. To conduct supervised machine learning, we need to split our dataset into training dataset and testing dataset. 
# 
# We use the train_test_split() function from the sklearn library. It will randomly split our dataset into training and testing set, with our testing size to be 20% of total data, specified by our 'test_size' parameter. The 'features' parameter are the MFCC coefficients extracted from the audio files. X_train contains the feature used for training, y_train include the target labels of the training set; and X_test and y_test are the data used for testing. 

# In[100]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)


# ## 5. Model Selection

# Here, I used the random forest classifier to process our data because i's a commonly used machine learning algorithm for classification tasks. Specifically, it performs well for datasets with a large number of features like audio data (Mbaabu, 2020).
# 
# We were introduced to decision tree in previous sessions. A decision tree would produce a output by following a path till the leaf node, which produces a result. Instead of just one decision tree, random forest works by building multiple decision trees on random subsets of the data and features. Then it combines the predictions of decision tress to make a final prediction by choosing the result with the most vote. By combining multiple results, it helps to reduce overfitting and improve the model's ability to generalize.

# Each single decision tree classifier in sklearn uses an impurity function to determine the best split parameter. The process is similar to what we learned in preclass readings: the subspace will be continuously partitioned, and each split's impurity is calculated, and the minimum impurity item is chosen (Scikit learn, 2023). 
# 
# The decision tree calculate the feature importance by the neagtive value of node impurity weighted by the probability of reaching that node. The node probability is given by the number of samples that reach the node over the total number of samples (Ronaghan, 2018). As the node probability increases, the feature importance increases. As the Gini impurity decreases, the feature importance increases. The below formula expresses the feature importance (Ronaghan, 2018).

# ![Feature%20Importance%20Function.png](attachment:Feature%20Importance%20Function.png)

# In[101]:


# Set up random forest classifier
rfc = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=10)


# In the code above, we use the RandomForestClassifier from sklearn to implement the model. 
# 
# The n_estimator parameter specifies the number of decision trees used in the forest, which I set to be 100.
# 
# The max_depth parameter limits the decision tree's depth to help mitigate overfitting of the data. 
# 
# The random_state parameter determine the seed state that the classifier use to generate the random forest. It is set to a 10 so that the algorithm's result could be reproduced when other conditions of the algorithm is not changed. There is actually no reason that it has to be 10, any arbitrary value would ensure that the result could be reproduced. 

# ## 6. Train Model

# Let's train the model with X_train and y_train data! X_train contains piano files, while y_train contains human voice files.

# In[102]:


# Train the random forest classifier
rfc.fit(X_train, y_train)


# ## 7. Predictions & Performance Metric

# We first make a prediction with the testing dataset.

# In[103]:


# Predict labels for test data
y_pred = clf.predict(X_test)


# Here, we will use three main metrics to test the model: accuracy, precision and recall. 
# 1. Accuracy measures the percentage of correctly classified files out of all the instances.
# 2. Precision measures the percentage of true positives of all predicted positives. It reflects how accurate are the positive predictions. 
# 3. Recall measures the proportion of true positives out of all actual positives. It measures how many positive cases are identified.
# 
# Let's see how they perform.


# Evaluate model accuracy, precision and recall.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred,pos_label='Piano')
recall = recall_score(y_test, y_pred, pos_label='Piano')


# Print results
print("Accuracy: {:.2f}%".format(accuracy*100))
print("Precision: {:.2f}%".format(precision*100))
print("Recall: {:.2f}%".format(recall*100))


# Oh no! The model classfies the results perfectly! It might mean that our data is overfitting. Let's use cross validation to see if the model is actually so bad-ass. 

# In[106]:


# Perform cross-validation using 10 folds
cv_results = cross_validate(rfc, features, labels, cv=10, scoring=('accuracy', 'precision', 'recall', 'f1'))

# Print the average results of the cross-validation
print("Cross-validation results:")
print("Accuracy:", np.mean(cv_results['test_accuracy']))
print("Precision:", np.mean(cv_results['test_precision']))
print("Recall:", np.mean(cv_results['test_recall']))
print("F1:", np.mean(cv_results['test_f1']))


# ## 8. Visualize


import seaborn as sns
from sklearn.metrics import confusion_matrix

# Fit the model on the entire dataset
rfc.fit(features, labels)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["Human", "Piano"])

# Plot confusion matrix
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["Human", "Piano"], yticklabels=["Human", "Piano"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# ## 10. References

# Jameslyons. (2013). Mel Frequency Cepstral Coefficient (MFCC) tutorial. Retrieved from
# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
# 
# "Visualize the MFCC series". (2023). librosa.feature.mfcc. Retrieved from
# https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
# 
# Onesmus Mbaabu. (2020). Introduction to Random Forest in Machine Learning. Retrieved from https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/
# 
# Scikit Learn. (2023). Decision Trees - Mathematical formulation. Retrieved from 
# https://scikit-learn.org/stable/modules/tree.html#tree
# 
# Stacey Ronaghan. (2018). The Mathematics of Decision Trees, Random Forest and Feature Importance in Scikit-learn and Spark. Retrieved from https://towardsdatascience.com/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3




