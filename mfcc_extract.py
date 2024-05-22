#extract mfcc features from the audio files from the recordings folder and output to the features folder

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def extract_mfcc_features(audio_file):
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return mfcc


recordings_folder = 'recordingss'

features_folder = 'features'

if not os.path.exists(features_folder):
    os.makedirs(features_folder)

for audio_file in os.listdir(recordings_folder):
    if not audio_file.endswith('.m4a'):
        continue
    # Extract the MFCC features
    mfcc = extract_mfcc_features(os.path.join(recordings_folder, audio_file))
    # Save the MFCC features to a file
    np.save(os.path.join(features_folder, audio_file.replace('.wav', '.npy')), mfcc)

    # Plot the MFCC features
    plt.figure()
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()
#print numpoy array features from the features folder
for feature_file in os.listdir(features_folder):
    if not feature_file.endswith('.npy'):
        continue
    mfcc = np.load(os.path.join(features_folder, feature_file))
    # print(mfcc)
    # print(mfcc.shape)