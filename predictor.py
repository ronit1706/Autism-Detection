import os
import numpy as np
import librosa
import joblib
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

#check is model.pkl exists
if not os.path.exists('model.pkl'):
    import model


# Load saved model
model = joblib.load('model.pkl')

# Directory containing test audio files
test_folder = "test_records"

# Function to extract MFCC features from an audio file and calculate row-wise averages
def extract_mfcc(audio_file, max_frames):
    audio_data, sample_rate = librosa.load(audio_file)
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    # print the plot for mfcc
    plt.figure()
    librosa.display.specshow(mfcc_features, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()



    # Calculate row-wise average
    mfcc_avg = np.mean(mfcc_features, axis=1, keepdims=True)
    # Pad or truncate to match the maximum number of frames
    if mfcc_avg.shape[0] < max_frames:
        pad_width = ((0, max_frames - mfcc_avg.shape[0]), (0, 0))
        mfcc_avg = np.pad(mfcc_avg, pad_width, mode='constant')
    elif mfcc_avg.shape[0] > max_frames:
        mfcc_avg = mfcc_avg[:max_frames, :]
    return mfcc_avg


max_frames = 4

# Choose an audio file for prediction
audio_file = os.path.join(test_folder, "non_joshi_2.m4a=-`")

# Extract MFCC features from the audio file
mfcc_features = extract_mfcc(audio_file, max_frames)

# Reshape data to have two dimensions
mfcc_features_reshaped = mfcc_features.reshape(1, -1)

# Make prediction
predicted_label = model.predict(mfcc_features_reshaped)

# Map predicted label to human-readable format
prediction_map = {1: "Autistic", 0: "Non-autistic"}
predicted_class = prediction_map[predicted_label[0]]

print("Predicted class:", predicted_class)
