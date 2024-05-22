import streamlit as st
import joblib
import numpy as np
import librosa
import warnings
from pydub import AudioSegment
import io
warnings.filterwarnings('ignore', category=UserWarning)

models = {
    'rf.pkl': 'Random Forest (~90% accuracy)',
    'ann.pkl': 'Artificial Neural Network (~72% accuracy)',
    'svm.pkl': 'Support Vector Machine (~54% accuracy)',
    'nb.pkl': 'Naive Bayes (~81% accuracy)',
}

st.title("Autism Detection")

# Choose a model (show the name of the model, but load the corresponding file)
model = st.selectbox("Choose a model", list(models.values()))
chosen_model = [k for k, v in models.items() if v == model][0]


model = joblib.load(chosen_model)


uploaded_file = st.file_uploader("Choose an audio file...", type="m4a")

# Define CSS styles for big text
yes_style = '<h1 style="color:red;text-align:center;">Prediction: Autistic</h1>'
no_style = '<h1 style="color:green;text-align:center;">Prediction: Non Autistic</h1>'


if uploaded_file:
    audio = AudioSegment.from_file(io.BytesIO(uploaded_file.getvalue()), format='m4a')
    samples = audio.get_array_of_samples()
    y = np.array(samples).astype(np.float32) / (2**15 - 1)  # Convert to floating-point
    sr = audio.frame_rate
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr)

    if not np.isnan(mfcc_features).any():  # Check if MFCC features are valid
        # Calculate row-wise average
        mfcc_avg = np.mean(mfcc_features, axis=1, keepdims=True)
        mfcc_features_reshaped = mfcc_avg.reshape(1, 20)
        # Make prediction
        predicted_label = model.predict(mfcc_features_reshaped)
        if predicted_label == 1:
            st.markdown(yes_style, unsafe_allow_html=True)
        else:
            st.markdown(no_style, unsafe_allow_html=True)
    else:
        st.write("Could not extract valid MFCC features from the audio file.")

