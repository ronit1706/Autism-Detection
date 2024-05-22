import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
if not os.path.exists('/Users/ronitkhurana/PycharmProjects/autism/features/aut_Recording_6.m4a.npy'):
    import mfcc_extract

# Load MFCC features
features_folder = "features"
autistic_files = [f for f in os.listdir(features_folder) if f.startswith("aut_")]
non_autistic_files = [f for f in os.listdir(features_folder) if f.startswith("split-")]

# Define a function to load and pad/truncate MFCC data
def load_and_average_data(files, max_frames):
    data_list = []
    for file in files:
        data = np.load(os.path.join(features_folder, file))
        # print(data.shape)
        avg = np.mean(data, axis=1)  # Calculate row-wise average
        # avg = [avg[0],avg[2],avg[4]]
        data_list.append(avg)  # Append the averaged data to the list



    return np.stack(data_list, axis=0)

# Load and process autistic data
autistic_data = load_and_average_data(autistic_files, 20)

# Load and process non-autistic data
non_autistic_data = load_and_average_data(non_autistic_files, 20)

# Combine data and labels
X = np.vstack((autistic_data, non_autistic_data))
y = np.hstack((np.ones(autistic_data.shape[0]), np.zeros(non_autistic_data.shape[0])))

# Split data into train and test sets equally for both classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

#RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Support Vector Machine
svm_classifier = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)

# Naive Bayes
nb_classifier = GaussianNB()

#Artificial Neural Network
ann_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest model trained with accuracy:", accuracy)

# Save the model
joblib.dump(rf_classifier, 'rf.pkl')

# Train the model
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("SVM model trained with accuracy:", accuracy)

# Save the model
joblib.dump(svm_classifier, 'svm.pkl')


# Train the model
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Naive Bayes model trained with accuracy:", accuracy)

# Save the model
joblib.dump(nb_classifier, 'nb.pkl')

# Train the model
ann_classifier.fit(X_train, y_train)

# Make predictions
y_pred = ann_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Artificial Neural Network model trained with accuracy:", accuracy)

# Save the model
joblib.dump(ann_classifier, 'ann.pkl')



