import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

dataset_path = "dataset_V3"
label_file = "labels_V3.txt"

# Fetch all .npy files from the dataset directory
files = [f for f in os.listdir(dataset_path) if f.endswith('.npy')]

X = []
y = []

labels = []

# Load data and labels
for idx, file in enumerate(files):
    data = np.load(os.path.join(dataset_path, file))
    X.extend(data)  # Note: using extend instead of append
    label = file.split('.')[0]
    labels.append(label)
    y.extend([idx] * data.shape[0])

X = np.array(X)
y = np.array(y)

# Save labels to labels.txt
with open(label_file, 'w') as f:
    for label in labels:
        f.write(label + "\n")

# Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train SVM model
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)

# Save the model and the scaler
model_filename = "svm_model_V3.pkl"
joblib.dump(clf, model_filename)
scaler_filename = "scaler_V3.pkl"
joblib.dump(scaler, scaler_filename)

print(f"Model saved to {model_filename}")
print(f"Scaler saved to {scaler_filename}")
print(f"Labels saved to {label_file}")