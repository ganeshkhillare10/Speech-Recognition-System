import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Function to extract features from audio files
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    with librosa.load(file_path) as audio_data:
        X, sample_rate = audio_data
        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))

    return result

# Load the dataset (replace 'your_dataset_path' with the actual path)
def load_data(dataset_path):
    emotions = {'happy': 0, 'sad': 1, 'neutral': 2, 'angry': 3, 'fearful': 4}

    X, y = [], []
    for emotion, label in emotions.items():
        emotion_path = os.path.join(dataset_path, emotion)
        for filename in os.listdir(emotion_path):
            file_path = os.path.join(emotion_path, filename)
            feature = extract_features(file_path)
            X.append(feature)
            y.append(label)

    return np.array(X), np.array(y)

# Main function
def main():
    dataset_path = 'your_dataset_path'  # Replace with the path to your dataset
    X, y = load_data(dataset_path)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Support Vector Machine (SVM) model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    # Make predictions on the testing set
    predictions = svm_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", classification_report(y_test, predictions))

if __name__ == "__main__":
    main()
