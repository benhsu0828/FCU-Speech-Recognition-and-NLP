import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import os



def zero_justification(raw_data):
    x = np.arange(256)
    coefs = np.polyfit(x, raw_data, 3)
    coefs_point = np.polyval(coefs, x)
    return raw_data - coefs_point

def processData(dir):
    file_path = os.walk(dir)
    for root, dirs, files in file_path:
        for file in files:
            if file.endswith(".wav"):
                sample_rate, raw_data = wavfile.read(os.path.join(root, file))
                raw_data = raw_data[0:16000]
                norm_data = raw_data / np.max(np.abs(raw_data))
                norm_data = norm_data - np.mean(norm_data)
                zj_frame_data = zero_justification(norm_data)
                yield norm_data, zj_frame_data

if __name__ == "__main__":
    sample_rate, raw_data = wavfile.read("star_noisy.wav")
    raw_data = raw_data[0:16000]

    model = Sequential([
        Flatten(input_shape=mfcc.shape),  # Adjust input shape accordingly
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # For binary classification (onset/offset detection)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        

