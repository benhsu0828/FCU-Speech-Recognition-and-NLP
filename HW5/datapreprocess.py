import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import random
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# === Configurations ===
SOURCE_DIR = "./Speaker Identification_Dataset-2021"
TARGET_DIR = "./data"
SAMPLE_RATE = 16000
N_MFCC = 13
MFCC_HOP_LENGTH = 192
MFCC_N_FFT = 512

# === Create output directories ===
os.makedirs(os.path.join(TARGET_DIR, "training_data"), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "testing_data"), exist_ok=True)

# === Helper Functions ===
def normalize_audio(y):
    """
    將音頻信號正規化到 [-1, 1] 範圍。
    """
    return y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

def extract_mfcc_features(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    提取 MFCC 特徵序列。
    """
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=MFCC_HOP_LENGTH,
        n_fft=MFCC_N_FFT
    )
    return mfcc.T  # shape: (frames, n_mfcc)

# === Step 1: Collect file paths ===
file_entries = []
for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        if file.lower().endswith(".wav"):
            speaker = os.path.basename(root)
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                file_entries.append((file_path, speaker))

if not file_entries:
    print("No .wav files found in the dataset directory.")
else:
    print(f"Total files collected: {len(file_entries)}")

# === Step 2: Shuffle and Split by Speaker ===
# 將資料按語者分組
speaker_groups = defaultdict(list)
for file_path, speaker in file_entries:
    speaker_groups[speaker].append(file_path)

# 按語者分割資料
train_entries = []
test_entries = []

for speaker, files in speaker_groups.items():
    random.shuffle(files)  # 對每個語者的資料隨機打亂
    n_total = len(files)
    n_train = int(0.8 * n_total)  # 80% 用於訓練
    train_entries.extend([(file, speaker) for file in files[:n_train]])
    test_entries.extend([(file, speaker) for file in files[n_train:]])

# 構建 splits 字典
splits = {
    "training_data": train_entries,
    "testing_data": test_entries
}

print(f"Training data: {len(splits['training_data'])} files")
print(f"Testing data: {len(splits['testing_data'])} files")

# === Step 3: Process and Extract Features ===
def process_file(entry):
    file_path, speaker = entry
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        y = normalize_audio(y)
        mfcc_feat = extract_mfcc_features(y)
        frames = []
        for frame_idx, frame in enumerate(mfcc_feat):
            frame_filename = f"{speaker}_frame{frame_idx}.npy"
            frame_path = os.path.join(TARGET_DIR, "training_data" if entry in splits["training_data"] else "testing_data", frame_filename)
            np.save(frame_path, frame)
            frames.append({
                "path": frame_path,
                "speaker": speaker,
                "split": "training_data" if entry in splits["training_data"] else "testing_data"
            })
        return frames
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

# 使用多線程處理音檔
with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_file, splits["training_data"] + splits["testing_data"]), total=len(splits["training_data"]) + len(splits["testing_data"])))

# 展平結果
csv_entries = [item for sublist in results for item in sublist]

# === Step 4: Save Metadata ===
df = pd.DataFrame(csv_entries)
df.to_csv(os.path.join(TARGET_DIR, "mfcc_metadata.csv"), index=False)

print("Feature extraction complete. Frames and metadata saved.")
print(f"Metadata saved to {os.path.join(TARGET_DIR, 'mfcc_metadata.csv')}")
print(f"Training data: {len(splits['training_data'])} files")
print(f"Testing data: {len(splits['testing_data'])} files")

# === Step 5: Save Features and Labels ===
features = np.array([np.load(entry["path"]) for entry in csv_entries])  # 加載所有特徵
labels = np.array([entry["speaker"] for entry in csv_entries])  # 加載所有標籤
split_labels = np.array([entry["split"] for entry in csv_entries])  # 加載所有分割標籤

# 保存到 .npz 文件
np.savez(os.path.join(TARGET_DIR, "mfcc_data.npz"), features=features, labels=labels, splits=split_labels)
print(f"Features and labels saved to {os.path.join(TARGET_DIR, 'mfcc_data.npz')}")
