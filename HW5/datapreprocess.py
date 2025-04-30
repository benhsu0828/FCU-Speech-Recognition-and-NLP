import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import random

# === Configurations ===
SOURCE_DIR = "./Speaker Identification_Dataset-2021"
TARGET_DIR = "./data"
SAMPLE_RATE = 16000
WINDOW_SIZE = SAMPLE_RATE  # 1 second
OVERLAP = 0.4
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))
VAD_THRESHOLD = 0.005

# === Create output directories ===
os.makedirs(os.path.join(TARGET_DIR, "training_data"), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "eval_data"), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "testing_data"), exist_ok=True)

# === Helper Functions ===
def is_voiced(y, threshold=VAD_THRESHOLD):
    energy = np.sum(y ** 2) / len(y)
    return energy > threshold

def split_voiced_segments(y, top_db=20):
    """
    使用 librosa.effects.split 根據能量檢測語音活動區域。
    :param y: 音頻信號
    :param top_db: 靜音檢測的分貝閾值（越低越敏感）
    :return: 語音活動區域的切片列表
    """
    intervals = librosa.effects.split(y, top_db=top_db)
    return intervals

# === Step 1: Collect file paths ===
file_entries = []

# 使用 os.walk 遍歷資料夾
for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        # 確保只處理 .wav 檔案
        if file.lower().endswith(".wav"):
            # 獲取語者名稱（資料夾名稱）
            speaker = os.path.basename(root)
            # 獲取完整檔案路徑
            file_path = os.path.join(root, file)
            # 確保檔案存在
            if os.path.isfile(file_path):
                file_entries.append((file_path, speaker))

# 檢查是否成功收集所有檔案
if not file_entries:
    print("No .wav files found in the dataset directory.")
else:
    print(f"Total files collected: {len(file_entries)}")

# === Step 2: Shuffle and Split ===
random.shuffle(file_entries)
n_total = len(file_entries)
n_train = int(0.7 * n_total)
n_eval = int(0.15 * n_total)

splits = {
    "training_data": file_entries[:n_train],
    "eval_data": file_entries[n_train:n_train + n_eval],
    "testing_data": file_entries[n_train + n_eval:]
}

# === Step 3: Process Audio Files ===
csv_entries = []

for split_name, entries in splits.items():
    split_dir = os.path.join(TARGET_DIR, split_name)
    for file_path, speaker in tqdm(entries, desc=f"Processing {split_name}"):
        try:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # 使用 librosa.effects.split 檢測語音活動區域
        intervals = split_voiced_segments(y)

        segment_idx = 0
        for start, end in intervals:
            segment = y[start:end]
            if len(segment) < WINDOW_SIZE:
                segment = np.pad(segment, (0, WINDOW_SIZE - len(segment)))

            segment_filename = f"{speaker}_{start}_seg{segment_idx}.wav"
            segment_path = os.path.join(split_dir, segment_filename)
            sf.write(segment_path, segment, SAMPLE_RATE)
            csv_entries.append({
                "path": segment_path,
                "speaker": speaker,
                "split": split_name
            })
            segment_idx += 1


# === Step 4: Save Metadata ===
df = pd.DataFrame(csv_entries)
df.to_csv(os.path.join(TARGET_DIR, "metadata.csv"), index=False)

print("Audio processing complete. Segments saved with metadata.csv.")
