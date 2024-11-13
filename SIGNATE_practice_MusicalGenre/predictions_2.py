#URL: https://signate.jp/competitions/103
#学習用データと評価用データのzipファイルはあらかじめ解凍しておく

import os
import csv
import librosa
import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import csv
import re

# 特徴量を抽出する関数
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # MFCC (メル周波数ケプストラム係数)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # 次元を20に増やす
    # クロマ特徴量
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # ゼロ交差率
    zcr = librosa.feature.zero_crossing_rate(y)
    # スペクトルロールオフ
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    # エネルギー
    spectral_energy = np.sum(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
    # テンポ（ビート）
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    # すべての特徴量を結合
    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(zcr),
        np.mean(spectral_rolloff),
        np.mean(spectral_energy),
        tempo  # テンポの平均を追加
    ])
    return features

# ファイルパスを取得し、数値順に並べ替える関数
def get_file_paths(base_folders, prefix):
    file_paths = []
    for folder in base_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.startswith(prefix) and file.endswith('.au'):
                    file_paths.append(os.path.join(root, file))
    
    # ファイル名の数字部分で並べ替える
    file_paths.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group()))
    return file_paths

# ラベル辞書を読み込む関数
def load_label_dict(master_file):
    label_dict = {}
    with open(master_file, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for row in reader:
            label_dict[row[0]] = int(row[1])
    return label_dict

# train_master.tsvを読み込み、ファイル名とラベルIDの対応を作成する関数
def create_label_list(train_files, train_master_file, label_dict):
    file_label_map = {}
    with open(train_master_file, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for row in reader:
            filename, label_name = row[0], row[1]
            if label_name.isdigit():  # ラベル名が数値の場合
                label_id = int(label_name)
                file_label_map[filename] = label_id
            elif label_name in label_dict:  # ラベル名が文字列の場合
                file_label_map[filename] = label_dict[label_name]
            else:
                raise ValueError(f"ラベル名 '{label_name}' がラベルマスタに見つかりません。")

    labels = [file_label_map[os.path.basename(file)] for file in train_files]
    return labels

# データ読み込みと特徴量作成
train_folders = ['train_sound_1', 'train_sound_2', 'train_sound_3']
train_files = get_file_paths(train_folders, 'train_')
label_dict = load_label_dict('label_master.tsv')
train_labels = create_label_list(train_files, 'train_master.tsv', label_dict)

X = np.array([extract_features(file) for file in train_files])
y = np.array(train_labels)

# 学習と検証用データに分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# OptunaによるLightGBMのハイパーパラメータ最適化
def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': 10,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500)
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    return score

# Optunaで最適化
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
best_params = study.best_params

# 最適化されたモデルで再学習
best_model = LGBMClassifier(**best_params)
best_model.fit(X_train, y_train)

# テストデータの特徴量を生成して予測
test_folders = ['test_sound_1', 'test_sound_2', 'test_sound_3']
test_files = get_file_paths(test_folders, 'test_')
X_test = np.array([extract_features(file) for file in test_files])
predictions = best_model.predict(X_test)

# 結果をpredictions_2.tsvとして保存
output_file = 'predictions_2.tsv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    for file_path, prediction in zip(test_files, predictions):
        writer.writerow([os.path.basename(file_path), prediction])

print(f'予測結果は {output_file} に保存されました。')
