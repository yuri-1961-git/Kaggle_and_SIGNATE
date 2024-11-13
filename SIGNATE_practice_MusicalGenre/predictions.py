#URL: https://signate.jp/competitions/103
#学習用データと評価用データのzipファイルはあらかじめ解凍しておく

import os
import pandas as pd
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
import csv

# 1. 学習用データのラベルIDを取得
train_master = pd.read_csv('train_master.tsv', sep='\t')

# 2. ラベルIDをファイル名に対応させる辞書を作成
label_map = {row['file_name']: row['label_id'] for _, row in train_master.iterrows()}

# 3. 学習データの特徴量を準備する関数
def extract_features_from_file(file_path):
    # 音声ファイルの読み込み
    y, sr = librosa.load(file_path)
    # メル周波数ケプストラム係数(MFCC)を特徴量として抽出
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # MFCCの平均を特徴量として使用
    return np.mean(mfcc.T, axis=0)

# 4. 学習用データの特徴量を準備
train_files = sorted(os.listdir('train_sound_1') + os.listdir('train_sound_2') + os.listdir('train_sound_3'))
X_train = []
y_train = []

# 5.  各ファイルに対して特徴量を抽出
for file_name in train_files:
    file_path = os.path.join('train_sound_1', file_name)  # フォルダに応じて修正
    if not os.path.exists(file_path):
        file_path = os.path.join('train_sound_2', file_name)
    if not os.path.exists(file_path):
        file_path = os.path.join('train_sound_3', file_name)

    # 特徴量の抽出
    features = extract_features_from_file(file_path)
    X_train.append(features)
    y_train.append(label_map[file_name])  # ラベルIDを取得

X_train = np.array(X_train)
y_train = np.array(y_train)

#学習用ファイルに対して、特徴量を抽出し、X_train に追加します。
#label_map[file_name] で、対応するラベルIDを取得し、y_train に追加します。

# 6. モデルの定義と学習
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. 評価用データの特徴量を準備
test_files = sorted(os.listdir('test_sound_1') + os.listdir('test_sound_2') + os.listdir('test_sound_3'))
X_test = []

for file_name in test_files:
    file_path = os.path.join('test_sound_1', file_name)  # フォルダに応じて修正
    if not os.path.exists(file_path):
        file_path = os.path.join('test_sound_2', file_name)
    if not os.path.exists(file_path):
        file_path = os.path.join('test_sound_3', file_name)

    features = extract_features_from_file(file_path)
    X_test.append(features)

X_test = np.array(X_test)

# 7. 予測を実行
predictions = model.predict(X_test)

# 8. 結果をpredictions.tsvとして保存
output_file = 'predictions.tsv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')  # タブ区切り
    #writer.writerow(['file_name', 'label_id'])  # ヘッダー行
    for file_name, prediction in zip(test_files, predictions):
        writer.writerow([file_name, prediction])

print(f'予測結果は {output_file} に保存されました。')
