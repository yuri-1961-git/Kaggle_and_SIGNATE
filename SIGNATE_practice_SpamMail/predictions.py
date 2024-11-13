# URL: https://signate.jp/competitions/104
# train.zipとtest.zipはあらかじめ解凍してディレクトリに保存しておく（ディレクトリ名はそれぞれtrain2, test2）

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import csv

# 1. 学習データとラベルの対応表を読み込み
train_master = pd.read_csv('train_master.tsv', sep='\t')

# 2. 学習用データのテキストを読み込む関数
def load_texts(data_dir, file_names):
    texts = []
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

# 3. train2フォルダからテキストを読み込み
train_texts = load_texts('train2', train_master['file_name'])
train_labels = train_master['label']

# 4. テキストデータの前処理と特徴抽出（TF-IDFベクトル化）
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train = vectorizer.fit_transform(train_texts)
y_train = train_labels

# 5. モデルの定義と学習（ロジスティック回帰を使用）
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. test2フォルダからテキストを読み込み
test_files = sorted(os.listdir('test2'))
test_texts = load_texts('test2', test_files)
X_test = vectorizer.transform(test_texts)

# 7. 予測を実行
predictions = model.predict(X_test)

# 8. 結果をpredictions.csvに保存
output_file = 'predictions.csv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    for file_name, prediction in zip(test_files, predictions):
        writer.writerow([file_name, int(prediction)])

print(f'予測結果は {output_file} に保存されました。')

# 以下スパムメール分類に寄与する単語を可視化

import numpy as np
import matplotlib.pyplot as plt

# 1. 単語とその重要度（係数）の対応を取得
feature_names = np.array(vectorizer.get_feature_names_out())
coef = model.coef_.flatten()

# 2. 単語と係数をソート（スパムに寄与する単語を探す）
top_positive_coefficients = np.argsort(coef)[::-1][:10]  # 上位10個（スパムに寄与する単語）
top_negative_coefficients = np.argsort(coef)[:10]  # 下位10個（ノンスパムに寄与する単語）

# 3. 上位10個のスパムに寄与する単語とその係数
top_positive_words = feature_names[top_positive_coefficients]
top_positive_values = coef[top_positive_coefficients]

# 4. 上位10個のノンスパムに寄与する単語とその係数
top_negative_words = feature_names[top_negative_coefficients]
top_negative_values = coef[top_negative_coefficients]

# 5. 可視化
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# スパム寄与単語の可視化
ax[0].barh(top_positive_words, top_positive_values, color='red')
ax[0].set_title("Top 10 Spam-Indicative Words")
ax[0].set_xlabel("Coefficient Value")

# ノンスパム寄与単語の可視化
ax[1].barh(top_negative_words, top_negative_values, color='green')
ax[1].set_title("Top 10 Non-Spam-Indicative Words")
ax[1].set_xlabel("Coefficient Value")

plt.tight_layout()
plt.show()
