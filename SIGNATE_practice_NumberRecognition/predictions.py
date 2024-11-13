#URL: https://signate.jp/competitions/104
#train.zipとtest.zipをディレクトリに格納（解凍は不要）

import os
from zipfile import ZipFile
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import csv
import re

# 1. train.zip と test.zip の解凍
with ZipFile('train.zip', 'r') as zip_ref:
    zip_ref.extractall()
with ZipFile('test.zip', 'r') as zip_ref:
    zip_ref.extractall()

# 2. train_master.tsv の読み込み
train_master = pd.read_csv('train_master.tsv', sep='\t')

# 3. 画像データとラベルIDの対応クラス作成
class DigitDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, is_test=False):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test  # is_testを追加

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # 画像をグレースケールで読み込む

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, self.dataframe.iloc[idx, 0]  # ファイル名を返す
        else:
            label = int(self.dataframe.iloc[idx, 1])  # ラベルを整数に変換
            return image, label

# 4. データの前処理
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = DigitDataset(train_master, 'train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 5. CNNモデルの定義
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # MaxPooling層
        # fc1の入力サイズを後で計算するために保留
        self.fc1 = None
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # conv1 と conv2 を通す
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # 畳み込み層とプーリング層を通過した後のサイズを計算
        x = torch.flatten(x, 1)
        
        # fc1の入力サイズを動的に計算
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128)
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 6. モデルのインスタンス化と学習設定
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. 学習ループ
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)  # labelsはTensor型として渡す
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

#8. 予測用のデータセット作成
test_images = os.listdir('test')
test_dataset = DigitDataset(pd.DataFrame(test_images, columns=['filename']), 'test', transform=transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#9 モデルの予測
model.eval()
predictions = []
with torch.no_grad():
    for images, filenames in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(zip(filenames, predicted.cpu().numpy()))  # ファイル名と予測をペアで保存

#10 結果を出力
# 結果をTSVファイルとして保存（filename順でソートしてヘッダ削除）
output_file = 'predictions.tsv'

# filenameから数字を抽出してソート
predictions.sort(key=lambda x: int(re.search(r'(\d+)', x[0]).group(1)))  # ファイル名の数字部分でソート

# 結果をTSVファイルとして保存
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')  # タブ区切り
    # ヘッダは削除して書き込まない
    writer.writerows(predictions)
print(f'予測結果は {output_file} に保存されました。')
