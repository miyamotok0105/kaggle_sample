# encoding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import time
import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import numpy as np
from matplotlib import pyplot as plt

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#正規化をしない前処理
to_tensor_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

train = os.listdir("train")
print("training ", len(train))

test = os.listdir("test")
print("test ", len(test))

import pandas as pd

df = pd.read_csv('train.csv')
print(df.head())

#クラス数
classes = set(df.Id.tolist())
print(list(classes)[:5])
print(len(classes))
print(type(classes))



df.Id.unique()

#classesの中身もユニークなのでclassesでも良い
label = {}
for i, c in enumerate(classes):
  label[c] = i

# label

df["label"] = df.Id.map(dict(label))
df.head()


#==================================================
root = './'
class CustomDataset(torch.utils.data.Dataset):

  def __init__(self, root, transform=None, train=True):
    # 指定する場合は前処理クラスを受け取ります。
    self.transform = transform
    # 画像とラベルの一覧を保持するリスト
    self.all_images = []
    self.all_labels = []
    self.images = []
    self.labels = []
    # ルートフォルダーパス
    root = "./"
    # 訓練の場合と検証の場合でフォルダわけ
    # 画像を読み込むファイルパスを取得します。
    root_path = os.path.join(root, 'train')
    self.all_images = df.Image.tolist()
    self.all_images = [os.path.join(root_path, file) for file in self.all_images]
    self.all_labels = df.label.tolist()
    slice_index = int(len(df.Image)*0.9)
    
    if train == True:
      self.images = self.all_images[:slice_index]
      self.labels = self.all_labels[:slice_index]
      #             print(self.images[:self.slice_index])
      #             print(self.labels[:self.slice_index])
    else:
      self.images = self.all_images[slice_index:]
      self.labels = self.all_labels[slice_index:]
      #             print(self.images[self.slice_index:])
      #             print(self.labels[self.slice_index:])
    print("len ", len(self.images))
    # 画像一覧を取得します。
      
  def __getitem__(self, index):
    # インデックスを元に画像のファイルパスとラベルを取得します。
    image = self.images[index]
    label = self.labels[index]
    # 画像ファイルパスから画像を読み込みます。
    with open(image, 'rb') as f:
      image = Image.open(f)
      image = image.convert('RGB')
    # 前処理がある場合は前処理をいれます。
    if self.transform is not None:
      image = self.transform(image)
    # 画像とラベルのペアを返却します。
    return image, label
      
  def __len__(self):
    # ここにはデータ数を指定します。
    return len(self.images)
        
#==================================================

#==================================================
#画像の前処理を定義
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 定義したDatasetとDataLoaderを使います。
custom_train_dataset = CustomDataset(root, data_transforms["train"], train=True)
train_loader = torch.utils.data.DataLoader(dataset=custom_train_dataset,
                                           batch_size=64, 
                                           shuffle=True)
custom_test_dataset = CustomDataset(root, data_transforms["test"])
test_loader = torch.utils.data.DataLoader(dataset=custom_test_dataset,
                                           batch_size=64, 
                                           shuffle=False)

for i, (images, labels) in enumerate(train_loader):
    print(images.size())
    print(images[0].size())    
    print(labels[0].item())
    #ここに訓練などの処理をきます。
    break

#==================================================

#==================================================


#クラス数
num_classes = len(classes)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

net = models.alexnet(pretrained=True)
net = net.to(device)


#ネットワークのパラメータを凍結
for param in net.parameters():
    param.requires_grad = False
net = net.to(device)
#最終層を2クラス用に変更
num_ftrs = net.classifier[6].in_features
net.classifier[6] = nn.Linear(num_ftrs, num_classes).to(device)


#最適化関数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#==================================================
num_epochs = 20

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    
    #train
    net.train()
    for i, (images, labels) in enumerate(train_loader):
      images, labels = images.to(device), labels.to(device)
      
      optimizer.zero_grad()
      outputs = net(images)
      loss = criterion(outputs, labels)
      train_loss += loss.item()
      train_acc += (outputs.max(1)[1] == labels).sum().item()
      loss.backward()
      optimizer.step()
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_acc / len(train_loader.dataset)
    
    #val
    net.eval()
    with torch.no_grad():
      for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        val_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc = val_acc / len(test_loader.dataset)
    
    print ('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, lr：{learning_rate}' 
                       .format(epoch+1, num_epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc, learning_rate=optimizer.param_groups[0]["lr"]))
    #学習率調整
    lr_scheduler.step()
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)


# num_epochs = 100

# train_loss_list = []
# train_acc_list = []
# val_loss_list = []
# val_acc_list = []

# for epoch in range(num_epochs):
#     train_loss = 0
#     train_acc = 0
#     val_loss = 0
#     val_acc = 0
    
#     #train
#     net.train()
#     for i, (images, labels) in enumerate(train_loader):
#       images, labels = images.to(device), labels.to(device)
      
#       optimizer.zero_grad()
#       outputs = net(images)
#       loss = criterion(outputs, labels)
#       train_loss += loss.item()
#       train_acc += (outputs.max(1)[1] == labels).sum().item()
#       loss.backward()
#       optimizer.step()
    
#     avg_train_loss = train_loss / len(train_loader.dataset)
#     avg_train_acc = train_acc / len(train_loader.dataset)
    
#     #val
#     net.eval()
#     with torch.no_grad():
#       for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = net(images)
#         loss = criterion(outputs, labels)
#         val_loss += loss.item()
#         val_acc += (outputs.max(1)[1] == labels).sum().item()
#     avg_val_loss = val_loss / len(test_loader.dataset)
#     avg_val_acc = val_acc / len(test_loader.dataset)
    
#     print ('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}' 
#                    .format(epoch+1, num_epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))
#     train_loss_list.append(avg_train_loss)
#     train_acc_list.append(avg_train_acc)
#     val_loss_list.append(avg_val_loss)
#     val_acc_list.append(avg_val_acc)

#==================================================

import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(num_epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
plt.plot(range(num_epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.grid()
plt.savefig('loss.png')

plt.figure()
plt.plot(range(num_epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
plt.plot(range(num_epochs), val_acc_list, color='green', linestyle='--', label='val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Training and validation accuracy')
plt.grid()
plt.savefig('acc.png')

#==================================================


