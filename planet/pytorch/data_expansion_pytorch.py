# -*- coding: utf-8 -*-
import pandas as pd
from torch import np # Torch wrapper for Numpy

import os
from PIL import Image

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.preprocessing import MultiLabelBinarizer

IMG_PATH = '../train-jpg-rgb/'
IMG_EXT = '.jpg'
DATA = '../train_v2.csv'

class KaggleAmazonDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):
    
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
"Some images referenced in the CSV file were not found"
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)


class AugmentedAmazonDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.
    This dataset is augmented

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):
    
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
"Some images referenced in the CSV file were not found"
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)
        self.augmentNumber = 14 # TODO, do something about this harcoded value

    def __getitem__(self, index):
        real_length = self.real_length()
        real_index = index % real_length
        
        img = Image.open(self.img_path + self.X_train[real_index] + self.img_ext)
        img = img.convert('RGB')
        
        ## Augmentation code
        if 0 <= index < real_length:
            pass
        
        ### Mirroring and Rotating
        elif real_length <= index < 2 * real_length:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif 2 * real_length <= index < 3 * real_length:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif 3 * real_length <= index < 4 * real_length:
            img = img.transpose(Image.ROTATE_90)
        elif 4 * real_length <= index < 5 * real_length:
            img = img.transpose(Image.ROTATE_180)
        elif 5 * real_length <= index < 6 * real_length:
            img = img.transpose(Image.ROTATE_270)

        ### Color balance
        elif 6 * real_length <= index < 7 * real_length:
            img = Color(img).enhance(0.95)
        elif 7 * real_length <= index < 8 * real_length:
            img = Color(img).enhance(1.05)
        ## Contrast
        elif 8 * real_length <= index < 9 * real_length:
            img = Contrast(img).enhance(0.95)
        elif 9 * real_length <= index < 10 * real_length:
            img = Contrast(img).enhance(1.05)
        ## Brightness
        elif 10 * real_length <= index < 11 * real_length:
            img = Brightness(img).enhance(0.95)
        elif 11 * real_length <= index < 12 * real_length:
            img = Brightness(img).enhance(1.05)
        ## Sharpness
        elif 12 * real_length <= index < 13 * real_length:
            img = Sharpness(img).enhance(0.95)
        elif 13 * real_length <= index < 14 * real_length:
            img = Sharpness(img).enhance(1.05)
        else:
            raise IndexError("Index out of bounds")
            
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.from_numpy(self.y_train[real_index])
        return img, label
    
    def __len__(self):
        return len(self.X_train.index) * self.augmentNumber
    
    def real_length(self):
        return len(self.X_train.index)


def augmented_train_valid_split(dataset, test_size = 0.25, shuffle = False, random_seed = 0):
    """ Return a list of splitted indices from a DataSet.
    Indices can be used with DataLoader to build a train and validation set.
    
    Arguments:
        A Dataset
        A test_size, as a float between 0 and 1 (percentage split) or as an int (fixed number split)
        Shuffling True or False
        Random seed
    """
    length = dataset.real_length()
    indices = list(range(1,length))
    
    if shuffle == True:
        random.seed(random_seed)
        random.shuffle(indices)
    
    if type(test_size) is float:
        split = floor(test_size * length)
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or a float' % str)
    return indices[split:], indices[:split]


#transformations = transforms.Compose([transforms.Scale(32),transforms.ToTensor()])
#dset_train = KaggleAmazonDataset(DATA,IMG_PATH,IMG_EXT,transformations)
#print("train:", len(dset_train))
#train_loader = DataLoader(dset_train,
#                          batch_size=256,
#                          shuffle=True,
#                          num_workers=4, # 1 for CUDA
#                          pin_memory=True # CUDA only
#                         )

ds_transform = transforms.Compose([transforms.Scale(32),transforms.ToTensor()])
#ds_transform = transforms.ToTensor()
X_train = AugmentedAmazonDataset(DATA,IMG_PATH,IMG_EXT,
                            ds_transform
                            )

# Creating a validation split
train_idx, valid_idx = augmented_train_valid_split(X_train, 15000)

nb_augment = X_train.augmentNumber
augmented_train_idx = [i * nb_augment + idx for idx in train_idx for i in range(0,nb_augment)]

train_sampler = SubsetRandomSampler(augmented_train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Both dataloader loads from the same dataset but with different indices
train_loader = DataLoader(X_train,
                      batch_size=64,
                      sampler=train_sampler,
                      num_workers=4,
                      pin_memory=True)

valid_loader = DataLoader(X_train,
                      batch_size=64,
                      sampler=valid_sampler,
                      num_workers=1,
                      pin_memory=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, 17)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1) # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x)

#model = Net() # On CPU
model = Net().cuda() # On GPU
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        data, target = Variable(data), Variable(target)
        #print(data.size())
        #print(target.size())
        optimizer.zero_grad()
        output = model(data)
        
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        if batch_idx % 100 == 0:
            for batch_index in range(output.size()[0]):
                #print(batch_index)
                #print(output[batch_index].size()[0])
                for arr_index in range(output[batch_index].size()[0]):
                    #print(arr_index)
                    print("output ",float(output[batch_index][arr_index].data.cpu().numpy()))
                    print("out ", out)
                    print("target ", float(target[batch_index][arr_index].data.cpu().numpy()))



for epoch in range(1, 5):
    train(epoch)