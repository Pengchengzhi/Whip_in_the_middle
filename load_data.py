import sys
import numpy as np
import time

import torch
import torchvision
from torch.utils import data
from PIL import Image

import glob
from skimage import color

''' Example of how to load data in the main.py
from load_data import Imgnet_Dataset as myDataset
from torch.utils import data

data_root = './'          # Working directory
data_train = myDataset(data_root, shuffle = True, size = 256, mode = 'train')
train_loader = data.DataLoader(data_train, batch_size = 10, shuffle = False)

data_test = myDataset(data_root, shuffle = True, size = 256, mode = 'test')
test_loader = data.DataLoader(data_train, batch_size = 10, shuffle = False)

for i, (data, label) in enumerate(train_loader):
    print(label)
'''

#### Definition of loader
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class Imgnet_Dataset(data.Dataset):
    def __init__(self, root,
        shuffle = True,
        size = 256,
        mode = 'test',
        loader = pil_loader):

        tic = time.time()
        self.root = root
        self.loader = loader
        self.size = size
        self.trainpath = glob.glob(root + 'train/*.JPEG')
        self.testpath = glob.glob(root + 'test/*.JPEG')

        self.path = []
        if mode == 'train':
            for item in self.trainpath:
                self.path.append(item)
        elif mode == 'test':
            for item in self.testpath:
                self.path.append(item)

        np.random.seed(37212)
        if shuffle:
            perm = np.random.permutation(len(self.path))
            self.path = [self.path[i] for i in perm]

        print('Load %d images, used %fs' % (len(self.path), time.time()-tic))

    #### Get input img and its label
    def __getitem__(self, index):
        mypath = self.path[index]
        label = mypath.split('_')[0].split('\\')[1]
        img = self.loader(mypath)
        img = np.array(img)        
        if (img.shape[0] != self.size) or (img.shape[1] != self.size):          # Resize
            img = np.array(Image.fromarray(img).resize((self.size, self.size)))

        #img_lab = color.rgb2lab(np.array(img))          # RGB to LAB
        #img_lab = torch.FloatTensor(np.transpose(img_lab, (2,0,1)))
        #img_l = torch.unsqueeze(img_lab[0],0) / 100.           # L channel [0, 100]
        #img_ab = (img_lab[1::] + 0) / 110.          # ab channel [-110, - 110]
            
        img = (img - 127.5) / 127.5          # Normalization
        img = torch.FloatTensor(np.transpose(img, (2,0,1)))          # 3*size*size

        return img, label
        
    def __len__(self):
        return len(self.path)
