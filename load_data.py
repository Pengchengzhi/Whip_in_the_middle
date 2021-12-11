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
import torchvision.transforms as transforms
import numpy as np

data_root = './'          # Working directory, containing /train and /val
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])          # Nomalizaion

blur1 = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2))
blur2 = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2))
blur3 = transforms.GaussianBlur(kernel_size=(1, 1), sigma=(0.1, 2))           # No blur


data_train = myDataset(data_root, mode = 'train',
                       transform = transforms.Compose([
                           transforms.Resize([256,256]),          # Resize to 256
                           transforms.RandomHorizontalFlip(),          # Random flip
                           transforms.RandomVerticalFlip(),
                           transforms.RandomRotation(90),          # Random rotation
                           transforms.RandomPerspective(0.2),          # Random perspective change
                           transforms.RandomCrop([224, 224]),          # Random Crop
                           transforms.RandomChoice([blur1, blur2, blur3]),          # Random blur
                           transforms.ToTensor(),
                           normalize,
                       ]))
train_loader = data.DataLoader(data_train, batch_size = 10, shuffle = True)

data_val = myDataset(data_root, mode = 'val',
                       transform = transforms.Compose([
                          transforms.Resize([256,256]),
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomVerticalFlip(),
                           transforms.RandomRotation(90),
                           transforms.RandomPerspective(0.2),
                           transforms.RandomCrop([224, 224]),
                           transforms.RandomChoice([blur1, blur2, blur3]),
                           transforms.ToTensor(),
                           normalize,
                       ]))
val_loader = data.DataLoader(data_val, batch_size = 10, shuffle = False)

for i, (data, label) in enumerate(train_loader):
    print(i)
'''


#### Definition of loader
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class Imgnet_Dataset(data.Dataset):
    def __init__(self, root,
        #shuffle = True,
        #size = 224,
        mode = 'train',
        transform = None,
        loader = pil_loader):

        tic = time.time()
        self.root = root
        self.loader = loader
        #self.size = size
        self.transform = transform
        self.trainpath = glob.glob(root + 'train/*/*.JPEG')
        self.valpath = glob.glob(root + 'val/*/*.JPEG')
        
        tarlist = np.load('interested_class_in.npy', allow_pickle=True)
        self.dict = {}
        for i in range(len(tarlist)):
            self.dict[tarlist[i]] = i

        self.path = []
        if mode == 'train':
            for item in self.trainpath:
                self.path.append(item)
        elif mode == 'val':
            for item in self.valpath:
                self.path.append(item)

        #np.random.seed(37212)
        #if shuffle:
        #    perm = np.random.permutation(len(self.path))
        #    self.path = [self.path[i] for i in perm]

        print('Load %d images, used %fs' % (len(self.path), time.time()-tic))

    #### Get input img and its label
    def __getitem__(self, index):
        mypath = self.path[index]
        #label = mypath.split('_')[-2].split('\\')[-1]
        label = mypath.split('_')[-2].split('/')[-1]
        img = self.loader(mypath)
        if self.transform is not None:
            img = self.transform(img)
            
        #img = np.array(img)
        #if (img.shape[0] != self.size) or (img.shape[1] != self.size):          # Resize
        #    img = np.array(Image.fromarray(img).resize((self.size, self.size)))

        #img_lab = color.rgb2lab(np.array(img))          # RGB to LAB
        #img_lab = torch.FloatTensor(np.transpose(img_lab, (2,0,1)))
        #img_l = torch.unsqueeze(img_lab[0],0) / 100.           # L channel [0, 100]
        #img_ab = (img_lab[1::] + 0) / 110.          # ab channel [-110, - 110]
        #img = (img - 127.5) / 127.5          # Normalization
        #img = img / 255          # Normalization
        #img = torch.FloatTensor(np.transpose(img, (2,0,1)))          # 3*size*size
        
        return img, self.dict[label]
        
    def __len__(self):
        return len(self.path)
