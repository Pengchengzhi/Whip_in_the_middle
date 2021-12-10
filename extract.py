import os, sys
import tarfile
import shutil
import numpy as np
from tqdm import tqdm

#### Initialization
Ratio = 1          # Ratio of images used
Fraction = 0.8          # Fraction of training dataset

root = '/Volumes/Jingyu-SSD/ImageNet/'
# if os.path.isdir(root + 'image/'):
#     shutil.rmtree(root + 'image/')
# if os.path.isdir(root + 'train/'):
#     shutil.rmtree(root + 'train/')
# if os.path.isdir(root + 'test/'):
#     shutil.rmtree(root + 'test/')

train_folder = root + 'train/'
test_folder = root + 'test/'

os.mkdir(root+'image')
os.mkdir(root+'test')
os.mkdir(root+'train')

#### Untar tar_small
#tarlist = os.listdir(root + 'tar_small/')
tarlist = np.load('interested_class_in.npy', allow_pickle=True)
image_dir = root + 'image/'          # image dir

for tarf in tarlist:
    print(tarf)
    path = root + 'tar_file/' + str(tarf) +  '.tar'
    tf = tarfile.open(path)

    inp_path = image_dir + str(tarf) + '/'
    if os.path.isdir(inp_path):
        shutil.rmtree(inp_path)
    tf.extractall(inp_path)
	

#### Extract each class
for tarf in tqdm(tarlist):
    inp_path = image_dir + str(tarf) + '/'
    imagelist = os.listdir(inp_path) 

    L = int(len(imagelist)*Ratio)
    idex = np.array(range(L))
    np.random.shuffle(idex)

    for i in range(L):
        if Fraction*L <= i:
            tmp = imagelist[idex[i]]
            img = inp_path + str(tmp)
            os.makedirs(os.path.dirname(test_folder+f'{str(tarf)}/'), exist_ok=True)
            shutil.copy(img, test_folder+f'{str(tarf)}/')
        else:
            tmp = imagelist[idex[i]]
            img = inp_path + str(tmp)
            os.makedirs(os.path.dirname(train_folder+f'{str(tarf)}/'), exist_ok=True)
            shutil.copy(img, train_folder+f'{str(tarf)}/')
