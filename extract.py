import os, sys
import tarfile
import shutil
import numpy as np

#### Initialization
Ratio = 0.1          # Ratio of images used
Fraction = 0.8          # Fraction of training dataset

root = './'
if os.path.isdir(root + 'image/'):
    shutil.rmtree(root + 'image/')
if os.path.isdir(root + 'train/'):
    shutil.rmtree(root + 'train/')
if os.path.isdir(root + 'test/'):
    shutil.rmtree(root + 'test/')

train_folder = root + 'train/'
test_folder = root + 'test/'

os.mkdir('image')
os.mkdir('test')
os.mkdir('train')

#### Untar tar_small
tarlist = os.listdir(root + 'tar_small/') 
image_dir = root + 'image/'          # image dir

for tarf in tarlist:
    print(tarf)
    path = root + 'tar_small/' + str(tarf)
    tf = tarfile.open(path)

    inp_path = image_dir + str(tarf) + '/'
    if os.path.isdir(inp_path):
        shutil.rmtree(inp_path)
    tf.extractall(inp_path)
	

#### Extract each class
for tarf in tarlist:
    inp_path = image_dir + str(tarf) + '/'
    imagelist = os.listdir(inp_path) 

    L = int(len(imagelist)*Ratio)
    idex = np.array(range(L))
    np.random.shuffle(idex)

    for i in range(L):
        if Fraction*L <= i:
            tmp = imagelist[idex[i]]
            img = inp_path + str(tmp)
            shutil.copy(img, test_folder)
        else:
            tmp = imagelist[idex[i]]
            img = inp_path + str(tmp)
            shutil.copy(img, train_folder)
