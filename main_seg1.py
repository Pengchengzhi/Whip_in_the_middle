from load_data import Imgnet_Dataset as myDataset
from torch.utils import data

data_root = './'          # Working directory
data_train = myDataset(data_root, shuffle = True, size = 256, mode = 'train')
train_loader = data.DataLoader(data_train, batch_size = 10, shuffle = False)

data_test = myDataset(data_root, shuffle = True, size = 256, mode = 'test')
test_loader = data.DataLoader(data_train, batch_size = 10, shuffle = False)

for i, (data, label) in enumerate(train_loader):
    print(label)
