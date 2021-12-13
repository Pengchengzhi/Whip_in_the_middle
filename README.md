# Whip_in_the_middle

## Log

* **11.07 (Matton):** Created a file structure to organize the code. Initiated the training code.
* **11.11 (Chengzhi):** Implemented loss 1 and use only that loss to train the model, based on the templet by Matton. Implemented test code. Created CIFAR-10 dataset and used a random hierarchy for training. The training accuracy for the current model is 97.63%, while the testing accuracy is 19.68%.
* **11.13 (Yuchuw):** Added the img_extractor and data_loader. Directory for downloading the dataset: /gpfs/accounts/eecs542f21_class_root/eecs542f21_class/shared_data/yuchuw/Dataset/. You could use any FTP applications to get access to it.
* **11.13 (Jingyu):** Build a draft tree structure based on anytree package. Use the `structure_released.xml` to get the wordnet hierarchy for each class in ImageNet. Continue working on the development of tree structure and function. 
* **11.16 (Chengzhi):** See file `cifar10_train_update_tree.ipynb`. Add weighted MSE loss as the new loss function. Add tree vector update fucntion. Add a ResNet classification model, training and test code, for comparision to our model. However, it's very strange I can't reproduce previous experiment results. I set the parameters into zeros and ones, so theoretically I changed nothing, but now training acc is 50% and testing acc is 27%. On the same dataset, compaire to 11.11, `cifar10_train.ipynb`.
* **12.8 (Jingyu):** Solved the previous `structure_released.xml` issue and construct the tree for synthetic dataset successfully. Will continue working on initialization of label function.
* **12.10 (Yuchuw):** Modified the `load_data.py` and `extract.py`. (1) Now, we have train and val folders containing 56 classes' folders; (2) Generated data Augmentation: Crop, Rotation, Flip, Perspective change, and Gaussion Blur; (3) Added `main_baseline_augmentation.py` using our augmentation and dataset.
* **12.10 (Jingyu):** Besline training using resnet18 (torchvision exampple) reached Acc@1 72.435, Acc@5 93.773 on the validation set of the synthetic dataset. Will do testing ASAP and also upload the weight.
* **12.11 (Jingyu):** Now the tree should be ready. The method for using the tree to get the vector is simple, add `from tree import build_tree` and call `label_dict = build_tree(dim_N=10)`. The labels are stored inside this label_dict. An example you may refer to query the label is `label = label_dict['n04141076']`. You can change the dim and length_list for each level by specifying `length_list=[10,5,4,3,2,2,2,1,1]`.
* **12.11 (Chengzhi):** Implemented train and evaluation code. Using the latest dataloader and dataset. The training does not converge at first, and then solved by introducing an normalization factor. 
* **12.12 (Mattonn):** Initialize labels to be uniformly distributed across a hypersphere. To use this label initialization, call build_tree with the flag brocolli_flag=False. This will return a dictionary of labels mapped to their vector representations.
* **12.12 (Chengzhi):** The training loss for the first round was 11, then in the following rounds it keeps at around 4. The sudden drop was solved by tweaking `hidden_dim`. Then I plotted the label vector for dim=2, found that label vectors are overlapping. So I spent the whole day tweaking `hidden_dim` and `length_list`. The training code runs very slow, takes 80 minutes to train 20 iterations. At the end, the training loss drops smoothly, but the prediction accuracy on train set keeps struggling at 1.5% - 2%.




