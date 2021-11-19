## Maintainer: Jingyu Song #####
## Contact: jingyuso@umich.edu #####

import numpy as np
# import anytree
import os
# from exin.exin import find_class
import xml.etree.ElementTree as ET

imagenet_tree = ET.parse('structure_released.xml')
root = imagenet_tree.getroot()

sysnet = root[1]


def build_dict(sysnet_elem,name=None):
    if len(sysnet_elem) == 0:
        return name
    else:
        items_dict = {}
        for child in sysnet_elem:
            items_dict[child.attrib['wnid']] = (build_dict(child,child.attrib['words']),child.attrib['words'])
    return items_dict

def search_in_dict(item, imagenet_dict):
    return search_in_dict_path(item,(imagenet_dict,0),[])

def search_in_dict_path(item, imagenet_dict,path):
    if type(imagenet_dict[0]) == type('a'):
        return 0
    if item in imagenet_dict[0].keys():
        return path + [(item,imagenet_dict[0][item][1])] + [imagenet_dict[0][item][0]]
    elif len(imagenet_dict[0]) == 0:
        return 0
    else:
        for (k,v) in imagenet_dict[0].items():
            whole_path = search_in_dict_path(item,v,path + [(k,v[1])])
            if whole_path == 0:
                continue
            return whole_path
    return 0

whole_dict = build_dict(sysnet)
# print(search_in_dict('n02356798',whole_dict))

dataset_folder = "/Volumes/Jingyu-SSD/ImageNet/train"
dataset_dir = sorted(os.listdir(dataset_folder))
# print(dataset_dir)

class_names = [x.split('.')[0] for x in dataset_dir]
# print(class_names)
# print(class_names[0])
print(search_in_dict('n02110627', whole_dict))

wordnet_list = []
imagenet_list = []
for class_name in class_names:
    temp = search_in_dict(class_name, whole_dict)
    wordnet_list.append(temp)
    # if temp[1][0] == 'n01317541': # domestic_list
    #     wordnet_list.append(temp)
    #     imagenet_list.append(class_name)

print(wordnet_list)
# print(wordnet_list[101])
# print(wordnet_list[100])
# Now we have a domestic list

# print(wordnet_list[0])
# print(imagenet_list)
# print(len(imagenet_list))
np.save("interested_class_in.npy", imagenet_list, allow_pickle=True)
np.save("interested_class_wn.npy", wordnet_list, allow_pickle=True)