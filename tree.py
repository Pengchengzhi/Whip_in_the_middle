## Maintainer: Jingyu Song #####
## Contact: jingyuso@umich.edu #####

from os import name
import numpy as np
from anytree import NodeMixin, RenderTree
from anytree import search

imagenet_list = np.load("interested_class_in.npy", allow_pickle=True)
wordnet_list = np.load("interested_class_wn.npy", allow_pickle=True)

# print(wordnet_list)

class MyClass(NodeMixin):  # Add Node feature
    def __init__(self, name, length, width, parent=None, children=None, in_id='middle'):
        super(MyClass, self).__init__()
        self.name = name
        self.length = length
        self.width = width
        self.parent = parent
        if children:  # set children only if given
            self.children = children
        if in_id:
            self.in_id = in_id



# my0 = MyClass('my0', 3, 4)
# my1 = MyClass('my1', 1, 0, parent=my0)
# my2 = MyClass('my2', 0, 2, parent=my0)
# print(my0.children[0].name)

def find_current_parent(treeNode: MyClass, count):
    parent = treeNode
    if count <= 0:
        return parent
    else:
        parent = parent.children[count-1]
        return parent

def print_tree(tree):
    for pre, fill, node in RenderTree(tree):
        treestr = u"%s%s %s" % (pre, node.name, node.in_id)
        print(treestr.ljust(8), node.length, node.width)

# for pre, fill, node in RenderTree(my0):
#     treestr = u"%s%s" % (pre, node.name)
#     print(treestr.ljust(8), node.length, node.width)


# print(len(wordnet_list))
# for i in range(len(wordnet_list)):
#     if type(wordnet_list[-1]) != tuple

current_node = None
current_idx = None
last_node = None


tree = MyClass('head', 1, 1)
current_parent = tree
node_list = [tree]

for i in range(len(wordnet_list)):
    last_query = 'head'
    wordnet_list[i].pop() # the last item is not needed 
    for count, query_class in enumerate(wordnet_list[i]):
        # print(query_class)
        # print(i)
        query_class = query_class[0]
        # print(f'count is {count} and query class is {query_class}')
        if len(search.findall(tree, filter_=lambda node: node.name in (query_class))) == 0:
            # add new node
            # print(tree.children)
            # current_parent = find_current_parent(tree, count)
            # print(search.findall(tree, filter_=lambda node: node.name in (last_query)))
            current_parent = search.findall(tree, filter_=lambda node: node.name in (last_query))
            
            node_list.append(MyClass(query_class, 1, 1, parent=current_parent[0])) # default attribute
        last_query = query_class
    node_list[-1].in_id = imagenet_list[i]
    if query_class != imagenet_list[i]:
        print('warning')
        #     continue
        # current_parent = search.findall(my0, filter_=lambda node: node.name in (query_class))[0]

print_tree(tree)


# print(search.findall(my0, filter_=lambda node: node.name in ('my3')))

