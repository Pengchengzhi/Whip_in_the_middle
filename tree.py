## Maintainer: Jingyu Song #####
## Contact: jingyuso@umich.edu #####

from os import name
from cvxpy import constraints
from cvxpy.expressions.cvxtypes import problem
import numpy as np
from anytree import NodeMixin, RenderTree
from anytree import search, LevelOrderGroupIter, walker
import cvxpy as cp
from numpy.lib.function_base import append
from uniform_sampling import uniform_hypersphere
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# print(wordnet_list)

class TreeClass(NodeMixin):  # Add Node feature
    def __init__(self, name, dim_N = 2, length=1, parent=None, children=None, in_id='middle'):
        super(TreeClass, self).__init__()
        self.name = name
        self.length = length
        self.parent = parent
        
        if children:  # set children only if given
            self.children = children
        if in_id:
            self.in_id = in_id
        self.vector = None # unit vector of pointing direction
        self.dim_N = dim_N # dim of the feature vector

# my0 = MyClass('my0', 3, 4)
# my1 = MyClass('my1', 1, 0, parent=my0)
# my2 = MyClass('my2', 0, 2, parent=my0)
# print(my0.children[0].name)

def find_current_parent(treeNode: TreeClass, count):
    parent = treeNode
    if count <= 0:
        return parent
    else:
        parent = parent.children[count-1]
        return parent

def print_tree(tree):
    for pre, fill, node in RenderTree(tree):
        treestr = u"%s%s %s" % (pre, node.name, node.in_id)
        print(treestr.ljust(8), node.length, node.vector.reshape(node.dim_N))

def get_similar_nodes(leaf_nodes, search_node):
    parent_label = search_node.ancestors[-1].name
    search_name = search_node.name
    similar_nodes = []
    for node in leaf_nodes:
        if node.ancestors[-1].name == parent_label and node.name != search_name:
            similar_nodes.append(node)
    return similar_nodes

def find_closest_idx(hypersphere, search_node):
    search_space = [(i, point) for i, point in enumerate(hypersphere)]
    closest_dist = sys.float_info.max
    closest_idx = -1
    for i, point in search_space:
        dist = np.linalg.norm(np.array(point)-np.array(search_node.vector))
        if dist < closest_dist:
            closest_idx = i
            closest_dist = dist

    return closest_idx

def get_leaf_dict(tree, id_list):
    leaf_nodes = []
    for children in LevelOrderGroupIter(tree):
        for node in children:
            if node.is_leaf:
                leaf_nodes.append(node)
    label_dict = {}
    for node in leaf_nodes:
        label_dict[node.name] = node.vector
    return label_dict


def hypersphere_init_label(tree: TreeClass):
    # Get leaf nodes from tree
    leaf_nodes = []
    for children in LevelOrderGroupIter(tree):
        for node in children:
            if node.is_leaf:
                leaf_nodes.append(node)

    # Arbitrary point initialization on hypersphere
    hypersphere = uniform_hypersphere(tree.dim_N, len(leaf_nodes))
    # all_points = hypersphere
    for i, node in enumerate(leaf_nodes):
        node.vector = hypersphere[i]

    # Assign similar nodes to similar areas on hypersphere
    while len(leaf_nodes) != 0:
        # Take arbitrary node
        search_node = leaf_nodes.pop()
        # Assign node value on hypersphere
        search_node.vector = np.array(hypersphere.pop())
        # Find similar nodes 
        similar_nodes = get_similar_nodes(leaf_nodes, search_node)
        # For each similar node,
        for node in similar_nodes:
            # Find closest index and assign to similar node
            closest_idx = find_closest_idx(hypersphere, search_node)
            node.vector = np.array(hypersphere[closest_idx])
            hypersphere.pop(closest_idx)
            leaf_nodes.remove(node)
        
    

def tree_init_label(tree: TreeClass, length_list=[10,5,4,3,2,2,2,1,1]):
    # put all nodes in a list so we can do further processing
    # level_group = 
    # [['head'], ['n00015388', 'n00021939'], ['n01317541', 'n03575240'], ['n02084071', 'n03183080'], ['n02103406', 'n03800933'], 
    # ['n02103841', 'n02104523', 'n02107420', 'n02108089', 'n02108422', 'n02108254', 'n02108672', 'n02109047', 'n02109525', ...], 
    # ['n02104029', 'n02104365', 'n02106966', 'n02104882', 'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855', ...], 
    # ['n02107142', 'n02107312', 'n02110627', 'n02105056', 'n02105162', 'n03452741', 'n04515003', 'n02676566', 'n03272010', ...], 
    # ['n03228016', 'n04222847'], ['n02804610', 'n03838899', 'n04141076']]
    level_group = [[node for node in children] for children in LevelOrderGroupIter(tree)]
    dim_N = level_group[0][0].dim_N
    max_level = len(level_group)

    # do the first level initialization so we have
    if len(level_group[0][0].children) == 2:
        level_group[0][0].length = 0
        level_group[0][0].children[0].vector = np.ones((dim_N,1)) * np.sqrt(dim_N) / dim_N
        level_group[0][0].children[1].vector = -level_group[0][0].children[0].vector
    
    # inilialize label and node for each other
    # length_list = [10,5,4,3,2,2,2,1,1]
    for i in np.arange(1,max_level-1):
        query_list = level_group[i]
        for node in query_list:
            node.length = length_list[i-1]
            if len(node.children) > 0:
                # means not the end node
                if len(node.children) == 1:
                    # follow the parent vector:
                    node.children[0].vector = node.vector
                # inilialize label if children>=2
                else:
                    vectors = np.array(uniform_hypersphere(dim_N, len(node.children)))
                    #TODO: reject dot product > 0.5?
                    for j in range(len(node.children)):
                        node.children[j].vector = vectors[j,:]


def get_label_dict(tree, id_list):
    label_dict = {}
    w = walker.Walker()
    for id in id_list:
        walk_node = search.findall(tree, filter_=lambda node: node.name in (id))[0]
        walk_list = w.walk(tree, walk_node)[2]
        vec_label = np.zeros((walk_node.dim_N,1))
        for node in walk_list:
            vec_label += node.length* node.vector.reshape(vec_label.shape)
        label_dict[id] = vec_label
    
    return label_dict




# def obj_func(x, parent_vector):
#     result = []
#     for i in range(x.shape[1]):
#         result.append(cp.sum(cp.multiply(x[:,i], parent_vector)))
#         for j in range(i):
#             result.append(cp.sum(cp.multiply(x[:,i], x[:,j])))
#     return cp.sum(result)

# def label_solver(node_list):
#     # solve the label for each class
#     for node in node_list:
#         children = node.children
#         dim = children[0].dim_N
#         parent_vector = node.vector
#         if len(children) <= 1:
#             children[0].vector = parent_vector
#         else:
#             x = cp.Variable((dim, len(children)))
#             # Create constraints.
#             constraints = [cp.norm(x, axis=1) <= 1]

#             # Form objective.
            
#             obj = cp.Minimize(obj_func(x, parent_vector))
#             prob = cp.Problem(obj, constraints=constraints)
#             prob.solve()
#             print("status:",prob.status)
#             print("objective:",prob.value)
#             print("levels:",x.value)

#     return None


# for pre, fill, node in RenderTree(my0):
#     treestr = u"%s%s" % (pre, node.name)
#     print(treestr.ljust(8), node.length, node.width)


# print(len(wordnet_list))
# for i in range(len(wordnet_list)):
#     if type(wordnet_list[-1]) != tuple

def build_tree(dim_N = 3,length_list=[10,5,4,3,2,2,2,1,1], brocolli_flag=True):
    # length_list = [10,5,4,3,2,2,2,1,1] # for current dataset
    imagenet_list = np.load("interested_class_in.npy", allow_pickle=True)
    wordnet_list = np.load("interested_class_wn.npy", allow_pickle=True)
    #dim_N = dim_N
    current_node = None
    current_idx = None
    last_node = None

    
    # starting from head to final level
    # length_list = length_list

    tree = TreeClass('head',dim_N=dim_N)
    tree.vector = np.zeros(tree.dim_N) # inilialize the data for tree
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
                
                node_list.append(TreeClass(query_class, dim_N=dim_N, parent=current_parent[0])) # default attribute
            last_query = query_class
        node_list[-1].in_id = imagenet_list[i]
        if query_class != imagenet_list[i]:
            print('warning')
            #     continue
            # current_parent = search.findall(my0, filter_=lambda node: node.name in (query_class))[0]

    if brocolli_flag:
        tree_init_label(tree,length_list=length_list)
        label_dict = get_label_dict(tree,imagenet_list)
    else:
        hypersphere_init_label(tree)
        label_dict = get_leaf_dict(tree, imagenet_list)


    return label_dict


label_dict = build_tree(dim_N=3, brocolli_flag=False)
print(label_dict)
# leaf_nodes = []
# for children in LevelOrderGroupIter(tree):
#     for node in children:
#         if node.is_leaf:
#             leaf_nodes.append(node)

# x_data, y_data, z_data, category = [], [], [], []

# for node in leaf_nodes:
#     x_data.append(node.vector[0])
#     y_data.append(node.vector[1])
#     z_data.append(node.vector[2])
#     category.append(node.ancestors[-1].name)

# color_map = {}
# i = 0
# for cat in category:
#     if cat not in color_map:
#         color_map[cat] = i
#         i += 1

# color_list = []
# for cat in category:
#     color_list.append(color_map[cat])

# fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# ax = plt.axes(projection='3d')
# ax.scatter(x_data, y_data, z_data, c=color_list)
# plt.show()
# print('hi')
# print(search.findall(my0, filter_=lambda node: node.name in ('my3')))
