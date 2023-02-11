import os
import javalang
from javalang.ast import Node
import numpy as np


def _name(node):
    return type(node).__name__


def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token


def dfsSearch1(children):
    if not isinstance(children, (str, Node, list, tuple)):
        return
    if isinstance(children, (str, Node)):
        if str(children) == '':
            return
        # ss = str(children)
        if str(children).startswith('"'):
            return
        if str(children).startswith("'"):
            return
        if str(children).startswith("/*"):
            return
        global num_nodes
        num_nodes += 1
        listt1.append(children)
        return
    for child in children:
        if isinstance(child, (str, Node, list, tuple)):
            dfsSearch1(child)


def _traverse_tree(root):
    global num_nodes
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),

        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        current_node_json = queue_json.pop(0)

        global listt
        global listt1
        listt1 = []
        dfsSearch1(current_node.children)
        children = listt1
        for child in children:
            child_json = {
                "node": get_token(child),
                "children": []
            }

            current_node_json['children'].append(child_json)
            if isinstance(child, (Node)):
                queue_json.append(child_json)
                queue.append(child)
    return root_json, num_nodes


def dfsDict(root, listtfinal):
    listtfinal.append(str(root['node']))
    if len(root['children']):
        pass
    else:
        return
    for dictt in root['children']:
        dfsDict(dictt, listtfinal)


def get_flist(dataset, split, cross):
    flist = []
    if '_' in cross:
        train, test = cross.split('_')
        train_root = f'../dataset/{dataset}/code{train}/'
        test_root = f'../dataset/{dataset}/code{test}/'

        train_data = np.load(f'../dataset/train_split/{dataset}/train_{split}.npy')
        train_ids = list(train_data.T[0])
        train_ids.extend(train_data.T[1])
        train_ids = set(train_ids)
        all_ids = [int(i.replace('.txt', '')) for i in os.listdir(train_root)]
        test_ids = set(all_ids).difference(train_ids)
        flist.extend([train_root + str(i) + '.txt' for i in train_ids])
        flist.extend([test_root + str(i) + '.txt' for i in test_ids])
    else:
        root_path = f'../dataset/{dataset}/code{cross}/'
        flist.extend([root_path + i for i in os.listdir(root_path)])

    return flist


def get_sentences(flist):
    sentences = []
    z = 0
    for l in flist:
        if not os.path.exists(l):
            continue
        ff = open(l, 'r')
        z += 1
        content = ff.read()
        ff.close()

        tree = javalang.parse.parse(content)
        sample, size = _traverse_tree(tree)
        listtfinal = []
        dfsDict(sample, listtfinal)
        sentences.append(listtfinal)
    return sentences


def get_train_test_data(dataset, split, cross):
    train_data = np.load(f'../dataset/train_split/{dataset}/train_{split}.npy')
    val_data = np.load(f'../dataset/train_split/{dataset}/val_{split}.npy')
    test_data = np.load(f'../dataset/train_split/{dataset}/test_{split}.npy')

    if '_' in cross:
        train, test = cross.split('_')
        train_root = f'../dataset/{dataset}/code{train}/'
        test_root = f'../dataset/{dataset}/code{test}/'
        train_data = [[f'{train_root}{d[0]}.txt', f'{train_root}{d[1]}.txt', str(d[2])] for d in train_data]
        val_data = [[f'{test_root}{d[0]}.txt', f'{test_root}{d[1]}.txt', str(d[2])] for d in val_data]
        test_data = [[f'{test_root}{d[0]}.txt', f'{test_root}{d[1]}.txt', str(d[2])] for d in test_data]
        return train_data, val_data, test_data
    else:
        root_path = f'../dataset/{dataset}/code{cross}/'
        train_data = [[f'{root_path}{d[0]}.txt', f'{root_path}{d[1]}.txt', str(d[2])] for d in train_data]
        val_data = [[f'{root_path}{d[0]}.txt', f'{root_path}{d[1]}.txt', str(d[2])] for d in val_data]
        test_data = [[f'{root_path}{d[0]}.txt', f'{root_path}{d[1]}.txt', str(d[2])] for d in test_data]
        return train_data, val_data, test_data


if __name__ == '__main__':
    flist = get_flist('gcj', 'random0', '1_2')
    print(flist[-10:])
    sentences = get_sentences(flist)
