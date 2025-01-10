import numpy as np
import scipy.sparse as sp
import torch
import math
import os
import pandas as pd
from datetime import datetime
import gc
import torch.nn.functional as F
import random
from scipy.stats import geom
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.data import download_url
import requests
import tarfile
from torch_geometric.datasets import Amazon, Reddit

torch.set_printoptions(threshold=np.inf)
# device_GPU = torch.device('cuda')
device = torch.device('cpu')


def load_dataset(dummy_r, eps_x, eps_y, rate, train_rate, val_rate, layerx, layery, path="./node_level",dataset_name=""):

    print(f'Loading {dataset_name} dataset...')

    if dataset_name in ["cora", "citeseer"]:
        download_and_extract(path, dataset_name)    #The dataset is downloaded for the first time.

        dir = f"{path}/{dataset_name}/"
        idx_features_labels = np.genfromtxt(f"{dir}{dataset_name}.content", dtype=np.dtype(str))
        features = torch.Tensor(np.array(sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32).todense()))
        labels = encode_onehot(idx_features_labels[:, -1])
        labels = torch.Tensor(labels)
        labels_truth = torch.LongTensor(np.where(labels)[1])
        if dataset_name == "cora":
            idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
            idx_map = {j: i for i, j in enumerate(idx)}
            edges_unordered = np.genfromtxt(f"{dir}{dataset_name}.cites", dtype=np.int32)
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        elif dataset_name == "citeseer":
            idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
            idx_map = {j: i for i, j in enumerate(idx)}
            edges_unordered = np.genfromtxt(f"{dir}{dataset_name}.cites", dtype=np.dtype(str))
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.dtype(str)).reshape(edges_unordered.shape)
            edges = edges[~np.any(edges == 'None', axis=1)]
            edges = edges.astype(np.int32)
        edges = edges.tolist()
        edges = swap_and_append(edges)

    elif dataset_name in ["lastfm", "facebook"]:
        download_dataset(path, dataset_name)     #The dataset is downloaded for the first time.

        target_file = os.path.join(path, dataset_name, "target.csv")
        labels = pd.read_csv(target_file)['target']
        labels = torch.from_numpy(labels.to_numpy(dtype=np.int32))
        num_classes = labels.max().item() + 1
        labels = torch.eye(num_classes)[labels]
        labels_truth = torch.LongTensor(np.where(labels)[1])
        feature_file = os.path.join(path, dataset_name, "features.csv")
        x = pd.read_csv(feature_file).drop_duplicates()
        x = x.pivot(index='node_id', columns='feature_id', values='value').fillna(0)
        x = x.reindex(range(len(labels)), fill_value=0)
        features = torch.from_numpy(x.to_numpy()).float()
        edge_file = os.path.join(path, dataset_name, "edges.csv")
        edge_index = pd.read_csv(edge_file)
        edges = torch.from_numpy(edge_index.to_numpy(dtype=np.int32))
        edges = edges.tolist()
        edges = swap_and_append(edges)

    elif dataset_name in ["amazon", "reddit"]:
        data_file_root = Path(path) / dataset_name
        if dataset_name == "amazon":      #The dataset is downloaded for the first time.
            dataset = Amazon(root=data_file_root, name='Computers')[0]
        elif dataset_name == 'reddit':
            dataset = Reddit(root=data_file_root)[0]
            dataset = filter_class_by_count(dataset, remove_unlabeled=True)
        dataset = remove_isolated_nodes(dataset)
        labels = dataset.y
        classes = int(labels.max() + 1)
        labels = torch.LongTensor(labels)
        labels_truth = labels
        labels = F.one_hot(labels, num_classes=classes)
        features = dataset.x
        features = torch.FloatTensor(features)
        edge_index = dataset.edge_index
        edges = edge_index.t().tolist()
        edges = [[edge[1], edge[0]] for edge in edges]


    for i in range(features.shape[0]):
        data = [i, i]
        edges.append(data)
    output_list = []
    for pair in edges:
        output_list.append({pair[0]: pair[1]})
    dgree_max = max_dgree(output_list, rate)
    edges = reduce_dict_count(output_list, dgree_max)     #clipping
    edges = [[list(d.keys())[0], list(d.values())[0]] for d in edges]
    edges = dummy(dummy_r, edges, features)          #dummy
    degree = dgree_matrix(edges, features)
    edges = torch.tensor(edges)
    degree = degree.to(torch.float32)
    features = torch.nn.functional.normalize(features, p=2, dim=1)

    num = features.shape[0]
    numbers = list(range(num))
    random.shuffle(numbers)
    idx_train = numbers[:int(num * train_rate)]
    idx_val = numbers[int(num * train_rate):int(num * val_rate)]
    idx_test = numbers[int(num * val_rate):num]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    batch_size = 1000
    epsilon1 = eps_x
    epsilon2 = eps_y

    labels = labels.to(torch.float32)
    labels = zero_out_labels(labels, idx_test)

    sum, labels_noise = layer_agg(layerx=layerx, layery=layery, features=features, labels=labels, edges=edges, batch_size=batch_size, dgree_max=dgree_max, epsilon1=epsilon1, epsilon2=epsilon2, degree=degree)

    return sum, edges, labels_truth, labels_noise, degree, idx_train, idx_val, idx_test

def remove_isolated_nodes(data):
    "Remove isolated nodes."
    mask = data.y.new_zeros(data.num_nodes, dtype=bool)
    mask[data.edge_index[0]] = True
    mask[data.edge_index[1]] = True
    data = data.subgraph(mask)
    return data

def layer_agg(layerx,layery,features,labels,edges,batch_size,dgree_max,epsilon1,epsilon2,degree):

    "The number of aggregation layers for node representations and labels."

    current_sum = features
    for i in range(layerx):
        if i == 0:
            current_sum = process_gcn_layer(current_sum, edges, batch_size,dgree_max, epsilon=epsilon1)
        else:
            current_sum = process_gcn_layer(current_sum, edges, batch_size, dgree_max)
        if i < layerx-1:
            current_sum = current_sum.to(torch.float32)
            current_sum = current_sum * degree.view(-1, 1)
        print_time(f"Layer {i + 1} feature aggregation completed.")

        if i > 0:
            del previous_sum
            gc.collect()
        previous_sum = current_sum

    current_label = labels
    for i in range(layery):
        if i == 0:
            current_label = process_labels_layer(current_label, edges, batch_size, dgree_max, epsilon=epsilon2)
        else:
            current_label = process_labels_layer(current_label, edges, batch_size,dgree_max)
        if i < layery-1:
            current_label = current_label.to(torch.float32)
            current_label = current_label * degree.view(-1, 1)
        print_time(f"Layer {i + 1} label aggregation completed.")

        if i > 0:
            del previous_label
            gc.collect()
        previous_label = current_label
    labels_noise = current_label

    return current_sum,labels_noise


def print_time(message):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{message} Current time: {formatted_time}")

def process_gcn_layer(features, edges, batch_size, dgree_max, epsilon=None):

    "Node representation aggregation and adding distributed noise."

    list_of_dicts = key_value(features, edges)
    shares1, shares2 = split_two(list_of_dicts, batch_size)
    share = selectMPC_two(shares1, shares2, 3, features)

    sensitivity_gaussian = math.sqrt(2 * dgree_max)
    sensitivity_laplace = 2 * dgree_max

    if epsilon:
        shares_1 = AGG_features(share[0])
        shares_2 = AGG_features(share[1])
        shares_3 = AGG_features(share[2])

        'Gaussian'
        noise1 = gaussian_noise(shares_1, sensitivity_gaussian, epsilon)
        noise2 = gaussian_noise(shares_2, sensitivity_gaussian, epsilon)
        noise3 = gaussian_noise(shares_3, sensitivity_gaussian, epsilon)

        "Laplace"
        # beta = np.sqrt(np.random.beta(1, 3-1, features.shape))
        # noise1 = beta * laplace_noise(shares_1, sensitivity_laplace, epsilon)
        # noise2 = beta * laplace_noise(shares_2, sensitivity_laplace, epsilon)
        # noise3 = beta * laplace_noise(shares_3, sensitivity_laplace, epsilon)

        shares_1 = shares_1 + noise1
        shares_2 = shares_2 + noise2
        shares_3 = shares_3 + noise3

    else:
        shares_1 = AGG_features(share[0])
        shares_2 = AGG_features(share[1])
        shares_3 = AGG_features(share[2])

    features_sum = shares_1 + shares_2 + shares_3

    del list_of_dicts, shares1, shares2, shares_1, shares_2, shares_3, share
    gc.collect()
    return features_sum

def process_labels_layer(labels, edges, batch_size, dgree_max, epsilon=None):

    "Label aggregation and adding distributed noise."

    list_of_labels = key_value(labels, edges)
    labels1, labels2 = split_two(list_of_labels, batch_size)
    labels_share = selectMPC_two(labels1, labels2, 3, labels)

    sensitivity_gaussian = math.sqrt(2 * dgree_max )
    sensitivity_laplace = 2 * dgree_max

    if epsilon:
        labels_1 = AGG_features(labels_share[0])
        labels_2 = AGG_features(labels_share[1])
        labels_3 = AGG_features(labels_share[2])

        noise1 = gaussian_noise(labels_1, sensitivity_gaussian, epsilon)
        noise2 = gaussian_noise(labels_2, sensitivity_gaussian, epsilon)
        noise3 = gaussian_noise(labels_3, sensitivity_gaussian, epsilon)

        # beta = np.sqrt(np.random.beta(1, 3-1, labels.shape))
        # noise1 = beta * laplace_noise(labels_1, sensitivity_laplace, epsilon)
        # noise2 = beta * laplace_noise(labels_2, sensitivity_laplace, epsilon)
        # noise3 = beta * laplace_noise(labels_3, sensitivity_laplace, epsilon)

        labels_1 = labels_1 + noise1
        labels_2 = labels_2 + noise2
        labels_3 = labels_3 + noise3

    else:
        labels_1 = AGG_features(labels_share[0])
        labels_2 = AGG_features(labels_share[1])
        labels_3 = AGG_features(labels_share[2])

    labels_sum = labels_1 + labels_2 + labels_3

    del list_of_labels, labels1, labels2, labels_share, labels_1, labels_2, labels_3
    gc.collect()

    return labels_sum



def laplace_noise(data, sensitivity, epsilon):
    noise = np.random.laplace(0, sensitivity / epsilon, data.shape)
    return noise

def gaussian_noise(tensor1, sensitivity, epsilon):
    delta_f = sensitivity * math.sqrt(2 * math.log(1.25 / 1e-5)) / epsilon
    noise = torch.tensor(np.random.normal(0, delta_f/math.sqrt(3), tensor1.shape),dtype=torch.float32)
    return noise


def selectMPC_two(list1, list2, num_parts, features):

    "select computing parties (3)"

    combined_list = list1 + list2
    random.shuffle(combined_list)
    combined_array = np.array(combined_list)
    sublist_length = len(combined_array) // num_parts
    remaining = len(combined_array) % num_parts

    sublists = []
    start_index = 0

    for i in range(num_parts):
        end_index = start_index + sublist_length + (1 if i < remaining else 0)
        sublists.append(combined_array[start_index:end_index].tolist())
        start_index = end_index

    keys = set()
    for item in list1:
        keys.update(item.keys())

    for sublist in sublists:
        sublist_keys = set(item for d in sublist for item in d.keys())
        missing_keys = keys - sublist_keys
        for key in missing_keys:
            sublist.append({key: torch.zeros(features.shape[1])})

    return sublists

def max_dgree(list_of_dicts, rate):

    "Calculate the maximum degree."

    count_dict = {}
    for d in list_of_dicts:
        for key in d.keys():
            count_dict[key] = count_dict.get(key, 0) + 1

    count_dict = {k: v for k, v in sorted(count_dict.items(), key=lambda item: item[0])}
    dgree = list(count_dict.values())
    sorted_list = sorted(dgree)
    index = int(rate * len(sorted_list))
    result = sorted_list[index-1]
    print("max_dgree:",result)
    return result

def reduce_dict_count(input_list, dgree):

    "clipping"

    positions_dict = {}
    new_value = ''
    for index, dictionary in enumerate(input_list):
        key = list(dictionary.keys())[0]
        if key not in positions_dict:
            positions_dict[key] = []
        positions_dict[key].append(index)

    for key, positions in positions_dict.items():
        if len(positions) > dgree:
            for k in range(len(positions) - dgree):
                random_index = random.choice(positions)
                input_list[random_index] = new_value
                positions.remove(random_index)

    new_list = [x for x in input_list if x != '' and x is not None and len(str(x)) > 0]
    return new_list

def swap_and_append(arr):

    "Process undirected graphs."

    new_arr = []
    for array in arr:
        if len(array) == 2:
            new_array = [array[1], array[0]]
            new_arr.append(new_array)
        new_arr.append(array)
    return new_arr

def key_value(input_features, edges):

    "construct key-value"

    list_of_dicts = []
    for row in edges:
        new_dict = {}
        if row[1].item() == input_features.shape[0]:
            new_dict[row[0].item()] = torch.zeros(input_features.shape[1])
        else:
            new_dict[row[0].item()] = input_features[row[1]]
        list_of_dicts.append(new_dict)
    return list_of_dicts

def split_two(dicts,batch_size):

    "secret sharing (2)"

    shares_1 = []
    shares_2 = []
    for d in dicts:
        for key, value in d.items():
            shared_vectors = generate_TWO_vectors(value,batch_size)
            assert len(shared_vectors) == 2
            shares_1.append({key: shared_vectors[0]})
            shares_2.append({key: shared_vectors[1]})

    return shares_1,shares_2

def AGG_features(dicts):
    AGG_result = add_dicts(dicts)
    AGG_result = sorted(AGG_result, key=lambda x: list(x.keys())[0])
    values_list = process_dicts_list(AGG_result)
    return values_list

def add_dicts(dicts):
    result_dict = {}
    for dictionary in dicts:
        for key, value in dictionary.items():
            if key in result_dict:
                result_dict[key] = result_dict[key] + value
            else:
                result_dict[key] = value
    result_list = [{key: value} for key, value in result_dict.items()]
    return result_list

def process_dicts_list(dicts):
    values_list = [list(dictionary.values()) for dictionary in dicts]
    new_tensor_list = [tensor[0] for tensor in values_list]
    tensor_matrix = torch.stack(new_tensor_list)
    return tensor_matrix



def generate_TWO_vectors(tensor, batch_size):
    shape = tensor.shape
    tensor = tensor.to(torch.float32)
    batches = torch.split(tensor, batch_size)

    vector1_batches = []
    vector2_batches = []

    for batch in batches:
        vector_1 = torch.randn(batch.shape, device=batch.device)
        vector_2 = batch - vector_1

        vector1_batches.append(vector_1)
        vector2_batches.append(vector_2)

    final_vector1 = torch.cat(vector1_batches, dim=0)
    final_vector2 = torch.cat(vector2_batches, dim=0)

    return [final_vector1, final_vector2]


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def dummy(p,A,input_features):
    random_number = geom.rvs(p, size=input_features.shape[0])
    for i, num in enumerate(random_number):
        sub_array = [i, input_features.shape[0]]
        A.extend([sub_array] * num)
    return A

def dgree_matrix(edges,input_fea):
    output_list = []
    for pair in edges:
        output_list.append({pair[0]: pair[1]})
    count_dict = {}
    for d in output_list:
        for key in d.keys():
            count_dict[key] = count_dict.get(key, 0) + 1
    count_dict = sorted(count_dict.items(), key=lambda item: item[0])
    count_dict = torch.tensor([1.0 / v for _, v in count_dict])
    return count_dict

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int32))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)



def filter_class_by_count(data:Data, remove_unlabeled=False):
    min_count = 5000
    assert hasattr(data, 'y'), "The data object must have a 'y' attribute."

    y = F.one_hot(data.y)
    counts = y.sum(dim=0)  # Count occurrences of each class

    y = y[:, counts >= min_count]
    mask = y.sum(dim=1).bool()  # Nodes to keep based on the filtered classes

    data = data.clone()
    data.y = y.argmax(dim=1)

    if remove_unlabeled:
        data = data.subgraph(mask)
        print("Filtered data node count:", data.num_nodes)
    else:
        data.y[~mask] = -1

        if hasattr(data, 'train_mask'):
            data.train_mask = data.train_mask & mask
            data.val_mask = data.val_mask & mask
            data.test_mask = data.test_mask & mask

    return data



def download_url(url, save_dir):
    file_name = os.path.basename(url)
    save_path = os.path.join(save_dir, file_name)
    if os.path.exists(save_path):
        return
    print(f"Downloading {url} to {save_path}")

def download_dataset(root, name):
    url = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level'
    available_datasets = {
        'twitch',
        'facebook',
        'github',
        'deezer',
        'lastfm',
        'wikipedia'
    }
    name = name.lower()
    if name not in available_datasets:
        raise ValueError(f"Dataset '{name}' is not available. Choose from {available_datasets}.")

    raw_dir = os.path.join(root, name)
    os.makedirs(raw_dir, exist_ok=True)

    parts = ['edges', 'features', 'target']
    paths = {}
    for part in parts:
        file_name = f"{part}.csv"
        save_path = os.path.join(raw_dir, file_name)
        download_url(f"{url}/{name}/{file_name}", raw_dir)
        paths[file_name] = save_path

    return paths

def download_and_extract(target_dir, dataset_name):

    base_url = "https://linqs-data.soe.ucsc.edu/public/lbc"
    dataset_url = f"{base_url}/{dataset_name.lower()}.tgz"

    extract_path = os.path.join(target_dir, dataset_name.lower())

    if os.path.exists(extract_path):
        return

    os.makedirs(target_dir, exist_ok=True)

    archive_path = os.path.join(target_dir, f"{dataset_name.lower()}.tgz")

    print(f"Downloading {dataset_name} from {dataset_url}...")
    response = requests.get(dataset_url, stream=True)
    if response.status_code == 200:
        with open(archive_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
    else:
        raise Exception(
            f"Failed to download {dataset_name} from {dataset_url}. HTTP status code: {response.status_code}")

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)

    os.remove(archive_path)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def zero_out_labels(labels, idx_test):
    new_labels = labels.clone()
    new_labels[idx_test, :] = 0
    return new_labels
