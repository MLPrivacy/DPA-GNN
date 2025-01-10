import math
import os
from functools import partial
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import ToSparseTensor, RandomNodeSplit
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
from transforms import Normalize, FilterTopClass
# RR on edge
import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_sparse import SparseTensor
from pathlib import Path

def randomized_response_edge(adj_matrix, epsilon_e):
    p = torch.exp(torch.tensor(epsilon_e)) / (1 + torch.exp(torch.tensor(epsilon_e)))
    random_matrix = torch.rand(adj_matrix.size())
    flip_mask = random_matrix > p
    flipped_adj_matrix = torch.abs(adj_matrix - flip_mask.float())
    print('rr',torch.equal(flipped_adj_matrix,adj_matrix))
    return flipped_adj_matrix

def edge_index_to_adj_matrix(edge_index, num_nodes):
    # 将 edge_index 转换为邻接矩阵
    adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    return adj_matrix

def adj_matrix_to_edge_index(adj_matrix):
    # 将邻接矩阵转换回 edge_index
    edge_index, _ = dense_to_sparse(adj_matrix)
    return edge_index


class KarateClub(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level'
    available_datasets = {
        'twitch',
        'facebook',
        'github',
        'deezer',
        'lastfm',
        'wikipedia'
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in self.available_datasets

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['edges.csv', 'features.csv', 'target.csv']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for part in ['edges', 'features', 'target']:
            download_url(f'{self.url}/{self.name}/{part}.csv', self.raw_dir)

    def process(self):
        target_file = os.path.join(self.raw_dir, self.raw_file_names[2])
        y = pd.read_csv(target_file)['target']
        y = torch.from_numpy(y.to_numpy(dtype=int))
        num_nodes = len(y)

        edge_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        edge_index = pd.read_csv(edge_file)
        edge_index = torch.from_numpy(edge_index.to_numpy()).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges


        feature_file = os.path.join(self.raw_dir, self.raw_file_names[1])
        x = pd.read_csv(feature_file).drop_duplicates()
        x = x.pivot(index='node_id', columns='feature_id', values='value').fillna(0)
        x = x.reindex(range(num_nodes), fill_value=0)
        x = torch.from_numpy(x.to_numpy()).float()

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'KarateClub-{self.name}()'

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

def remove_isolated_nodes(data):
    mask = data.y.new_zeros(data.num_nodes, dtype=bool)
    mask[data.edge_index[0]] = True
    mask[data.edge_index[1]] = True
    data = data.subgraph(mask)
    return data


def load_amazon_computers():
    from torch_geometric.datasets import Amazon
    data_file_root = Path('./datasets') / 'amazon'
    dataset = Amazon(root=data_file_root, name='Computers')
    dataset = dataset[0]
    dataset = remove_isolated_nodes(dataset)
    # print(f'Datset: {dataset}:')
    return dataset

def load_reddit():
    from torch_geometric.datasets import Reddit
    data_file_root = Path('./datasets') / 'reddit'
    dataset = Reddit(root = data_file_root)
    dataset = dataset[0]
    dataset = filter_class_by_count(dataset, remove_unlabeled=True)
    dataset = remove_isolated_nodes(dataset)
    classes = int(dataset.y.max() + 1)
    # print(f'classes: {classes}:')
    # print(f'Datset: {dataset}:')
    return dataset


supported_datasets = {
    'cora': partial(Planetoid, name='cora'),
    'citeseer': partial(Planetoid, name='citeseer'),
    'pubmed': partial(Planetoid, name='pubmed'),
    'facebook': partial(KarateClub, name='facebook'),
    'lastfm': partial(KarateClub, name='lastfm'),
    'amazon': "",
    'reddit': ""
}


def load_dataset(
        dataset:        dict(help='name of the dataset', option='-d', choices=supported_datasets) = 'cora',
        data_dir:       dict(help='directory to store the dataset') = './datasets',
        data_range:     dict(help='min and max feature value', nargs=2, type=float) = (0, 1),
        val_ratio:      dict(help='fraction of nodes used for validation') = .25,
        test_ratio:     dict(help='fraction of nodes used for test') = .25,
        e_eps:     dict(help='epsilon_e') = 1.25
        ):


    if dataset in ['cora', 'citeseer', 'lastfm', 'facebook']:
        data = supported_datasets[dataset](root=os.path.join(data_dir, dataset))
        data = data[0]
    elif dataset in ['amazon']:
        data = load_amazon_computers()
    elif dataset in ['reddit']:
        data = load_reddit()


    epsilon_e = e_eps
    # print('eps_e',epsilon_e)
    n = len(data.y)
    print('n:', n)
    # put dense matrix on CPU
    adj = SparseTensor(row=data.edge_index[0].cpu(), col=data.edge_index[1].cpu(),
                       sparse_sizes=(n, n)).to_dense()
    p = 1.0 / (1.0 + math.exp(epsilon_e))
    # return 1 with probability p, but does not flip diagonal edges since no self loop allowed
    res = ((adj + torch.bernoulli(torch.full((n, n), p))) % 2).bool()
    res.fill_diagonal_(0)
    nonzero_idx = torch.nonzero(res)
    col = nonzero_idx[:, 0]
    row = nonzero_idx[:, 1]
    result = torch.stack((row, col))
    adjrr_t = SparseTensor(col=row, row=col, sparse_sizes=(n, n))

    data = RandomNodeSplit(split='train_rest', num_val=val_ratio, num_test=test_ratio)(data)
    # print('mid data\n',data,'\n',data.edge_index)
    data = ToSparseTensor()(data)
    print('to sparse\n',data,'\n',data.adj_t)
    data.adj_t = adjrr_t
    data.name = dataset
    data.num_classes = int(data.y.max().item()) + 1

    if data_range is not None:
        low, high = data_range
        data = Normalize(low, high)(data)
    print('data after rr\n',data,'\n',data.adj_t)
    # print('y',data.y.max().item())
    # print('loaded')
    return data
