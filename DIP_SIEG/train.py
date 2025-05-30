# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
import os, sys
# import os.path as osp
# import shutil
# import copy as cp
# from tqdm import tqdm
from functools import partial
# import psutil
import pdb
#
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, average_precision_score
# import scipy.sparse as ssp
import torch
# from torch import Tensor
from torch.nn import BCEWithLogitsLoss
# from torch.utils.data import DataLoader
# from torch.utils.data import IterableDataset

# import torch_geometric.transforms as T
from torch_geometric.data import Data
# from torch_geometric.data import DataLoader as PygDataLoader
# from torch_geometric.utils import to_networkx, to_undirected
#
# from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
#
# import warnings
# from scipy.sparse import SparseEfficiencyWarning
# warnings.simplefilter('ignore', SparseEfficiencyWarning)

from torch_geometric.datasets import Planetoid
from dataset import SEALDynamicDataset, SEALIterableDataset
from preprocess import preprocess, preprocess_full
# from graphormer.collator import collator
from utils import *
from models import *
# from timer_guard import TimerGuard

# import logging
# logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
#                     stream=sys.stdout,
#                     level=logging.INFO,
#                     datefmt='%Y-%m-%d %H:%M:%S')

# ngnn code
# import datetime

#
import dgl
# from dgl.data.utils import load_graphs, save_graphs
# from dgl.dataloading import GraphDataLoader
# from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset
# from tqdm import tqdm

# import ngnn_models
# import ngnn_utils
# import wp_utils
# import seal18_utils


import networkx as nx
import scipy.sparse as sp
from torch_geometric.utils import add_self_loops, negative_sampling, to_undirected, dense_to_sparse
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def preprocess_features(features):
    #print(features.sum())
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    #print(features.sum())
    return features
def new_load_data(dataset_name, splits_file_path=None):
    graph_adjacency_list_file_path = os.path.join(ROOT_DIR, 'dataset', 'new_data', dataset_name, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join(ROOT_DIR, 'dataset', 'new_data', dataset_name,
                                                            'out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if dataset_name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])

    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))

    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = preprocess_features(features)
    g = adj

    splits_file_path = os.path.join(ROOT_DIR, splits_file_path)
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    return g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels



# wp type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2none(v):
    if v.lower()=='none':
        return None
    else:
        return str(v)

# ngnn dataset
class SEALOGBLDataset(Dataset):
    def __init__(
        self,
        data_pyg,
        preprocess_fn,
        root,
        graph,
        split_edge,
        percent=100,
        split="train",
        ratio_per_hop=1.0,
        directed=False,
        dynamic=True,
    ) -> None:
        super().__init__()
        self.data_pyg = data_pyg
        self.preprocess_fn = preprocess_fn
        self.root = root
        self.graph = graph
        self.split = split
        self.split_edge = split_edge
        self.percent = percent
        self.ratio_per_hop = ratio_per_hop
        self.directed = directed
        self.dynamic = dynamic
        if "weights" in self.graph.edata:
            self.edge_weights = self.graph.edata["weights"]
        else:
            self.edge_weights = None
        if "feat" in self.graph.ndata:
            self.node_features = self.graph.ndata["feat"]
        else:
            self.node_features = None

        pos_edge, neg_edge = ngnn_utils.get_pos_neg_edges(
            self.split, self.split_edge, self.graph, self.percent
        )
        self.links = torch.cat([pos_edge, neg_edge], 0)  # [Np + Nn, 2] [1215518, 2]
        self.labels = np.array([1] * len(pos_edge) + [0] * len(neg_edge))  # [1215518]

        if not self.dynamic:
            self.g_list, tensor_dict = self.load_cached()
            self.labels = tensor_dict["y"]

        # compute degree from dataset_pyg
        if 'Graphormer' in args.model:
            if 'edge_weight' in data_pyg:
                edge_weight = data_pyg.edge_weight.view(-1)
            else:
                edge_weight = torch.ones(data_pyg.edge_index.size(1), dtype=int)
            import scipy.sparse as ssp
            A = ssp.csr_matrix(
                (edge_weight, (data_pyg.edge_index[0], data_pyg.edge_index[1])), 
                shape=(data_pyg.num_nodes, data_pyg.num_nodes))
            if directed:
                A_undirected = ssp.csr_matrix((np.concatenate([edge_weight, edge_weight]), (np.concatenate([data_pyg.edge_index[0], data_pyg.edge_index[1]]), np.concatenate([data_pyg.edge_index[1], data_pyg.edge_index[0]]))), shape=(data_pyg.num_nodes, data_pyg.num_nodes))
                degree_undirected = A_undirected.sum(axis=0).flatten().tolist()[0]
                degree_in = A.sum(axis=0).flatten().tolist()[0]
                degree_out = A.sum(axis=1).flatten().tolist()[0]
                self.degree = torch.Tensor([degree_undirected, degree_in, degree_out]).long()
            else:
                degree_undirected = A.sum(axis=0).flatten().tolist()[0]
                self.degree = torch.Tensor([degree_undirected]).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.dynamic:
            g, y = self.g_list[idx], self.labels[idx]
            x = None if "x" not in g.ndata else g.ndata["x"]
            w = None if "w" not in g.edata else g.eata["w"]
            return g, g.ndata["z"], x, w, y

        src, dst = self.links[idx][0].item(), self.links[idx][1].item()
        y = self.labels[idx]  # 1
        subg = ngnn_utils.k_hop_subgraph(
            src, dst, 1, self.graph, self.ratio_per_hop, self.directed
        )

        # Remove the link between src and dst.
        direct_links = [[], []]
        for s, t in [(0, 1), (1, 0)]:
            if subg.has_edges_between(s, t):
                direct_links[0].append(s)
                direct_links[1].append(t)
        if len(direct_links[0]):
            subg.remove_edges(subg.edge_ids(*direct_links))

        NIDs, EIDs = subg.ndata[dgl.NID], subg.edata[dgl.EID]  # [32] [72]

        z = ngnn_utils.drnl_node_labeling(subg.adj(scipy_fmt="csr"), 0, 1)  # [32]
        edge_weights = (
            self.edge_weights[EIDs] if self.edge_weights is not None else None
        )
        x = self.node_features[NIDs] if self.node_features is not None else None  # [32, 128]

        subg_aug = subg.add_self_loop()
        if edge_weights is not None:  # False
            edge_weights = torch.cat(
                [
                    edge_weights,
                    torch.ones(subg_aug.num_edges() - subg.num_edges()),
                ]
            )

        # compute structure from pyg data
        if 'Graphormer' in args.model:
            subg.x = x
            subg.z = z
            subg.node_id = NIDs
            subg.edge_index = torch.cat([subg.edges()[0].unsqueeze(0), subg.edges()[1].unsqueeze(0)], 0)
            if self.preprocess_fn is not None:
                self.preprocess_fn(subg, directed=self.directed, degree=self.degree)

        return subg_aug, z, x, edge_weights, y, subg

    @property
    def cached_name(self):
        return f"SEAL_{self.split}_{self.percent}%.pt"

    def process(self):
        g_list, labels = [], []
        self.dynamic = True
        for i in tqdm(range(len(self))):
            g, z, x, weights, y = self[i]
            g.ndata["z"] = z
            if x is not None:
                g.ndata["x"] = x
            if weights is not None:
                g.edata["w"] = weights
            g_list.append(g)
            labels.append(y)
        self.dynamic = False
        return g_list, {"y": torch.tensor(labels)}

    def load_cached(self):
        path = os.path.join(self.root, self.cached_name)
        if os.path.exists(path):
            return load_graphs(path)

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        g_list, labels = self.process()
        save_graphs(path, g_list, labels)
        return g_list, labels


def train(num_datas):
    model.train()

    y_pred, y_true = torch.zeros([num_datas]), torch.zeros([num_datas])
    start = 0
    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    for data in pbar:
        if args.ngnn_code:  # ngnn_code
            g, z, x, edge_weights, y = [
                item.to(device) if item is not None else None for item in data
            ]
            # g.to(device)没法把这些pairwise结构属性to(device)，只能手动一下
            if 'Graphormer' in args.model:
                g.attn_bias = g.attn_bias.to(device)
                g.edge_index = g.edge_index.to(device)
                g.x = g.x.to(device)
                g.z = g.z.to(device)
                if args.use_len_spd:
                    g.len_shortest_path = g.len_shortest_path.to(device)
                if args.use_num_spd:
                    g.num_shortest_path = g.num_shortest_path.to(device)
                if args.use_cnb_jac:
                    g.undir_jac = g.undir_jac.to(device)
                if args.use_cnb_aa:
                    g.undir_aa = g.undir_aa.to(device)
                if args.use_cnb_ra:
                    g.undir_ra = g.undir_ra.to(device)
                if args.use_degree:
                    g.undir_degree = g.undir_degree.to(device)
                    if directed:
                        g.in_degree = g.in_degree.to(device)
                        g.out_degree = g.out_degree.to(device)

            num_datas_in_batch = y.numel()  # sieg
            optimizer.zero_grad()
            logits = model(g, z, x, edge_weight=edge_weights)
            loss = BCEWithLogitsLoss()(logits.view(-1), y.to(torch.float))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * g.batch_size
            # sieg
            end = min(start+num_datas_in_batch, num_datas)
            y_pred[start:end] = logits.view(-1).cpu().detach()
            y_true[start:end] = y.view(-1).cpu().to(torch.float)
            start = end
        else:  # sieg_code
            data = data.to(device)
            num_datas_in_batch = data.y.numel()
            optimizer.zero_grad()
            x = data.x if args.use_feature else None
            edge_weight = data.edge_weight if args.use_edge_weight else None
            node_id = data.node_id if emb else None
            new_data = data.clone()
            new_data.x = x
            new_data.edge_weight = edge_weight
            new_data.node_id = node_id
            logits = model(new_data, args.dot_enh)
            loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
            loss.backward()
            optimizer.step()
            if args.scheduler: scheduler.step()
            total_loss += loss.item() * data.num_graphs
            end = min(start+num_datas_in_batch, num_datas)
            y_pred[start:end] = logits.view(-1).cpu().detach()
            y_true[start:end] = data.y.view(-1).cpu().to(torch.float)
            start = end
    #result['Confuse'] = confusion_matrix(y_true, y_pred)
    #result['ACC'] = accuracy_score(y_true, y_pred)
    #result['Precision'] = precision_score(y_true, y_pred)
    #result['Recall'] = recall_score(y_true, y_pred)
    #result['F1'] = f1_score(y_true, y_pred)
    result = {}
    result['AUC'] = roc_auc_score(y_true, y_pred)
    return total_loss / len(train_dataset), result


def test_model(model, loader, num_datas):
    model.eval()

    y_pred, y_true = torch.zeros([num_datas]), torch.zeros([num_datas])
    start = 0
    x_srcs, x_dsts = [], []
    for data in tqdm(loader, ncols=70):
        if args.ngnn_code:  # ngnn_code
            g, z, x, edge_weights, y = [
                item.to(device) if item is not None else None for item in data
            ]
            # g.to(device)没法把这些pairwise结构属性to(device)，只能手动一下
            if 'Graphormer' in args.model:
                g.attn_bias = g.attn_bias.to(device)
                g.edge_index = g.edge_index.to(device)
                g.x = g.x.to(device)
                g.z = g.z.to(device)
                if args.use_len_spd:
                    g.len_shortest_path = g.len_shortest_path.to(device)
                if args.use_num_spd:
                    g.num_shortest_path = g.num_shortest_path.to(device)
                if args.use_cnb_jac:
                    g.undir_jac = g.undir_jac.to(device)
                if args.use_cnb_aa:
                    g.undir_aa = g.undir_aa.to(device)
                if args.use_cnb_ra:
                    g.undir_ra = g.undir_ra.to(device)
                if args.use_degree:
                    g.undir_degree = g.undir_degree.to(device)
                    if directed:
                        g.in_degree = g.in_degree.to(device)
                        g.out_degree = g.out_degree.to(device)

            num_datas_in_batch = y.numel()  # sieg
            logits = model(g, z, x, edge_weight=edge_weights)
            # sieg
            end = min(start+num_datas_in_batch, num_datas)
            y_pred[start:end] = logits.view(-1).cpu()
            y_true[start:end] = y.view(-1).cpu().to(torch.float)
            start = end
        else:  # sieg_code
            data = data.to(device)
            num_datas_in_batch = data.y.numel()
            x = data.x if args.use_feature else None
            edge_weight = data.edge_weight if args.use_edge_weight else None
            node_id = data.node_id if emb else None
            new_data = data.clone()
            new_data.x = x
            new_data.edge_weight = edge_weight
            new_data.node_id = node_id
            logits = model(new_data, args.dot_enh)
            end = min(start+num_datas_in_batch, num_datas)
            y_pred[start:end] = logits.view(-1).cpu()
            y_true[start:end] = data.y.view(-1).cpu().to(torch.float)
            start = end

        if args.output_logits and loader == final_test_loader:
            _, center_indices = np.unique(data.batch.cpu().numpy(), return_index=True)
            x_srcs += data.node_id[center_indices].tolist()
            x_dsts += data.node_id[center_indices+1].tolist()
    if args.output_logits and loader == final_test_loader:
        logits_file = log_file.replace('log.txt', 'logits.txt')
        with open(logits_file, 'a') as f:
            print(f'x_src: (len:{len(x_srcs)})', file=f)
            print(x_srcs, file=f)
            print(f'x_dst: (len:{len(x_dsts)})', file=f)
            print(x_dsts, file=f)
            print(f'y_pred: (len:{len(y_pred.tolist())})', file=f)
            print(y_pred.tolist(), file=f)
            print(f'y_true: (len:{len(y_true.tolist())})', file=f)
            print(y_true.tolist(), file=f)

    pos_test_pred = y_pred[y_true==1]
    neg_test_pred = y_pred[y_true==0]
    return y_pred, y_true, pos_test_pred, neg_test_pred


def eval_model(**kwargs):
    eval_metric = kwargs["eval_metric"]
    if eval_metric == 'hits':
        pos_val_pred = kwargs["pos_val_pred"]
        neg_val_pred = kwargs["neg_val_pred"]
        pos_test_pred = kwargs["pos_test_pred"]
        neg_test_pred = kwargs["neg_test_pred"]
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif eval_metric == 'mrr':
        pos_val_pred = kwargs["pos_val_pred"]
        neg_val_pred = kwargs["neg_val_pred"]
        pos_test_pred = kwargs["pos_test_pred"]
        neg_test_pred = kwargs["neg_test_pred"]
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif eval_metric == 'auc':
        val_pred = kwargs["val_pred"]
        val_true = kwargs["val_true"]
        test_pred = kwargs["test_pred"]
        test_true = kwargs["test_true"]
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    return results

@torch.no_grad()
def test(eval_metric):
    model.eval()

    val_pred, val_true, pos_val_pred, neg_val_pred = test_model(model, val_loader, len(val_dataset))

    test_pred, test_true, pos_test_pred, neg_test_pred = test_model(model, test_loader, len(test_dataset))

    result = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                      val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric=eval_metric)
    if eval_metric != 'auc':
        result_auc = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                          val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric='auc')
        for key in result_auc.keys():
            result[key] = result_auc[key]
    return result

@torch.no_grad()
def final_test(eval_metric):
    model.eval()

    val_pred, val_true, pos_val_pred, neg_val_pred = test_model(model, final_val_loader, len(final_val_dataset))

    test_pred, test_true, pos_test_pred, neg_test_pred = test_model(model, final_test_loader, len(final_test_dataset))

    result = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                      val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric=eval_metric)
    if eval_metric != 'auc':
        result_auc = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                          val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric='auc')
        for key in result_auc.keys():
            result[key] = result_auc[key]
    return result

@torch.no_grad()
def test_multiple_models_origin(models, eval_metric):
    num_models = len(models)
    for m in models:
        m.eval()

    y_preds, y_trues = [[] for _ in range(num_models)], [[] for _ in range(num_models)]
    for data in tqdm(val_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        for i, model in enumerate(models):
            logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_preds[i].append(logits.view(-1).cpu())
            y_trues[i].append(data.y.view(-1).cpu().to(torch.float))
    val_preds = [torch.cat(y_preds[i]) for i in range(num_models)]
    val_trues = [torch.cat(y_trues[i]) for i in range(num_models)]
    pos_val_preds = [val_preds[i][val_trues[i]==1] for i in range(num_models)]
    neg_val_preds = [val_preds[i][val_trues[i]==0] for i in range(num_models)]
    mem = psutil.virtual_memory()
    print(f' after val - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')

    y_preds, y_trues = [[] for _ in range(num_models)], [[] for _ in range(num_models)]
    for data in tqdm(test_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        for i, model in enumerate(models):
            logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_preds[i].append(logits.view(-1).cpu())
            y_trues[i].append(data.y.view(-1).cpu().to(torch.float))
    test_preds = [torch.cat(y_preds[i]) for i in range(num_models)]
    test_trues = [torch.cat(y_trues[i]) for i in range(num_models)]
    pos_test_preds = [test_preds[i][test_trues[i]==1] for i in range(num_models)]
    neg_test_preds = [test_preds[i][test_trues[i]==0] for i in range(num_models)]

    mem = psutil.virtual_memory()
    print(f' after test - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')
    results = eval_multiple_models(num_models,
                                pos_val_preds=pos_val_preds, neg_val_preds=neg_val_preds, pos_test_preds=pos_test_preds, neg_test_preds=neg_test_preds,
                                val_preds=val_preds, val_trues=val_trues, test_preds=test_preds, test_trues=test_trues, eval_metric=eval_metric)
    if eval_metric != 'auc':
        results_auc = eval_multiple_models(num_models,
                                    pos_val_preds=pos_val_preds, neg_val_preds=neg_val_preds, pos_test_preds=pos_test_preds, neg_test_preds=neg_test_preds,
                                    val_preds=val_preds, val_trues=val_trues, test_preds=test_preds, test_trues=test_trues, eval_metric='auc')
        for i in range(num_models):
            for key in results_auc[i].keys():
                results[i][key] = results_auc[i][key]

    return results


@torch.no_grad()
def test_multiple_models(models, loader, num_datas):
    num_models = len(models)
    for m in models:
        m.eval()

    y_preds, y_trues = [torch.zeros([num_datas]) for _ in range(num_models)], [torch.zeros([num_datas]) for _ in range(num_models)]
    start = 0
    for data in tqdm(loader, ncols=70):
        data = data.to(device)
        num_datas_in_batch = data.y.numel()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        new_data = data.clone()
        new_data.x = x
        new_data.edge_weight = edge_weight
        new_data.node_id = node_id
        end = min(start+num_datas_in_batch, num_datas)
        for i, model in enumerate(models):
            logits = model(new_data)
            y_preds[i][start:end] = logits.view(-1).cpu()
            y_trues[i][start:end] = data.y.view(-1).cpu().to(torch.float)
        start = end
    pos_test_preds = [y_preds[i][y_trues[i]==1] for i in range(num_models)]
    neg_test_preds = [y_preds[i][y_trues[i]==0] for i in range(num_models)]

    mem = psutil.virtual_memory()
    print(f'       max - {mem.percent:7} - {mem.free/1024**3:12.2f} - {mem.available/1024**3:13.2f} - {mem.used/1024**3:12.2f}')
    return y_preds, y_trues, pos_test_preds, neg_test_preds


def eval_multiple_models(num_models, **kwargs):
    eval_metric = kwargs["eval_metric"]
    Results = []
    for i in range(num_models):
        if eval_metric == 'hits':
            pos_val_preds = kwargs["pos_val_preds"]
            neg_val_preds = kwargs["neg_val_preds"]
            pos_test_preds = kwargs["pos_test_preds"]
            neg_test_preds = kwargs["neg_test_preds"]
            Results.append(evaluate_hits(pos_val_preds[i], neg_val_preds[i], pos_test_preds[i], neg_test_preds[i]))
        elif eval_metric == 'mrr':
            pos_val_preds = kwargs["pos_val_preds"]
            neg_val_preds = kwargs["neg_val_preds"]
            pos_test_preds = kwargs["pos_test_preds"]
            neg_test_preds = kwargs["neg_test_preds"]
            Results.append(evaluate_mrr(pos_val_preds[i], neg_val_preds[i], pos_test_preds[i], neg_test_preds[i]))
        elif eval_metric == 'auc':
            val_preds = kwargs["val_preds"]
            val_trues = kwargs["val_trues"]
            test_preds = kwargs["test_preds"]
            test_trues = kwargs["test_trues"]
            Results.append(evaluate_auc(val_preds[i], val_trues[i], test_preds[i], test_trues[i]))

    return Results


def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in args.eval_hits_K:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results


def evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (valid_mrr, test_mrr)

    return results


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    valid_ap = average_precision_score(val_true, val_pred)
    test_ap = average_precision_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)
    results['AP'] = (valid_ap, test_ap)
    #results['Confuse'] = (confusion_matrix(val_true, val_pred), confusion_matrix(test_true, test_pred))
    #results['ACC'] = (accuracy_score(val_true, val_pred), accuracy_score(test_true, test_pred))
    #results['Precision'] = (precision_score(val_true, val_pred), precision_score(test_true, test_pred))
    #results['Recall'] = (recall_score(val_true, val_pred), recall_score(test_true, test_pred))
    #results['F1'] = (f1_score(val_true, val_pred), f1_score(test_true, test_pred))

    return results

# Data settings
parser = argparse.ArgumentParser(description='OGBL (SEAL)')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--cmd_time', type=str, default='ignore_time')
parser.add_argument('--root', type=str, default='dataset',
                    help="root of dataset")
parser.add_argument('--dataset', type=str, default='ogbl-collab')
parser.add_argument('--fast_split', action='store_true',
                    help="for large custom datasets (not OGB), do a fast data split")
# GNN settings
parser.add_argument('--model', type=str, default='DGCNNGraphormer_noNeigFeat')#DGCNN
parser.add_argument('--sortpool_k', type=float, default=0.6)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=128)#32
parser.add_argument('--mlp_hidden_channels', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=64)#32
# Subgraph extraction settings
parser.add_argument('--sample_type', type=int, default=0)
parser.add_argument('--num_hops', type=int, default=1)
parser.add_argument('--ratio_per_hop', type=float, default=1.0)
parser.add_argument('--max_nodes_per_hop', type=int, default=None)#changed
parser.add_argument('--node_label', type=str, default='drnl', 
                    help="which specific labeling trick to use")
parser.add_argument('--use_feature', type=bool, default = True,
                    help="whether to use raw node features as GNN input")#changed
parser.add_argument('--use_feature_GT', type=bool, default = True,
                    help="whether to use raw node features as GNN input")#changed
parser.add_argument('--use_edge_weight', type=bool, default = True,
                    help="whether to consider edge weight in GNN")#changed
parser.add_argument('--use_rpe', action='store_true', help="whether to use RPE as GNN input")
parser.add_argument('--replacement', action='store_true', help="whether to enable replacement sampleing in random walk")
parser.add_argument('--trackback', action='store_true', help="whether to enabale trackback path searching in random walk")
parser.add_argument('--num_walk', type=int, default=200, help='total number of random walks')
parser.add_argument('--num_step', type=int, default=4, help='total steps of random walk')
parser.add_argument('--rpe_hidden_dim', type=int, default=16, help='dimension of RPE embedding')
parser.add_argument('--gravity_type', type=int, default=0)
parser.add_argument('--readout_type', type=int, default=0)
# Training settings
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.00001)#0.0001
parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--epochs', type=int, default=50)#50
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--train_percent', type=float, default=100)#2
parser.add_argument('--val_percent', type=float, default=100)#1
parser.add_argument('--test_percent', type=float, default=100)#1
parser.add_argument('--final_val_percent', type=float, default=100)
parser.add_argument('--final_test_percent', type=float, default=100)
parser.add_argument('--dynamic_train', action='store_true', 
                    help="dynamically extract enclosing subgraphs on the fly")
parser.add_argument('--dynamic_val', action='store_true')
parser.add_argument('--dynamic_test', action='store_true')
parser.add_argument('--slice_type', type=int, default=0,
                    help="type of saving sampled subgraph in disk")
parser.add_argument('--num_workers', type=int, default=0,
                    help="number of workers for dynamic mode; 0 if not dynamic")#16
parser.add_argument('--train_node_embedding', action='store_true',
                    help="also train free-parameter node embeddings together with GNN")
parser.add_argument('--dont_z_emb_agg', action='store_true')
parser.add_argument('--pretrained_node_embedding', type=str, default=None, 
                    help="load pretrained node embeddings as additional node features")
# Testing settings
parser.add_argument('--use_valedges_as_input', action='store_true')
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--data_appendix', type=str, default='', 
                    help="an appendix to the data directory")
parser.add_argument('--save_appendix', type=str, default='', 
                    help="an appendix to the save directory")
parser.add_argument('--keep_old', action='store_true', 
                    help="do not overwrite old files in the save directory")
parser.add_argument('--continue_from', type=int, nargs='*', default=None, 
                    help="from which run and epoch checkpoint to continue training")
parser.add_argument('--part_continue_from', type=int, nargs='*', default=None, 
                    help="from which run and epoch checkpoint to continue training")
parser.add_argument('--output_logits', action='store_true')
parser.add_argument('--only_test', action='store_true', 
                    help="only test without training")
parser.add_argument('--only_final_test', action='store_true', 
                    help="only final test without training")
parser.add_argument('--test_multiple_models', type=str, nargs='+', default=[], 
                    help="test multiple models together")
parser.add_argument('--use_heuristic', type=str, default=None,
                    help="test a link prediction heuristic (CN or AA)")
parser.add_argument('--num_heads', type=int, default=8)#32
parser.add_argument('--use_len_spd', type=bool, default = True)#changed
parser.add_argument('--use_num_spd', type=bool, default = True)#changed
parser.add_argument('--use_cnb_jac', type=bool, default = True)#changed
parser.add_argument('--use_cnb_aa', type=bool, default = True)#changed
parser.add_argument('--use_cnb_ra', type=bool, default = True)#changed
parser.add_argument('--use_degree', action='store_true', default=False)
parser.add_argument('--grpe_cross', type=bool, default = True)#changed
parser.add_argument('--use_ignn', action='store_true', default=False)
parser.add_argument('--mul_bias', action='store_true', default=False,
                    help="add bias to attention if true else multiple")
parser.add_argument('--max_z', type=int, default=1000)  # set a large max_z so that every z has embeddings to look up
# ngnn_args
parser.add_argument('--ngnn_code', action='store_true', default=False)
parser.add_argument('--use_full_graphormer', action='store_true', default=False)

parser.add_argument(
    "--ngnn_type",
    type=str,
    default="all",
    choices=["none", "input", "hidden", "output", "all"],
    help="You can set this value from 'none', 'input', 'hidden' or 'all' " \
            "to apply NGNN to different GNN layers.",
)
parser.add_argument(
    "--num_ngnn_layers", type=int, default=2, choices=[1, 2]
)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument(
    "--test_topk",
    type=int,
    default=1,
    help="select best k models for full validation/test each run.",
)
parser.add_argument(
    "--eval_hits_K",
    type=int,
    nargs="*",
    default=[10],
    help="hits@K for each eval step; " \
            "only available for datasets with hits@xx as the eval metric",
)

# wp_args
parser.add_argument('--wp_code', action='store_true', default=False)
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
parser.add_argument('--val-ratio', type=float, default=0.05,
                    help='ratio of validation links. If using the splitted data from SEAL,\
                     it is the ratio on the observed links, othewise, it is the ratio on the whole links.')
parser.add_argument('--practical-neg-sample', type=bool, default = False,
                    help='only see the train positive edges when sampling negative')
parser.add_argument('--wp-seed', type=int, default=1)
parser.add_argument('--drnl', type=str2bool, default=False,
                    help='whether to use drnl labeling')
parser.add_argument('--data-split-num',type=str, default='10',
                    help='If use-splitted is true, choose one of splitted data')
parser.add_argument('--observe-val-and-injection', type=str2bool, default = True,
                    help='whether to contain the validation set in the observed graph and apply injection trick')
parser.add_argument('--init-attribute', type=str2none, default='ones',
                    help='initial attribute for graphs without node attributes\
                    , options: n2v, one_hot, spc, ones, zeros, None')
parser.add_argument('--init-representation', type=str2none, default= None,
                    help='options: gic, vgae, argva, None')
parser.add_argument('--use-splitted', type=str2bool, default=True,
                    help='use the pre-splitted train/test data,\
                     if False, then make a random division')
parser.add_argument('--embedding-dim', type=int, default= 32,
                    help='Dimension of the initial node representation, default: 32)')

# seal18_args
parser.add_argument('--seal18_code', action='store_true', default=False)
parser.add_argument('--train-name', type=str, default=None)
parser.add_argument('--test-name', type=str, default=None)
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')

parser.add_argument('--dot_enh', type=int, default=0,
                        help='whether enhancing with dot product')

args = parser.parse_args()

if args.dot_enh == 1:
    args.use_degree = True


if (args.dataset in ('ogbl-vessel','ogbl-citation2','ogbl-ppa','Ecoli','PB','pubmed','chameleon')) and (args.max_nodes_per_hop==None):
    args.max_nodes_per_hop=100

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
if args.seed is not None: seed_torch(args.seed)

# if args.max_nodes_per_hop is not None and len(args.max_nodes_per_hop) == 1:
#     args.max_nodes_per_hop = args.max_nodes_per_hop[0]
# if args.max_nodes_per_hop is not None:
#     args.max_nodes_per_hop = None if args.max_nodes_per_hop < 0 else args.max_nodes_per_hop
if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
if args.data_appendix == '':
    args.data_appendix = '_h{}_{}_rph{}'.format(
        args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))
    if args.max_nodes_per_hop is not None:
        args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)
    if args.use_valedges_as_input:
        args.data_appendix += '_uvai'
if args.use_heuristic is not None:
    args.runs = 1


if args.dataset.startswith('ogbl-citation'):
    args.eval_metric = 'mrr'
    directed = True
elif args.dataset.startswith('ogbl-vessel'):
    args.eval_metric = 'auc'
    directed = False
elif args.dataset.startswith('ogbl'):
    args.eval_metric = 'hits'
    directed = False
else:  # assume other datasets are undirected
    args.eval_metric = 'auc'
    directed = False

if args.dataset.startswith('ogbl'):
    evaluator = Evaluator(name=args.dataset)
    
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:0')
device = 'cpu' if args.device == -1 or not torch.cuda.is_available() else f'cuda:{args.device}'
device = torch.device(device)

if args.use_heuristic:#default None
    # Test link prediction heuristics.
    num_nodes = data.num_nodes
    if 'edge_weight' in data and args.use_edge_weight:
        edge_weight = data.edge_weight.view(-1)
    else:
        edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

    A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])), 
                       shape=(num_nodes, num_nodes))

    pos_val_edge, neg_val_edge = get_pos_neg_edges('valid', split_edge, 
                                                   data.edge_index, 
                                                   data.num_nodes)
    pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge, 
                                                     data.edge_index, 
                                                     data.num_nodes)
    if directed:
        cn_types = ['undirected', 'in', 'out', 's2o', 'o2s']
    else:
        cn_types = ['in']

    pos_val_pred, pos_val_edge = eval(args.use_heuristic)(A, pos_val_edge, cn_types=cn_types)
    neg_val_pred, neg_val_edge = eval(args.use_heuristic)(A, neg_val_edge, cn_types=cn_types)
    pos_test_pred, pos_test_edge = eval(args.use_heuristic)(A, pos_test_edge, cn_types=cn_types)
    neg_test_pred, neg_test_edge = eval(args.use_heuristic)(A, neg_test_edge, cn_types=cn_types)

    for idx_type in range(len(cn_types)):
        cn_type = cn_types[idx_type]
        if args.eval_metric == 'hits':
            results = evaluate_hits(pos_val_pred[idx_type], neg_val_pred[idx_type], pos_test_pred[idx_type], neg_test_pred[idx_type])
        elif args.eval_metric == 'mrr':
            results = evaluate_mrr(pos_val_pred[idx_type], neg_val_pred[idx_type], pos_test_pred[idx_type], neg_test_pred[idx_type])
        elif args.eval_metric == 'auc':
            val_pred = torch.cat([pos_val_pred[idx_type], neg_val_pred[idx_type]])
            val_true = torch.cat([torch.ones(pos_val_pred[idx_type].size(0), dtype=int), 
                                  torch.zeros(neg_val_pred[idx_type].size(0), dtype=int)])
            test_pred = torch.cat([pos_test_pred[idx_type], neg_test_pred[idx_type]])
            test_true = torch.cat([torch.ones(pos_test_pred[idx_type].size(0), dtype=int), 
                                  torch.zeros(neg_test_pred[idx_type].size(0), dtype=int)])
            results = evaluate_auc(val_pred, val_true, test_pred, test_true)

        # for key, result in results.items():
        #     loggers[key].reset()
        #     loggers[key].add_result(0, result)
        # for key in loggers.keys():
        #     print(cn_type)
        #     print(key)
        #     loggers[key].print_statistics()
        #     with open(log_file, 'a') as f:
        #         print(cn_type, file=f)
        #         print(key, file=f)
        #         loggers[key].print_statistics(f=f)
    exit()
preprocess_func = preprocess_full if args.use_full_graphormer else preprocess#default false
preprocess_fn = partial(preprocess_func,
                        grpe_cross=args.grpe_cross,
                        use_len_spd=args.use_len_spd,
                        use_num_spd=args.use_num_spd,
                        use_cnb_jac=args.use_cnb_jac,
                        use_cnb_aa=args.use_cnb_aa,
                        use_cnb_ra=args.use_cnb_ra,
                        use_degree=args.use_degree,
                        gravity_type=args.gravity_type,
                )  if args.model.find('Graphormer') != -1 else None

if not args.ngnn_code:#default false
    if args.wp_code:
        if args.dataset in ('cora', 'citeseer','pubmed'):
            args.use_splitted = False
            args.practical_neg_sample = True
            args.observe_val_and_injection = False
            args.init_attribute=None
        if (args.dataset in ('Ecoli','PB','pubmed')) and (args.max_nodes_per_hop==None):
            args.max_nodes_per_hop=100
        if args.dataset=='Power':
            args.num_hops=3

        data = wp_utils.load_splitted_data(args)
        # data = wp_utils.load_unsplitted_data(args)
        data_observed, feature_results = wp_utils.set_init_attribute_representation(data, args)
        # PB:
        # data: Data(test_neg=[2, 1671], test_pos=[2, 1671], train_neg=[2, 14291], train_pos=[2, 14291], val_neg=[2, 752], val_pos=[2, 752])
        # data_observed: Data(edge_index=[2, 30086], x=[1223, 32])
        # edge_index: train_pos&val_pos bidirectional edge, x: x = torch.ones(data.num_nodes,args.embedding_dim).float()
        # feature_results: None
        # data.num_edges: None, data.num_features: 0, data.num_nodes: tensor(1223), data.num_node_features: 0
        data.edge_index = torch.cat([data.train_pos, data.val_pos, data.test_pos], dim=1)
        data.x = data_observed.x
        data.num_nodes = data.num_nodes.item()
        split_edge = {'train': {'edge': data.train_pos.t(), 'edge_neg': data.train_neg.t()},
                      'valid': {'edge': data.val_pos.t(), 'edge_neg': data.val_neg.t()},
                      'test': {'edge': data.test_pos.t(), 'edge_neg': data.test_neg.t()}}
        # directed现在是默认False，确认一下

    elif args.seal18_code:
        train_pos, train_neg, test_pos, test_neg = seal18_utils.seal18_prepare_data(args)
        train_pos = torch.cat([torch.from_numpy(train_pos[0]).unsqueeze(0),
                               torch.from_numpy(train_pos[1]).unsqueeze(0)], dim=0)
        train_neg = torch.cat([torch.Tensor(train_neg[0]).unsqueeze(0),
                               torch.Tensor(train_neg[1]).unsqueeze(0)], dim=0)
        test_pos = torch.cat([torch.from_numpy(test_pos[0]).unsqueeze(0),
                               torch.from_numpy(test_pos[1]).unsqueeze(0)], dim=0)
        test_neg = torch.cat([torch.Tensor(test_neg[0]).unsqueeze(0),
                               torch.Tensor(test_neg[1]).unsqueeze(0)], dim=0)
        val_num = int(0.1 * train_pos.shape[1])  # seal18是采好子图再拆分验证集
        val_pos = train_pos[:, :val_num]
        train_pos = train_pos[:, val_num:]
        val_neg = train_neg[:, :val_num]
        train_neg = train_neg[:, val_num:]
        # print(train_pos.shape, train_neg.shape, val_pos.shape, val_neg.shape, test_pos.shape, test_neg.shape)
        data = Data()
        data.edge_index = torch.cat([train_pos, val_pos, test_pos], dim=1)
        data.x = None
        data.num_nodes = (max(torch.max(train_pos), torch.max(val_pos), torch.max(test_pos)) + 1).item()
        split_edge = {'train': {'edge': train_pos.t(), 'edge_neg': train_neg.t()},
                      'valid': {'edge': val_pos.t(), 'edge_neg': val_neg.t()},
                      'test': {'edge': test_pos.t(), 'edge_neg': test_neg.t()}}
    else:  # sieg_code
        if args.dataset.startswith('ogbl'):
            dataset = PygLinkPropPredDataset(name=args.dataset, root=args.root)
            split_edge = dataset.get_edge_split()
            data = dataset[0]
        elif args.dataset in('cora', 'citeseer', 'pubmed'):
            dataset = Planetoid(root='dataset/'+args.dataset, name=args.dataset)
            split_edge = do_edge_split(dataset[0], args.fast_split, args.val_ratio, args.test_ratio)
            data = dataset[0]
            data.edge_index = split_edge['train']['edge'].t()
            root = f'{dataset.root}/sieg'
            # PubMed:
            # Data(edge_index=[2, 75352], test_mask=[19717], train_mask=[19717], val_mask=[19717], x=[19717, 500], y=[19717])
            # whole graph edges: 75352; whole graph num_nodes: 19717
            # data.num_edges: 75352, data.num_features: 500, data.num_nodes: 19717, data.num_node_features: 500
            # split_edge: {'train': {'edge': tensor([[0, 1378],...]), 'edge_neg': tensor}, 'valid': {'edge': tensor, 'edge_neg': tensor}, 'test': {'edge': tensor, 'edge_neg': tensor}}
        else:
            split_index = str(0)
            splitstr = ROOT_DIR + '/dataset/new_data_splits/' + args.dataset + '_split_0.6_0.2_' + split_index + '.npz'
            g, features, labels, _, _, _, num_features, num_labels = new_load_data(args.dataset, splitstr)
            A = g.toarray()
            edge_index, _ = dense_to_sparse(torch.tensor(A))
            dataset = Data(edge_index=edge_index, x=features.to(torch.float))
            root = f'{ROOT_DIR}/dataset/new_data_splits/sieg'
            split_edge = do_edge_split(dataset, args.fast_split, args.val_ratio, args.test_ratio)
            data = dataset
            data.edge_index = split_edge['train']['edge'].t()
    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        if not directed:
            val_edge_index = to_undirected(val_edge_index)
        data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=int)
        data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)

    print(get_dict_info(split_edge))
    print(f'data {data}')

    if args.wp_code:
        wp_root = os.path.join(root_dir, 'wp_data/splitted/{}'.format(args.dataset))
        path = wp_root + '_seal{}'.format(args.data_appendix)
    elif args.seal18_code:
        seal18_root = os.path.join(root_dir, 'seal18_data/{}'.format(args.dataset))
        path = seal18_root + '_seal{}'.format(args.data_appendix)
    else:
        path = root + '_seal{}'.format(args.data_appendix)  # sieg
    print(f'path {path}')
    use_coalesce = True if args.dataset == 'ogbl-collab' else False
    #if not args.dynamic_train and not args.dynamic_val and not args.dynamic_test:
    #    args.num_workers = 0

    dataset_class = 'SEALDynamicDataset' if args.dynamic_train else 'SEALIterableDataset'#eval trans string to function#default false
    train_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=args.train_percent, 
        split='train', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=directed, 
        sample_type=args.sample_type,
        shuffle=True,
        slice_type=args.slice_type,
        use_rpe=args.use_rpe,
        replacement=args.replacement,
        trackback=args.trackback,
        num_walk=args.num_walk,
        num_step=args.num_step,
        preprocess_fn=preprocess_fn,
    )


    dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALIterableDataset'#default false
    val_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=args.val_percent, 
        split='valid', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=directed, 
        sample_type=args.sample_type,
        slice_type=args.slice_type,
        use_rpe=args.use_rpe,
        replacement=args.replacement,
        trackback=args.trackback,
        num_walk=args.num_walk,
        num_step=args.num_step,
        preprocess_fn=preprocess_fn,
    )

    dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALIterableDataset'#default false
    test_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=args.test_percent, 
        split='test', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=directed, 
        sample_type=args.sample_type,
        slice_type=args.slice_type,
        use_rpe=args.use_rpe,
        replacement=args.replacement,
        trackback=args.trackback,
        num_walk=args.num_walk,
        num_step=args.num_step,
        preprocess_fn=preprocess_fn,
    )

    dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALIterableDataset'#default false
    final_val_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=args.final_val_percent, 
        split='valid', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=directed, 
        sample_type=args.sample_type,
        slice_type=args.slice_type,
        use_rpe=args.use_rpe,
        replacement=args.replacement,
        trackback=args.trackback,
        num_walk=args.num_walk,
        num_step=args.num_step,
        preprocess_fn=preprocess_fn,
    )

    dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALIterableDataset'#default false
    final_test_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=args.final_test_percent, 
        split='test', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=directed, 
        sample_type=args.sample_type,
        slice_type=args.slice_type,
        use_rpe=args.use_rpe,
        replacement=args.replacement,
        trackback=args.trackback,
        num_walk=args.num_walk,
        num_step=args.num_step,
        preprocess_fn=preprocess_fn,
    )

    if args.use_full_graphormer:#default false
        collate_fn=partial(collator)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True if args.dynamic_train else False,
                                num_workers=args.num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_workers, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_workers, collate_fn=collate_fn)
        final_val_loader = DataLoader(final_val_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_workers, collate_fn=collate_fn)
        final_test_loader = DataLoader(final_test_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_workers, collate_fn=collate_fn)
    else:
        train_loader = PygDataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True if args.dynamic_train else False,
                                num_workers=args.num_workers)
        val_loader = PygDataLoader(val_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_workers)
        test_loader = PygDataLoader(test_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_workers)
        final_val_loader = PygDataLoader(final_val_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_workers)
        final_test_loader = PygDataLoader(final_test_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_workers)

if args.train_node_embedding:
    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
elif args.pretrained_node_embedding:
    weight = torch.load(args.pretrained_node_embedding)
    emb = torch.nn.Embedding.from_pretrained(weight)
    emb.weight.requires_grad=False
else:
    emb = None

if 'DGCNN' in args.model:
    if args.ngnn_code:
        if 0 < args.sortpool_k <= 1:  # Transform percentile to number.
            if args.dataset.startswith("ogbl-citation"):
                # For this dataset, subgraphs extracted around positive edges are
                # rather larger than negative edges. Thus we sample from 1000
                # positive and 1000 negative edges to estimate the k (number of 
                # nodes to hold for each graph) used in SortPooling.
                # You can certainly set k manually, instead of estimating from
                # a percentage of sampled subgraphs.
                _sampled_indices = list(range(1000)) + list(
                    range(len(train_dataset) - 1000, len(train_dataset))
                )
            else:
                _sampled_indices = list(range(1000))
            _num_nodes = sorted(
                [train_dataset[i][0].num_nodes() for i in _sampled_indices]
            )
            _k = _num_nodes[int(math.ceil(args.sortpool_k * len(_num_nodes))) - 1]
            model_k = max(10, _k)
        else:
            model_k = int(args.sortpool_k)
    else:
        model_k = args.sortpool_k

print(f'args: {args}')
results_list = []
if args.dot_enh == 1:
    args.use_degree = False
for run in range(args.runs):
    if args.model == 'DGCNN':
        if not args.ngnn_code:  # sieg_code
            model = DGCNN(args, args.hidden_channels, args.num_layers, args.max_z, model_k, 
                          train_dataset, use_feature=args.use_feature, 
                          node_embedding=emb).to(device)
        else:  # ngnn_code
            model = ngnn_models.DGCNN(
                args.hidden_channels,
                args.num_layers,
                args.max_z,
                model_k,
                feature_dim=graph.ndata["feat"].size(1)
                if (args.use_feature and "feat" in graph.ndata)
                else 0,
                dropout=args.dropout,
                ngnn_type=args.ngnn_type,
                num_ngnn_layers=args.num_ngnn_layers,
            ).to(device)
    elif args.model == 'SAGE':
        model = SAGE(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                     args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'GCN':
        model = GCN(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                    args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'GIN':
        model = GIN(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                    args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'GCNGraphormer':
        model = GCNGraphormer(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                    use_feature=args.use_feature, use_feature_GT=args.use_feature_GT, node_embedding=emb).to(device)
    elif args.model == 'GCNFFNGraphormer':
        model = GCNFFNGraphormer(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                    use_feature=args.use_feature, use_feature_GT=args.use_feature_GT, node_embedding=emb).to(device)
    elif args.model == 'GCNGraphormer_noNeigFeat':
        z_emb_agg = False if args.dont_z_emb_agg else True
        model = GCNGraphormer_noNeigFeat(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                    use_feature=args.use_feature, use_feature_GT=args.use_feature_GT, node_embedding=emb, z_emb_agg=z_emb_agg).to(device)
    elif args.model == 'SingleFFN':
        model = SingleFFN(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                    use_feature=args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'FFNGraphormer':
        model = FFNGraphormer(args, args.hidden_channels, args.num_layers, args.max_z, train_dataset, 
                    use_feature=args.use_feature, use_feature_GT=args.use_feature_GT, node_embedding=emb).to(device)
    elif args.model == 'DGCNNGraphormer':
        model = DGCNNGraphormer(args, args.hidden_channels, args.num_layers, args.max_z,
                    k=model_k, train_dataset=train_dataset,
                    use_feature=args.use_feature, use_feature_GT=args.use_feature_GT,
                    node_embedding=emb, readout_type=args.readout_type).to(device)
    elif args.model == 'DGCNNGraphormer_noNeigFeat':
        model = DGCNNGraphormer_noNeigFeat(args, args.hidden_channels, args.num_layers, args.max_z,
                    k=model_k, train_dataset=train_dataset,
                    use_feature=args.use_feature, use_feature_GT=args.use_feature_GT,
                    node_embedding=emb, readout_type=args.readout_type).to(device)
    elif args.model == 'NGNNDGCNNGraphormer':
        model = NGNNDGCNNGraphormer(args, args.hidden_channels, args.num_layers, args.max_z,
                k=model_k, feature_dim=graph.ndata["feat"].size(1),
                use_feature=args.use_feature, use_feature_GT=args.use_feature_GT,
                node_embedding=emb, readout_type=args.readout_type).to(device)
    elif args.model == 'DGCNN_noNeigFeat':
        model = DGCNN_noNeigFeat(args, args.hidden_channels, args.num_layers, args.max_z, model_k, 
                        train_dataset, use_feature=args.use_feature, 
                        node_embedding=emb).to(device)
    elif args.model == 'NGNNDGCNN_noNeigFeat':
        model = ngnn_models.DGCNN_noNeigFeat(
            args.hidden_channels,
            args.num_layers,
            args.max_z,
            model_k,
            feature_dim=graph.ndata["feat"].size(1)
            if (args.use_feature and "feat" in graph.ndata)
            else 0,
            dropout=args.dropout,
            ngnn_type=args.ngnn_type,
            num_ngnn_layers=args.num_ngnn_layers,
        ).to(device)
    elif args.model == 'NGNNDGCNNGraphormer_noNeigFeat':
        model = NGNNDGCNNGraphormer_noNeigFeat(args, args.hidden_channels, args.num_layers, args.max_z,
                k=model_k, feature_dim=graph.ndata["feat"].size(1),
                use_feature=args.use_feature, use_feature_GT=args.use_feature_GT,
                node_embedding=emb, readout_type=args.readout_type).to(device)
    elif args.model == 'SingleGraphormer':
        model = SingleGraphormer(args, args.hidden_channels, args.num_layers, args.max_z, 
                      train_dataset=train_dataset, use_feature=args.use_feature, use_feature_GT=args.use_feature_GT, 
                      node_embedding=emb).to(device)
    print(model)
    parameters = list(model.parameters())
    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    lr = 2 * args.lr if args.scheduler else args.lr
    optimizer = torch.optim.Adam(params=parameters, lr=lr)  # , weight_decay=0.002
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2)
    total_params = sum(p.numel() for param in parameters for p in param)
    localtime = time.asctime(time.localtime(time.time()))
    print(f'{localtime} Total number of parameters is {total_params}')
    if args.model.find('DGCNN') != -1:
        print(f'SortPooling k is set to {model.k}')
    start_epoch = 1
    val_auc = 0
    # Training starts
    for epoch in range(start_epoch, start_epoch + args.epochs):
        loss, train_result = train(len(train_dataset))  # {'AUC': 0.9661961285501943}

        if epoch % args.eval_steps == 0:
            results = test(args.eval_metric)  # {'MRR': (0.7427010536193848, 0.7336037158966064), 'AUC': (0.9981022174148187, 0.9458885884261763)}
            # for key in loggers.keys():  # MRR
            #     result = results[key]
            #     loggers[key].add_result(run, result)
        tmp_val_auc, tmp_test_auc = results['AUC']
        tmp_val_ap, tmp_test_ap = results['AP']
        if tmp_val_auc > val_auc:
            val_auc = tmp_val_auc
            test_auc = tmp_test_auc
            val_ap = tmp_val_ap
            test_ap = tmp_test_ap
            best_epoch = epoch
        to_print = f'Epoch: {epoch:02d}, Best epoch: {best_epoch}, Loss: {loss:.4f}, Valid AUC: {100 * val_auc:.2f}%, Test AUC: ' \
                   f' {100 * test_auc:.2f}%, Valid AP: {100 * val_ap:.2f}%, Test AP: {100 * test_ap:.2f}%'
        print(to_print)
    if args.runs > 1:
        results_list.append([test_auc, val_auc, test_ap, val_ap])
        for idx, res in enumerate(results_list):
            print(f'repetition {idx}: test auc {res[0]:.4f}, val auc {res[1]:.4f}, test ap {res[2]:.4f}, val ap {res[3]:.4f}')


if args.runs > 1:
        test_auc_mean, val_auc_mean, test_ap_mean, val_ap_mean = np.mean(results_list, axis=0) * 100
        test_auc_std = np.sqrt(np.var(results_list, axis=0)[0]) * 100
        val_auc_std = np.sqrt(np.var(results_list, axis=0)[1]) * 100
        print('test_auc_mean: ', test_auc_mean, 'val_auc_mean: ', val_auc_mean)
        print('test_auc_std: ', test_auc_std, 'val_auc_std: ', val_auc_std)
        test_ap_std = np.sqrt(np.var(results_list, axis=0)[2]) * 100
        val_ap_std = np.sqrt(np.var(results_list, axis=0)[3]) * 100
        print('test_ap_mean: ', test_ap_mean, 'val_ap_mean: ', val_ap_mean)
        print('test_ap_std: ', test_ap_std, 'val_ap_std: ', val_ap_std)