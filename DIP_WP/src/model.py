import torch
from torch.nn import Linear, Parameter,Embedding
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter, scatter_add, scatter_max
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_geometric.nn import GCNConv, GINConv
import numpy as np
from torch_geometric.nn.inits import reset
import pdb

class LinkPred(MessagePassing):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 1,\
                 walk_len: int = 6, drnl: bool = False, z_max: int =100, MSE: bool=True):
        super(LinkPred, self).__init__()

        self.drnl = drnl
        if drnl == True:
            self.z_embedding = Embedding(z_max, hidden_channels)

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.wp = WalkPooling(hidden_channels * 2, hidden_channels, heads, walk_len)

        L = walk_len * 5 + 1

        self.classifier = MLP(L*heads,MSE=MSE)


    def forward(self, x, edge_index, edge_mask, batch, emb_pro, less_para, z = None, gpu_num = '0'):
        
        #using drnl
        if self.drnl == True:
            z_emb = self.z_embedding(z)
            if z_emb.ndim == 3:  # in case z has multiple integer labels
                z_emb = z_emb.sum(dim=1)
            z_emb = z_emb.view(x.size(0),-1)
            x = torch.cat((x,z_emb.view(x.size(0),-1)),dim=1)
        
        #GCN layers
        x = self.conv1(x, edge_index)
        x_out = x
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x_out = torch.cat((x_out,x),dim=1)
        #Walk Pooling
        feature_list = self.wp(x_out, edge_index, edge_mask, batch, gpu_num)

        #Classifier
        out = self.classifier(feature_list, emb_pro, less_para)

        return out


class WalkPooling(MessagePassing):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 1,\
                 walk_len: int = 6):
        super(WalkPooling, self).__init__()

        self.hidden_channels = hidden_channels
        self.heads = heads
        self.walk_len = walk_len

        # the linear layers in the attention encoder
        self.lin_key1 = Linear(in_channels, hidden_channels)
        self.lin_query1 = Linear(in_channels, hidden_channels)
        self.lin_key2 = Linear(hidden_channels, heads * hidden_channels)
        self.lin_query2 = Linear(hidden_channels, heads * hidden_channels)
    def attention_mlp(self, x, edge_index):
    
        query = self.lin_key1(x).reshape(-1,self.hidden_channels)
        key = self.lin_query1(x).reshape(-1,self.hidden_channels)

        query = F.leaky_relu(query,0.2)
        key = F.leaky_relu(key,0.2)

        query = F.dropout(query, p=0.5, training=self.training)
        key = F.dropout(key, p=0.5, training=self.training)

        query = self.lin_key2(query).view(-1, self.heads, self.hidden_channels)
        key = self.lin_query2(key).view(-1, self.heads, self.hidden_channels)

        row, col = edge_index
        weights = (query[row] * key[col]).sum(dim=-1) / np.sqrt(self.hidden_channels)
        
        return weights

    def weight_encoder(self, x, edge_index, edge_mask):        
     
        weights = self.attention_mlp(x, edge_index)
    
        omega = torch.sigmoid(weights[torch.logical_not(edge_mask)])
        
        row, col = edge_index
        num_nodes = torch.max(edge_index)+1

        # edge weights of the plus graph
        weights_p = softmax(weights,edge_index[1])

        # edge weights of the minus graph
        weights_m = weights - scatter_max(weights, col, dim=0, dim_size=num_nodes)[0][col]
        weights_m = torch.exp(weights_m)
        weights_m = weights_m * edge_mask.view(-1,1)
        norm = scatter_add(weights_m, col, dim=0, dim_size=num_nodes)[col] + 1e-16
        weights_m = weights_m / norm

        return weights_p, weights_m, omega

    def forward(self, x, edge_index, edge_mask, batch, gpu_num):
        device = torch.device('cuda:' + gpu_num if torch.cuda.is_available() else 'cpu')
        
        #encode the node representation into edge weights via attention mechanism
        weights_p, weights_m, omega = self.weight_encoder(x, edge_index, edge_mask)

        # pytorch geometric set the batch adjacency matrix to
        # be the diagonal matrix with each graph's adjacency matrix
        # stacked in the diagonal. Therefore, calculating the powers
        # of the stochastic matrix directly will cost lots of memory.
        # We compute the powers of stochastic matrix as follows
        # Let A = diag ([A_1,\cdots,A_n]) be the batch adjacency matrix,
        # we set x = [x_1,\cdots,x_n]^T be the batch feature matrix
        # for the i-th graph in the batch with n_i nodes, its feature 
        # is a n_i\times n_max matrix, where n_max is the largest number
        # of nodes for all graphs in the batch. The elements of x_i are
        # (x_i)_{x,y} = \delta_{x,y}. 

        # number of graphs in the batch
        batch_size = torch.max(batch)+1

        # for node i in the batched graph, index[i] is i's id in the graph before batch 
        index = torch.zeros(batch.size(0),1,dtype=torch.long)
        
        # numer of nodes in each graph
        _, counts = torch.unique(batch, sorted=True, return_counts=True)
        
        # maximum number of nodes for all graphs in the batch
        max_nodes = torch.max(counts)

        # set the values in index
        id_start = 0
        for i in range(batch_size):
            index[id_start:id_start+counts[i]] = torch.arange(0,counts[i],dtype=torch.long).view(-1,1)
            id_start = id_start+counts[i]

        index = index.to(device)
        
        #the output graph features of walk pooling
        nodelevel_p = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        nodelevel_m = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        linklevel_p = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        linklevel_m = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        graphlevel = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        # a link (i,j) has two directions i->j and j->i, and
        # when extract the features of the link, we usually average over
        # the two directions. indices_odd and indices_even records the
        # indices for a link in two directions
        indices_odd = torch.arange(0,omega.size(0),2).to(device)
        indices_even = torch.arange(1,omega.size(0),2).to(device)

        #x,dimension,indices
        omega = torch.index_select(omega, 0 ,indices_even)\
        + torch.index_select(omega,0,indices_odd)
        
        #node id of the candidate (or perturbation) link
        link_ij, link_ji = edge_index[:,torch.logical_not(edge_mask)]
        node_i = link_ij[indices_odd]
        node_j = link_ij[indices_even]

        # compute the powers of stochastic matrix
        for head in range(self.heads):

            # x on the plus graph and minus graph
            x_p = torch.zeros(batch.size(0),max_nodes,dtype=x.dtype).to(device)
            x_p = x_p.scatter_(1,index,1)
            x_m = torch.zeros(batch.size(0),max_nodes,dtype=x.dtype).to(device)
            x_m = x_m.scatter_(1,index,1)

            # propagage once
            x_p = self.propagate(edge_index, x= x_p, norm = weights_p[:,head])
            x_m = self.propagate(edge_index, x= x_m, norm = weights_m[:,head])
        
            # start from tau = 2
            for i in range(self.walk_len):
                x_p = self.propagate(edge_index, x= x_p, norm = weights_p[:,head])
                x_m = self.propagate(edge_index, x= x_m, norm = weights_m[:,head])
                
                # returning probabilities around i + j
                nodelevel_p_w = x_p[node_i,index[node_i].view(-1)] + x_p[node_j,index[node_j].view(-1)]
                nodelevel_m_w = x_m[node_i,index[node_i].view(-1)] + x_m[node_j,index[node_j].view(-1)]
                nodelevel_p[:,head*self.walk_len+i] = nodelevel_p_w.view(-1)
                nodelevel_m[:,head*self.walk_len+i] = nodelevel_m_w.view(-1)
  
                # transition probabilities between i and j
                linklevel_p_w = x_p[node_i,index[node_j].view(-1)] + x_p[node_j,index[node_i].view(-1)]
                linklevel_m_w = x_m[node_i,index[node_j].view(-1)] + x_m[node_j,index[node_i].view(-1)]
                linklevel_p[:,head*self.walk_len+i] = linklevel_p_w.view(-1)
                linklevel_m[:,head*self.walk_len+i] = linklevel_m_w.view(-1)

                # graph average of returning probabilities
                diag_ele_p = torch.gather(x_p,1,index)
                diag_ele_m = torch.gather(x_m,1,index)

                graphlevel_p = scatter_add(diag_ele_p, batch, dim = 0)
                graphlevel_m = scatter_add(diag_ele_m, batch, dim = 0)

                #graphlevel[:, head * self.walk_len + i] = graphlevel_m.view(-1)
                graphlevel[:,head*self.walk_len+i] = (graphlevel_p-graphlevel_m).view(-1)
         
        feature_list = graphlevel
        feature_list = torch.cat((feature_list,omega),dim=1)

        # feature_list = omega
        feature_list = torch.cat((feature_list,nodelevel_p),dim=1)
        feature_list = torch.cat((feature_list,nodelevel_m),dim=1)
        feature_list = torch.cat((feature_list,linklevel_p),dim=1)
        feature_list = torch.cat((feature_list,linklevel_m),dim=1)


        return feature_list

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j  

class MLP(torch.nn.Module):
    # adopt a MLP as classifier for graphs
    def __init__(self,input_size,MSE=True):
        super(MLP, self).__init__()
        self.nn = nn.BatchNorm1d(input_size)

        self.linear1 = torch.nn.Linear(input_size,input_size*20)
        self.linear2 = torch.nn.Linear(input_size*20,input_size*20)
        self.linear3 = torch.nn.Linear(input_size*20,input_size*10)
        self.linear4 = torch.nn.Linear(input_size*10,input_size)
        self.linear5 = torch.nn.Linear(input_size,1)

        self.act= nn.ReLU()
        self.MSE=MSE
    def forward(self, x, emb_pro, less_para):
        out= self.nn(x)

        out= self.linear1(out)
        out = self.act(out)
        if less_para == 0:
            out= self.linear2(out)
            out = self.act(out)
        out = self.linear3(out)
        out = self.act(out)
        out = self.linear4(out)
        out = self.act(out)

        out = F.dropout(out, p=0.5, training=self.training)
        out = self.linear5(out)
        if self.MSE:
            out = torch.sigmoid(out)
        return out

#use when ratio of training set is low
class LinkPred_Low(MessagePassing):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 1, \
                 walk_len: int = 6, drnl: bool = False, z_max: int = 100, MSE: bool = True):
        super(LinkPred_Low, self).__init__()

        self.drnl = drnl
        if drnl == True:
            self.z_embedding = Embedding(z_max, hidden_channels)

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.wp = WalkPooling_Low(hidden_channels * 2, hidden_channels, heads, walk_len)

        L = walk_len * 5 + 1

        self.classifier = MLP_Low(L * heads, walk_len, heads, MSE=MSE)

    def forward(self, x, edge_index, edge_mask, batch, emb_pro, less_para, z=None, gpu_num='0'):

        # using drnl
        if self.drnl == True:
            z_emb = self.z_embedding(z)
            if z_emb.ndim == 3:  # in case z has multiple integer labels
                z_emb = z_emb.sum(dim=1)
            z_emb = z_emb.view(x.size(0), -1)
            x = torch.cat((x, z_emb.view(x.size(0), -1)), dim=1)

        # get original dot product
        device = torch.device('cuda:' + gpu_num if torch.cuda.is_available() else 'cpu')
        row, col = edge_index
        weights = (x[row] * x[col]).sum(dim=-1)
        d_p = torch.sigmoid(weights[torch.logical_not(edge_mask)])
        indices_odd = torch.arange(0, d_p.size(0), 2).to(device)
        d_p = torch.index_select(d_p, 0, indices_odd)

        # GCN layers
        x = self.conv1(x, edge_index)
        x_out = x
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x_out = torch.cat((x_out, x), dim=1)
        # 这里说明了WP除了在pooling阶段外并没有利用有差异的结构去获得表征
        # Walk Pooling
        feature_list = self.wp(x_out, edge_index, edge_mask, batch, gpu_num, d_p)

        # Classifier
        out = self.classifier(feature_list, d_p, emb_pro, less_para)

        return out


class WalkPooling_Low(MessagePassing):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 1, \
                 walk_len: int = 6):
        super(WalkPooling_Low, self).__init__()

        self.hidden_channels = hidden_channels
        self.heads = heads
        self.walk_len = walk_len

        # the linear layers in the attention encoder
        self.lin_key1 = Linear(in_channels, hidden_channels)
        self.lin_query1 = Linear(in_channels, hidden_channels)
        self.lin_key2 = Linear(hidden_channels, heads * hidden_channels)
        self.lin_query2 = Linear(hidden_channels, heads * hidden_channels)

    def attention_mlp(self, x, edge_index):

        query = self.lin_key1(x).reshape(-1, self.hidden_channels)
        key = self.lin_query1(x).reshape(-1, self.hidden_channels)

        query = F.leaky_relu(query, 0.2)
        key = F.leaky_relu(key, 0.2)

        query = F.dropout(query, p=0.5, training=self.training)
        key = F.dropout(key, p=0.5, training=self.training)

        query = self.lin_key2(query).view(-1, self.heads, self.hidden_channels)
        # key = self.lin_key2(key).view(-1, self.heads, self.hidden_channels)
        key = self.lin_query2(key).view(-1, self.heads, self.hidden_channels)

        row, col = edge_index
        weights = (query[row] * key[col]).sum(dim=-1) / np.sqrt(self.hidden_channels)

        return weights

    def weight_encoder(self, x, edge_index, edge_mask):

        weights = self.attention_mlp(x, edge_index)

        omega = torch.sigmoid(weights[torch.logical_not(edge_mask)])

        row, col = edge_index
        num_nodes = torch.max(edge_index) + 1

        # edge weights of the plus graph
        weights_p = softmax(weights, edge_index[1])

        # edge weights of the minus graph
        weights_m = weights - scatter_max(weights, col, dim=0, dim_size=num_nodes)[0][col]
        weights_m = torch.exp(weights_m)
        weights_m = weights_m * edge_mask.view(-1, 1)
        norm = scatter_add(weights_m, col, dim=0, dim_size=num_nodes)[col] + 1e-16
        weights_m = weights_m / norm

        return weights_p, weights_m, omega

    def forward(self, x, edge_index, edge_mask, batch, gpu_num, d_p):
        device = torch.device('cuda:' + gpu_num if torch.cuda.is_available() else 'cpu')

        # encode the node representation into edge weights via attention mechanism
        weights_p, weights_m, omega = self.weight_encoder(x, edge_index, edge_mask)

        # pytorch geometric set the batch adjacency matrix to
        # be the diagonal matrix with each graph's adjacency matrix
        # stacked in the diagonal. Therefore, calculating the powers
        # of the stochastic matrix directly will cost lots of memory.
        # We compute the powers of stochastic matrix as follows
        # Let A = diag ([A_1,\cdots,A_n]) be the batch adjacency matrix,
        # we set x = [x_1,\cdots,x_n]^T be the batch feature matrix
        # for the i-th graph in the batch with n_i nodes, its feature
        # is a n_i\times n_max matrix, where n_max is the largest number
        # of nodes for all graphs in the batch. The elements of x_i are
        # (x_i)_{x,y} = \delta_{x,y}.

        # number of graphs in the batch
        batch_size = torch.max(batch) + 1

        # for node i in the batched graph, index[i] is i's id in the graph before batch
        index = torch.zeros(batch.size(0), 1, dtype=torch.long)

        # numer of nodes in each graph
        _, counts = torch.unique(batch, sorted=True, return_counts=True)

        # maximum number of nodes for all graphs in the batch
        max_nodes = torch.max(counts)

        # set the values in index
        id_start = 0
        for i in range(batch_size):
            index[id_start:id_start + counts[i]] = torch.arange(0, counts[i], dtype=torch.long).view(-1, 1)
            id_start = id_start + counts[i]

        index = index.to(device)

        # the output graph features of walk pooling
        nodelevel_p = torch.zeros(batch_size, (self.walk_len * self.heads)).to(device)
        nodelevel_m = torch.zeros(batch_size, (self.walk_len * self.heads)).to(device)
        linklevel_p = torch.zeros(batch_size, (self.walk_len * self.heads)).to(device)
        linklevel_m = torch.zeros(batch_size, (self.walk_len * self.heads)).to(device)
        graphlevel = torch.zeros(batch_size, (self.walk_len * self.heads)).to(device)
        # a link (i,j) has two directions i->j and j->i, and
        # when extract the features of the link, we usually average over
        # the two directions. indices_odd and indices_even records the
        # indices for a link in two directions
        indices_odd = torch.arange(0, omega.size(0), 2).to(device)
        indices_even = torch.arange(1, omega.size(0), 2).to(device)
        # x,dimension,indices
        omega = torch.index_select(omega, 0, indices_even) \
                + torch.index_select(omega, 0, indices_odd)

        # node id of the candidate (or perturbation) link
        link_ij, link_ji = edge_index[:, torch.logical_not(edge_mask)]
        node_i = link_ij[indices_odd]
        node_j = link_ij[indices_even]

        # compute the powers of stochastic matrix
        for head in range(self.heads):

            # x on the plus graph and minus graph
            x_p = torch.zeros(batch.size(0), max_nodes, dtype=x.dtype).to(device)
            x_p = x_p.scatter_(1, index, 1)
            x_m = torch.zeros(batch.size(0), max_nodes, dtype=x.dtype).to(device)
            x_m = x_m.scatter_(1, index, 1)

            # propagage once
            x_p = self.propagate(edge_index, x=x_p, norm=weights_p[:, head])
            x_m = self.propagate(edge_index, x=x_m, norm=weights_m[:, head])

            # start from tau = 2
            for i in range(self.walk_len):
                x_p = self.propagate(edge_index, x=x_p, norm=weights_p[:, head])
                x_m = self.propagate(edge_index, x=x_m, norm=weights_m[:, head])

                # returning probabilities around i + j
                nodelevel_p_w = x_p[node_i, index[node_i].view(-1)] + x_p[node_j, index[node_j].view(-1)]
                nodelevel_m_w = x_m[node_i, index[node_i].view(-1)] + x_m[node_j, index[node_j].view(-1)]
                nodelevel_p[:, head * self.walk_len + i] = nodelevel_p_w.view(-1)
                nodelevel_m[:, head * self.walk_len + i] = nodelevel_m_w.view(-1)

                # transition probabilities between i and j
                linklevel_p_w = x_p[node_i, index[node_j].view(-1)] + x_p[node_j, index[node_i].view(-1)]
                linklevel_m_w = x_m[node_i, index[node_j].view(-1)] + x_m[node_j, index[node_i].view(-1)]
                linklevel_p[:, head * self.walk_len + i] = linklevel_p_w.view(-1)
                linklevel_m[:, head * self.walk_len + i] = linklevel_m_w.view(-1)

                # graph average of returning probabilities
                diag_ele_p = torch.gather(x_p, 1, index)
                diag_ele_m = torch.gather(x_m, 1, index)

                graphlevel_p = scatter_add(diag_ele_p, batch, dim=0)
                graphlevel_m = scatter_add(diag_ele_m, batch, dim=0)

                # graphlevel[:, head * self.walk_len + i] = graphlevel_m.view(-1)
                graphlevel[:, head * self.walk_len + i] = (graphlevel_p - graphlevel_m).view(-1)

        # pdb.set_trace()
        feature_list = graphlevel
        feature_list = torch.cat((feature_list, omega), dim=1)
        # feature_list = torch.cat((feature_list, d_p.view(-1, 1)), dim=1)
        feature_list = torch.cat((feature_list, nodelevel_p), dim=1)
        feature_list = torch.cat((feature_list, nodelevel_m), dim=1)
        feature_list = torch.cat((feature_list, linklevel_p), dim=1)
        feature_list = torch.cat((feature_list, linklevel_m), dim=1)

        return feature_list

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class MLP_Low(torch.nn.Module):
    # adopt a MLP as classifier for graphs
    def __init__(self, input_size, walk_len, heads, MSE=True):
        super(MLP_Low, self).__init__()
        self.nn = nn.BatchNorm1d(input_size)

        self.linear1 = torch.nn.Linear(input_size, input_size * 20)
        self.linear2 = torch.nn.Linear(input_size * 20, input_size * 20)
        self.linear3 = torch.nn.Linear(input_size * 20, input_size * 10)
        self.linear4 = torch.nn.Linear(input_size * 10, input_size)
        self.linear5 = torch.nn.Linear(input_size, 1)

        self.linear_map = torch.nn.Linear(heads, input_size)
        self.linear1_wl = torch.nn.Linear(heads, input_size)
        self.linear2_wl = torch.nn.Linear(input_size, 1)

        self.act = nn.ReLU()
        self.MSE = MSE

        self.walk_len = walk_len
        self.heads = heads

    def forward(self, x, d_p, emb_pro, less_para):
        out = self.nn(x)

        if emb_pro == 1:
            out = self.linear1(out + self.linear_map(x[:, [self.walk_len * 2 + 2, self.walk_len * 4 + 2]]+x[:, [self.walk_len * 3 + 2, self.walk_len * 5 + 2]]))

        else:
            out = self.linear1(out)
        out = self.act(out)
        if less_para == 0:
            out = self.linear2(out)
            out = self.act(out)
        out = self.linear3(out)
        out = self.act(out)
        out = self.linear4(out)
        out = self.act(out)


        out = F.dropout(out, p=0.5, training=self.training)
        out = self.linear5(out)
        alp = self.linear1_wl(x[:, [self.walk_len * 2 + 2, self.walk_len * 4 + 2]]+x[:, [self.walk_len * 3 + 2, self.walk_len * 5 + 2]])

        out = out + self.linear2_wl(self.act(alp)) * d_p.view(-1, 1)

        if self.MSE:
            out = torch.sigmoid(out)
        return out

class LinkPred_Deg(MessagePassing):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 1, \
                 walk_len: int = 6, drnl: bool = False, z_max: int = 100, MSE: bool = True):
        super(LinkPred_Deg, self).__init__()

        self.drnl = drnl
        if drnl == True:
            self.z_embedding = Embedding(z_max, hidden_channels)

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.wp = WalkPooling_Low(hidden_channels * 2, hidden_channels, heads, walk_len)

        L = walk_len * 5 + 1

        self.classifier = MLP_Deg(L * heads, walk_len, heads, MSE=MSE)

    def forward(self, x, edge_index, edge_mask, batch, label, node_id, node_deg, emb_pro, less_para, z=None, gpu_num='0', loss_link=0):

        # using drnl
        if self.drnl == True:
            z_emb = self.z_embedding(z)
            if z_emb.ndim == 3:  # in case z has multiple integer labels
                z_emb = z_emb.sum(dim=1)
            z_emb = z_emb.view(x.size(0), -1)
            x = torch.cat((x, z_emb.view(x.size(0), -1)), dim=1)

        # get original dot product
        device = torch.device('cuda:' + gpu_num if torch.cuda.is_available() else 'cpu')
        row, col = edge_index
        weights = (x[row] * x[col]).sum(dim=-1)
        d_p = torch.sigmoid(weights[torch.logical_not(edge_mask)])
        indices_odd = torch.arange(0, d_p.size(0), 2).to(device)
        d_p = torch.index_select(d_p, 0, indices_odd)

        # GCN layers
        x = self.conv1(x, edge_index)
        x_out = x
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x_out = torch.cat((x_out, x), dim=1)
        # Walk Pooling
        feature_list = self.wp(x_out, edge_index, edge_mask, batch, gpu_num, d_p)

        # Classifier
        out = self.classifier(feature_list, d_p, label, node_id, node_deg, loss_link, emb_pro, less_para)

        return out

class MLP_Deg(torch.nn.Module):
    # adopt a MLP as classifier for graphs
    def __init__(self, input_size, walk_len, heads, MSE=True):
        super(MLP_Deg, self).__init__()
        self.nn = nn.BatchNorm1d(input_size)

        self.linear1 = torch.nn.Linear(input_size, input_size * 20)
        self.linear2 = torch.nn.Linear(input_size * 20, input_size * 20)
        self.linear3 = torch.nn.Linear(input_size * 20, input_size * 10)
        self.linear4 = torch.nn.Linear(input_size * 10, input_size)
        self.linear5 = torch.nn.Linear(input_size, 1)

        self.deg_dim = 1
        self.linear_map = torch.nn.Linear(self.deg_dim, input_size)
        self.linear1_wl = torch.nn.Linear(self.deg_dim, input_size)
        self.linear2_wl = torch.nn.Linear(input_size, 1)

        self.act = nn.ReLU()
        self.MSE = MSE

        self.walk_len = walk_len
        self.heads = heads

    def forward(self, x, d_p, label, node_id, node_deg, loss_link, emb_pro, less_para):
        out = self.nn(x)

        x_list_0 = torch.arange(0, label.size()[0] * 2, 2).view(-1,1)
        x_list_1 = torch.arange(1, label.size()[0] * 2, 2).view(-1,1)

        if emb_pro == 1:
            out = self.linear1(out + self.linear_map(node_deg[node_id[x_list_0]].float() + node_deg[node_id[x_list_1]].float()))
        else:
            out = self.linear1(out)
        out = self.act(out)
        if less_para == 0:
            out = self.linear2(out)
            out = self.act(out)
        out = self.linear3(out)
        out = self.act(out)
        out = self.linear4(out)
        out = self.act(out)

        out = F.dropout(out, p=0.5, training=self.training)
        out = self.linear5(out)
        alp = self.linear1_wl(node_deg[node_id[x_list_0]].float() + node_deg[node_id[x_list_1]].float())

        out = out + self.linear2_wl(self.act(alp)) * d_p.view(-1, 1)

        if self.MSE:
            out = torch.sigmoid(out)
        return out