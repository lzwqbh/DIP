"Implementation based on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/argva_node_clustering.py"
from torch_geometric.nn import GCNConv, ARGVA, ARGA
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from torch_geometric.nn import InnerProductDecoder

EPS = 1e-15

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1,hidden_channels2, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels1)
        self.lin2 = torch.nn.Linear(hidden_channels1, hidden_channels2)
        self.lin3 = torch.nn.Linear(hidden_channels2, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x

def recon_mul_loss(z, pos_edge_index, deg_wei, device, neg_edge_index = None):
    decoder = InnerProductDecoder()
    pos_loss = -torch.log(decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

    if neg_edge_index is None:
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0)).to(device)
    row, col = neg_edge_index
    neg_loss = -(deg_wei[row]*deg_wei[col]* torch.log(1 - decoder(z, neg_edge_index, sigmoid=True) + EPS)).mean()

    return pos_loss + neg_loss

def compute_scores(z,test_pos,test_neg):
    test = torch.cat((test_pos, test_neg),dim=1)
    labels = torch.zeros(test.size(1),1)
    labels[0:test_pos.size(1)] = 1
    row, col = test
    src = z[row]
    tgt = z[col]
    scores = torch.sigmoid(torch.sum(src * tgt,dim=1))
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc,ap


def CalARGVA(edge_index, x, test_and_val, args, node_deg, neg_row, neg_col):
    device = torch.device('cuda:' + args.gpu_num if torch.cuda.is_available() else 'cpu')
    print('___Calculating ARGVA embbeding___')
    test_pos,test_neg,val_pos,val_neg=test_and_val
    out_channels=128
    encoder = Encoder(x.size(1), hidden_channels=64, out_channels=args.embedding_dim)
    discriminator = Discriminator(in_channels=args.embedding_dim, hidden_channels1=16,hidden_channels2=64,
                                out_channels=out_channels)
    model = ARGVA(encoder, discriminator).to(device)

    edge_index = edge_index.to(device)
    x = x.to(device)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                           lr=0.001)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    num_nodes = torch.max(edge_index)+1
    best_val_auc=0
    epoch_num=2000
    if args.data_name=='pubmed':
        epoch_num=3000

    n_train = int(edge_index.size(1))
    deg_wei = torch.zeros(num_nodes, dtype=torch.float).to(device)
    for i in range(num_nodes):
        deg_wei[i] = 2 / (n_train / node_deg[i] + num_nodes) * ((num_nodes * (num_nodes - 1)) ** 0.5)
        # if node_deg[i] < 0.5:
        #     deg_wei[i] = 0
        # else:
        #     deg_wei[i] = 1

    for epoch in range(1, epoch_num+1):
        model.train()
        encoder_optimizer.zero_grad()
        z = model.encode(x, edge_index)

        for i in range(5):
            discriminator_optimizer.zero_grad()
            discriminator_loss = model.discriminator_loss(z)
            discriminator_loss.backward()
            discriminator_optimizer.step()

        perm = torch.randperm(neg_row.size(0))[:n_train]
        row, col = neg_row[perm], neg_col[perm]
        neg_edge_index = torch.stack([row, col], dim=0).to(device)

        if args.loss_emb < 0.5:
            loss = model.recon_loss(z, edge_index, neg_edge_index=neg_edge_index)
        else:
            loss = recon_mul_loss(z, edge_index, deg_wei, device, neg_edge_index=neg_edge_index)
        loss = loss + model.reg_loss(z)
        loss = loss + (1 / num_nodes) * model.kl_loss()
        loss.backward()
        encoder_optimizer.step()
        if epoch%1==0:
            model.eval()
            z = model.encode(x, edge_index)
            z = z.cpu().clone().detach()
            auc,_=compute_scores(z,val_pos,val_neg)
            if auc>best_val_auc:
                best_val_auc=auc
                record_z=z.clone().detach()
            print(f'Setp: {epoch:03d} /2000, Loss : {loss.item():.4f}, Val_auc:{best_val_auc:.4f}')


    auc,ap=compute_scores(record_z,test_pos,test_neg)
    print(f'argva prediction accuracy, AUC: {auc:.4f}, AP: {ap:.4f}')
    del model
    del discriminator
    del encoder
    return record_z, auc, ap