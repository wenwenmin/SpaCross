import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



class Readout(nn.Module):
    def __init__(self, K):
        super(Readout, self).__init__()
        self.K = K

    def forward(self, Z):
        Z_tilde = []
        n_node = Z.shape[0]
        step = n_node // self.K
        for i in range(0, n_node, step):
            if n_node - i < 2 * step:
                Z_tilde.append(torch.mean(Z[i:n_node], dim=0))
                break
            else:
                Z_tilde.append(torch.mean(Z[i:i + step], dim=0))

        Z_tilde = torch.cat(Z_tilde, dim=0)

        return Z_tilde.view(1, -1)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t())) 
        return adj
    
class Encoder(Module):
    def __init__(self, in_features, out_features, n_clusters, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, self.out_features))
        self.reset_parameters()

        self.sigm = nn.Sigmoid()
        self.read = Readout(K=n_clusters)
        self.dc = InnerProductDecoder(0.1, act=nn.Sigmoid())

        
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)

        emb = self.act(z)
        emb_a = self.act(z_a)

        g = self.read(emb)
        ret = self.sigm(g)  
        g_a = self.read(emb_a)
        ret_a = self.sigm(g_a) 

        A_rec = self.dc(emb)

        return hiden_emb, z_a, h, ret, ret_a, A_rec
    