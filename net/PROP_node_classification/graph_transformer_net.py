import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer
    
"""
from layer.graph_transformer_layer import GraphTransformerLayer
from layer.mlp_readout_layer import MLPReadout

class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, node_dim: int):
        """
        :param max_in_degree: 最大的入度限制
        :param node_dim: 节点特征的隐藏维度
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.node_dim = node_dim
        # 中心性编码的参数矩阵 (max_in_degree, node_dim)
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))

    def forward(self, x: torch.Tensor, g: dgl.DGLGraph) -> torch.Tensor:
        """
        :param x: 节点特征矩阵，形状为 (num_nodes, node_dim)
        :param g: DGL 图
        :return: 加入中心性编码后的节点特征矩阵
        """
        num_nodes = x.shape[0]

        # 计算节点的入度
        in_degree = g.in_degrees().long()  # 获取每个节点的入度
        in_degree = self.decrease_to_max_value(in_degree, self.max_in_degree - 1)  # 限制入度到最大值

        # 添加中心性编码
        x = x + self.z_in[in_degree]  # 根据入度为每个节点添加中心性偏置

        return x

    @staticmethod
    def decrease_to_max_value(x: torch.Tensor, max_value: int) -> torch.Tensor:
        """
        限制输入张量的值不超过最大值
        :param x: 输入张量
        :param max_value: 最大值
        :return: 限制后的张量
        """
        x = torch.clamp(x, max=max_value)  # 限制 x 的值不超过 max_value
        return x

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, adj, h):
        # Add self-loops to adjacency matrix
        adj = adj + torch.eye(adj.size(0), device=adj.device)

        # Degree matrix and normalization
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.diag(degree.pow(-0.5))
        adj_normalized = degree_inv_sqrt @ adj @ degree_inv_sqrt

        # Graph convolution
        h = adj_normalized @ h @ self.weight
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)

        # 第一层：1/4 -> 40
        self.layers.append(GraphConvolution(in_feats, hidden_size // 2))

        # 第二层：40 -> 80
        self.layers.append(GraphConvolution(hidden_size // 2, hidden_size))

    def forward(self, adj, inputs):
        h = inputs

        # 第一层
        h1 = self.layers[0](adj, h)  # GCN 第一层输出

        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        # 第二层
        h2 = self.layers[1](adj, h1)  # GCN 第二层输出

        h2 = F.relu(h2)

        h2 = self.dropout(h2)

        return h2

class GraphTransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.batch_size = net_params['batch_size']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 100 
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim) # Eigvec Feature (10) to hidden_dim (80)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim) # node feat is an integer
        # self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.GCNmodel = GCN(in_feats=5, hidden_size=80, num_layers=2, dropout=dropout) #---todo---
        
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,
                                              dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        #self.GCN_out_layer = GraphConvolution(out_dim, out_dim)#---todo---
        self.GCN_out_layer = GCN(in_feats=80, hidden_size=80, num_layers=2, dropout=dropout)
        self.centralityencoding = CentralityEncoding(max_in_degree=5, node_dim=80)


    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        # input embedding
        h = self.embedding_h(h) # in_dim to hidden_dim ---todo---
        #分解批图
        # subgraphs = dgl.unbatch(g)
        # adj_sparse = subgraphs[0].adjacency_matrix(scipy_fmt="csr")
        # adj_dense = torch.tensor(adj_sparse.todense(), dtype=torch.float32)
        # adj = adj_dense.to('cuda')
        #
        # # 提取特征
        # features_list = []
        #
        # for sg in subgraphs:
        #     node_features = sg.ndata['snapshots']
        #     features_list.append(node_features)
        #
        # features_list = [features.to('cuda') for features in features_list]
        # #
        # # # 前向传播
        # outputs = []
        #
        # for features in features_list:
        #     output = self.GCNmodel(adj, features)
        #     outputs.append(output)
        #
        # # 合并所有子图的输出
        # final_output = torch.cat(outputs, dim=0)
        # # print("最终输出嵌入特征大小:", final_output.shape)
        # h = final_output

        # if self.lap_pos_enc:
        #     h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
        #     h = h + h_lap_pos_enc
        # if self.wl_pos_enc:
        #     h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
        #     h = h + h_wl_pos_enc

        # CentralityEncoding
        h = self.centralityencoding(h, g) # ---todo---


        h = self.in_feat_dropout(h)
        
        # GraphTransformer Layers
        for conv in self.layers:
            h_init = h
            # 提取特征
            # features_list = []
            #
            # for sg in subgraphs:
            #     node_features = sg.ndata['features']
            #     features_list.append(node_features)
            #
            # features_list = [features.to('cuda') for features in features_list]
            #
            # # 前向传播
            # outputs = []
            #
            # for features in features_list:
            #     output = self.GCNmodel(adj, features)
            #     outputs.append(output)
            #
            # # 合并所有子图的输出
            # final_output = torch.cat(outputs, dim=0)
            # # print("最终输出嵌入特征大小:", final_output.shape)
            # h = final_output
            #
            # if self.lap_pos_enc:
            #     h_lap_pos_enc_new = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            #     h = h + h_lap_pos_enc_new
            h = conv(g, h) + h_init
            
        # output
        h_out = self.MLP_layer(h) #---todo---

        #GCNoutput
        # h_split = h.view(int(h.shape[0] / adj.shape[0]), int(adj.shape[0]), int(h.shape[1]))
        #
        # outputs = []
        #
        # for i in range(int(h.shape[0] / adj.shape[0])):
        #     h_i = h_split[i]
        #     h_i_out = self.GCN_out_layer(adj, h_i)
        #     outputs.append(h_i_out)
        # h_out = torch.cat(outputs, dim=0)

        # h_out = self.MLP_layer(h_out)
        return h_out
    

    def loss(self, pred, label):
        # 计算类别权重
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (cluster_sizes / V + 1e-6).float() / V
        weight *= (cluster_sizes > 0).float()

        class_counts = torch.bincount(label)
        weight = (1.0 / (class_counts + 1e-6)) * V / len(class_counts)
        weight = weight / weight.sum()  # 归一化

        # 交叉熵损失（加权）
        criterion1 = nn.CrossEntropyLoss(weight=weight)
        criterion2 = nn.CrossEntropyLoss()  # 不带权重的交叉熵损失
        celoss = criterion1(pred, label)
        celoss2 = criterion2(pred, label)

        # 均方误差损失
        label_one_hot = torch.zeros_like(pred).scatter_(1, label.unsqueeze(1), 1)
        criterion2 = nn.MSELoss()
        mse_loss = criterion2(pred, label_one_hot.float())

        # 计算Jaccard损失
        # 将pred转为概率值，获取最大概率的类别作为预测结果
        pred_class = torch.argmax(pred, dim=1)

        # 计算预测和标签之间的 Jaccard系数
        intersection = torch.sum((pred_class == label) & (label != -1)).float()  # 交集部分
        union = torch.sum((pred_class != -1) | (label != -1)).float()  # 并集部分

        # Jaccard系数
        jaccard_score = intersection / (union + 1e-6)  # 防止除以零
        jaccard_loss = 1 - jaccard_score  # 计算Jaccard损失，越接近1损失越大

        # 组合损失
        total_loss = celoss+mse_loss+jaccard_loss
        return total_loss



        
