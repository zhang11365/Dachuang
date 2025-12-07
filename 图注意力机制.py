import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data, HeteroData
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GATLayer(nn.Module):
    """
    基础图注意力层
    实现自注意力机制，学习节点之间的重要性权重
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # 特征变换矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 注意力机制参数
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h, adj):
        """
        前向传播
        Args:
            h: 节点特征矩阵 [N, in_features]
            adj: 邻接矩阵 [N, N]
        Returns:
            更新后的节点特征 [N, out_features]
        """
        # 特征线性变换
        Wh = torch.mm(h, self.W)  # [N, out_features]
        
        # 计算注意力系数
        N = Wh.size()[0]
        
        # 扩展特征矩阵用于注意力计算
        Wh_repeated = Wh.repeat(N, 1)  # [N*N, out_features]
        Wh_repeated_interleave = Wh.repeat_interleave(N, dim=0)  # [N*N, out_features]
        
        # 拼接特征并计算注意力
        a_input = torch.cat([Wh_repeated, Wh_repeated_interleave], dim=1)  # [N*N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a)).view(N, N)  # [N, N]
        
        # 掩码处理：只考虑有连接的节点对
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # 注意力归一化
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 注意力加权聚合
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class MultiHeadGATLayer(nn.Module):
    """
    多头图注意力层
    并行多个注意力机制，捕获不同类型的依赖关系
    """
    def __init__(self, in_features, out_features, n_heads=8, dropout=0.6, alpha=0.2, concat=True):
        super(MultiHeadGATLayer, self).__init__()
        self.n_heads = n_heads
        self.concat = concat
        
        # 创建多个注意力头
        self.attentions = nn.ModuleList([
            GATLayer(in_features, out_features, dropout=dropout, 
                    alpha=alpha, concat=True) for _ in range(n_heads)
        ])
        
    def forward(self, h, adj):
        """
        前向传播
        """
        # 并行计算所有注意力头的输出
        head_outputs = [att(h, adj) for att in self.attentions]
        
        if self.concat:
            # 拼接所有头的输出
            return torch.cat(head_outputs, dim=1)
        else:
            # 平均所有头的输出
            return torch.mean(torch.stack(head_outputs), dim=0)