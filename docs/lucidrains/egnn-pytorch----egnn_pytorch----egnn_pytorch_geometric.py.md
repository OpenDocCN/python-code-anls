# `.\lucidrains\egnn-pytorch\egnn_pytorch\egnn_pytorch_geometric.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn, einsum, broadcast_tensors
from torch import nn, einsum, broadcast_tensors
# 从 torch 库中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F

# 从 einops 库中导入 rearrange, repeat
from einops import rearrange, repeat
# 从 einops.layers.torch 库中导入 Rearrange
from einops.layers.torch import Rearrange

# 导入类型相关的模块
from typing import Optional, List, Union

# 尝试导入 torch_geometric 库
try:
    import torch_geometric
    # 从 torch_geometric.nn 中导入 MessagePassing
    from torch_geometric.nn import MessagePassing
    # 从 torch_geometric.typing 中导入 Adj, Size, OptTensor, Tensor
    from torch_geometric.typing import Adj, Size, OptTensor, Tensor
except:
    # 如果导入失败，则将相关类型设为 object 类型
    Tensor = OptTensor = Adj = MessagePassing = Size = object
    # 设置 PYG_AVAILABLE 为 False
    PYG_AVAILABLE = False
    
    # 为了避免类型建议时出现错误，将相关类型设为 object 类型
    Adj = object
    Size = object
    OptTensor = object
    Tensor = object

# 从当前目录下的 egnn_pytorch 文件中导入所有内容
from .egnn_pytorch import *

# 定义全局线性注意力类 GlobalLinearAttention_Sparse
class GlobalLinearAttention_Sparse(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        # 初始化序列规范化层 norm_seq 和 queries 规范化层 norm_queries
        self.norm_seq = torch_geomtric.nn.norm.LayerNorm(dim)
        self.norm_queries = torch_geomtric.nn.norm.LayerNorm(dim)
        # 初始化两个稀疏注意力层 attn1 和 attn2
        self.attn1 = Attention_Sparse(dim, heads, dim_head)
        self.attn2 = Attention_Sparse(dim, heads, dim_head)

        # 无法将 pyg norms 与 torch sequentials 连接
        # 初始化前馈神经网络规范化层 ff_norm
        self.ff_norm = torch_geomtric.nn.norm.LayerNorm(dim)
        # 初始化前馈神经网络 ff
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    # 定义前向传播函数
    def forward(self, x, queries, batch=None, batch_uniques=None, mask = None):
        res_x, res_queries = x, queries
        # 对输入 x 和 queries 进行序列规范化
        x, queries = self.norm_seq(x, batch=batch), self.norm_queries(queries, batch=batch)

        # 计算引导向量
        induced = self.attn1.sparse_forward(queries, x, batch=batch, batch_uniques=batch_uniques, mask = mask)
        # 计算输出
        out = self.attn2.sparse_forward(x, induced, batch=batch, batch_uniques=batch_uniques)

        # 更新 x 和 queries
        x =  out + res_x
        queries = induced + res_queries

        # 对 x 进行前馈神经网络规范化
        x_norm = self.ff_norm(x, batch=batch)
        # 前馈神经网络处理 x
        x = self.ff(x_norm) + x_norm
        return x, queries

# 定义 EGNN_Sparse 类，继承自 MessagePassing
class EGNN_Sparse(MessagePassing):
    """ Different from the above since it separates the edge assignment
        from the computation (this allows for great reduction in time and 
        computations when the graph is locally or sparse connected).
        * aggr: one of ["add", "mean", "max"]
    """
    # 初始化函数，设置模型参数
    def __init__(
        self,
        feats_dim,
        pos_dim=3,
        edge_attr_dim = 0,
        m_dim = 16,
        fourier_features = 0,
        soft_edge = 0,
        norm_feats = False,
        norm_coors = False,
        norm_coors_scale_init = 1e-2,
        update_feats = True,
        update_coors = True, 
        dropout = 0.,
        coor_weights_clamp_value = None, 
        aggr = "add",
        **kwargs
    ):
        # 检查聚合方法是否有效
        assert aggr in {'add', 'sum', 'max', 'mean'}, 'pool method must be a valid option'
        # 检查是否需要更新特征或坐标
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'
        # 设置默认聚合方法
        kwargs.setdefault('aggr', aggr)
        # 调用父类的初始化函数
        super(EGNN_Sparse, self).__init__(**kwargs)
        # 设置模型参数
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.coor_weights_clamp_value = None

        # 计算边的输入维度
        self.edge_input_dim = (fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)
        # 根据 dropout 设置创建 Dropout 层或 Identity 层
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 边的 MLP 网络
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            SiLU()
        )

        # 如果 soft_edge 为真，则创建边权重网络
        self.edge_weight = nn.Sequential(nn.Linear(m_dim, 1), 
                                         nn.Sigmoid()
        ) if soft_edge else None

        # 节点的 LayerNorm 或 Identity 层
        self.node_norm = torch_geometric.nn.norm.LayerNorm(feats_dim) if norm_feats else None
        # 坐标的 CoorsNorm 或 Identity 层
        self.coors_norm = CoorsNorm(scale_init = norm_coors_scale_init) if norm_coors else nn.Identity()

        # 节点的 MLP 网络
        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        ) if update_feats else None

        # 坐标的 MLP 网络
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            self.dropout,
            SiLU(),
            nn.Linear(self.m_dim * 4, 1)
        ) if update_coors else None

        # 初始化模型参数
        self.apply(self.init_)

    # 初始化函数，设置模型参数的初始化方式
    def init_(self, module):
        # 如果模块类型为 nn.Linear
        if type(module) in {nn.Linear}:
            # 使用 xavier_normal_ 初始化权重，zeros_ 初始化偏置
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, batch: Adj = None, 
                angle_data: List = None,  size: Size = None) -> Tensor:
        """ Inputs: 
            * x: (n_points, d) where d is pos_dims + feat_dims
            * edge_index: (2, n_edges)
            * edge_attr: tensor (n_edges, n_feats) excluding basic distance feats.
            * batch: (n_points,) long tensor. specifies xloud belonging for each point
            * angle_data: list of tensors (levels, n_edges_i, n_length_path) long tensor.
            * size: None
        """
        # 将输入的 x 分为坐标和特征
        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        
        # 计算相对坐标和相对距离
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist  = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        # 如果使用傅立叶特征
        if self.fourier_features > 0:
            # 对相对距离进行傅立叶编码
            rel_dist = fourier_encode_dist(rel_dist, num_encodings = self.fourier_features)
            rel_dist = rearrange(rel_dist, 'n () d -> n d')

        # 如果存在边属性，则将边属性和相对距离拼接
        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        # 进行消息传递和更新节点信息
        hidden_out, coors_out = self.propagate(edge_index, x=feats, edge_attr=edge_attr_feats,
                                                           coors=coors, rel_coors=rel_coors, 
                                                           batch=batch)
        # 返回节点坐标和隐藏层输出的拼接
        return torch.cat([coors_out, hidden_out], dim=-1)


    def message(self, x_i, x_j, edge_attr) -> Tensor:
        # 通过边属性和节点特征计算消息
        m_ij = self.edge_mlp( torch.cat([x_i, x_j, edge_attr], dim=-1) )
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """The initial call to start propagating messages.
            Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        # 检查输入并收集数据
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)
        
        # 获取消息
        m_ij = self.message(**msg_kwargs)

        # 如果需要更新坐标
        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)
            # 如果设置了夹紧值，则夹紧权重
            if self.coor_weights_clamp_value:
                coor_weights_clamp_value = self.coor_weights_clamp_value
                coor_weights.clamp_(min = -clamp_value, max = clamp_value)

            # 如果需要归一化，则对相对坐标进行归一化
            kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])

            mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"], **aggr_kwargs)
            coors_out = kwargs["coors"] + mhat_i
        else:
            coors_out = kwargs["coors"]

        # 如果需要更新特征
        if self.update_feats:
            # 如果传递了软边参数，则加权边
            if self.soft_edge:
                m_ij = m_ij * self.edge_weight(m_ij)
            m_i = self.aggregate(m_ij, **aggr_kwargs)

            hidden_feats = self.node_norm(kwargs["x"], kwargs["batch"]) if self.node_norm else kwargs["x"]
            hidden_out = self.node_mlp( torch.cat([hidden_feats, m_i], dim = -1) )
            hidden_out = kwargs["x"] + hidden_out
        else: 
            hidden_out = kwargs["x"]

        # 返回更新后的节点信息
        return self.update((hidden_out, coors_out), **update_kwargs)
    # 定义对象的字符串表示形式
    def __repr__(self):
        # 创建一个空字典
        dict_print = {}
        # 返回对象的字符串表示形式，包含对象的属性字典
        return "E(n)-GNN Layer for Graphs " + str(self.__dict__) 
class EGNN_Sparse_Network(nn.Module):
    r"""Sample GNN model architecture that uses the EGNN-Sparse
        message passing layer to learn over point clouds. 
        Main MPNN layer introduced in https://arxiv.org/abs/2102.09844v1

        Inputs will be standard GNN: x, edge_index, edge_attr, batch, ...

        Args:
        * n_layers: int. number of MPNN layers
        * ... : same interpretation as the base layer.
        * embedding_nums: list. number of unique keys to embedd. for points
                          1 entry per embedding needed. 
        * embedding_dims: list. point - number of dimensions of
                          the resulting embedding. 1 entry per embedding needed. 
        * edge_embedding_nums: list. number of unique keys to embedd. for edges.
                               1 entry per embedding needed. 
        * edge_embedding_dims: list. point - number of dimensions of
                               the resulting embedding. 1 entry per embedding needed. 
        * recalc: int. Recalculate edge feats every `recalc` MPNN layers. 0 for no recalc
        * verbose: bool. verbosity level.
        -----
        Diff with normal layer: one has to do preprocessing before (radius, global token, ...)
    """
    def forward(self, x, edge_index, batch, edge_attr,
                bsize=None, recalc_edge=None, verbose=0):
        """ Recalculate edge features every `self.recalc_edge` with the
            `recalc_edge` function if self.recalc_edge is set.

            * x: (N, pos_dim+feats_dim) will be unpacked into coors, feats.
        """
        # NODES - Embedd each dim to its target dimensions:
        x = embedd_token(x, self.embedding_dims, self.emb_layers)

        # regulates whether to embed edges each layer
        edges_need_embedding = True  
        for i,layer in enumerate(self.mpnn_layers):
            
            # EDGES - Embedd each dim to its target dimensions:
            if edges_need_embedding:
                edge_attr = embedd_token(edge_attr, self.edge_embedding_dims, self.edge_emb_layers)
                edges_need_embedding = False

            # attn tokens
            global_tokens = None
            if exists(self.global_tokens):
                unique, amounts = torch.unique(batch, return_counts)
                num_idxs = torch.cat([torch.arange(num_idxs_i) for num_idxs_i in amounts], dim=-1)
                global_tokens = self.global_tokens[num_idxs]

            # pass layers
            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0
            if not is_global_layer:
                x = layer(x, edge_index, edge_attr, batch=batch, size=bsize)
            else: 
                # only pass feats to the attn layer
                x_attn = layer[0](x[:, self.pos_dim:], global_tokens)
                # merge attn-ed feats and coords
                x = torch.cat( (x[:, :self.pos_dim], x_attn), dim=-1)
                x = layer[-1](x, edge_index, edge_attr, batch=batch, size=bsize)

            # recalculate edge info - not needed if last layer
            if self.recalc and ((i%self.recalc == 0) and not (i == len(self.mpnn_layers)-1)) :
                edge_index, edge_attr, _ = recalc_edge(x) # returns attr, idx, any_other_info
                edges_need_embedding = True
            
        return x

    def __repr__(self):
        return 'EGNN_Sparse_Network of: {0} layers'.format(len(self.mpnn_layers))
```