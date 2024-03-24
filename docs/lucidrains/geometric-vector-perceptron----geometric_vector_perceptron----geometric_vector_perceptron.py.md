# `.\lucidrains\geometric-vector-perceptron\geometric_vector_perceptron\geometric_vector_perceptron.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch_geometric.nn 模块中导入 MessagePassing 类
from torch_geometric.nn import MessagePassing

# types

# 导入类型提示相关的模块和类型
from typing import Optional, List, Union
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor, Tensor

# helper functions

# 定义一个函数，判断输入值是否存在
def exists(val):
    return val is not None

# classes

# 定义 GVP 类，继承自 nn.Module 类
class GVP(nn.Module):
    def __init__(
        self,
        *,
        dim_vectors_in,
        dim_vectors_out,
        dim_feats_in,
        dim_feats_out,
        feats_activation = nn.Sigmoid(),
        vectors_activation = nn.Sigmoid(),
        vector_gating = False
    ):
        super().__init__()
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in

        self.dim_vectors_out = dim_vectors_out
        dim_h = max(dim_vectors_in, dim_vectors_out)

        # 初始化权重参数
        self.Wh = nn.Parameter(torch.randn(dim_vectors_in, dim_h))
        self.Wu = nn.Parameter(torch.randn(dim_h, dim_vectors_out))

        self.vectors_activation = vectors_activation

        # 定义输出特征的网络结构
        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_h + dim_feats_in, dim_feats_out),
            feats_activation
        )

        # 根据 vector_gating 参数选择是否使用向量门控
        self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out) if vector_gating else None

    # 前向传播函数
    def forward(self, data):
        feats, vectors = data
        b, n, _, v, c  = *feats.shape, *vectors.shape

        # 断言向量维度和特征维度是否匹配
        assert c == 3 and v == self.dim_vectors_in, 'vectors have wrong dimensions'
        assert n == self.dim_feats_in, 'scalar features have wrong dimensions'

        # 计算 Vh 和 Vu
        Vh = einsum('b v c, v h -> b h c', vectors, self.Wh)
        Vu = einsum('b h c, h u -> b u c', Vh, self.Wu)

        # 计算向量的模长
        sh = torch.norm(Vh, p = 2, dim = -1)

        # 拼接特征和模长
        s = torch.cat((feats, sh), dim = 1)

        # 计算特征输出
        feats_out = self.to_feats_out(s)

        # 如果存在 scalar_to_vector_gates，则计算门控
        if exists(self.scalar_to_vector_gates):
            gating = self.scalar_to_vector_gates(feats_out)
            gating = gating.unsqueeze(dim = -1)
        else:
            gating = torch.norm(Vu, p = 2, dim = -1, keepdim = True)

        # 计算向量输出
        vectors_out = self.vectors_activation(gating) * Vu
        return (feats_out, vectors_out)

# 定义 GVPDropout 类，继承自 nn.Module 类
class GVPDropout(nn.Module):
    """ Separate dropout for scalars and vectors. """
    def __init__(self, rate):
        super().__init__()
        self.vector_dropout = nn.Dropout2d(rate)
        self.feat_dropout = nn.Dropout(rate)

    # 前向传播函数
    def forward(self, feats, vectors):
        return self.feat_dropout(feats), self.vector_dropout(vectors)

# 定义 GVPLayerNorm 类，继承自 nn.Module 类
class GVPLayerNorm(nn.Module):
    """ Normal layer norm for scalars, nontrainable norm for vectors. """
    def __init__(self, feats_h_size, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.feat_norm = nn.LayerNorm(feats_h_size)

    # 前向传播函数
    def forward(self, feats, vectors):
        vector_norm = vectors.norm(dim=(-1,-2), keepdim=True)
        normed_feats = self.feat_norm(feats)
        normed_vectors = vectors / (vector_norm + self.eps)
        return normed_feats, normed_vectors

# 定义 GVP_MPNN 类，继承自 MessagePassing 类
class GVP_MPNN(MessagePassing):
    r"""The Geometric Vector Perceptron message passing layer
        introduced in https://openreview.net/forum?id=1YLJDvSx6J4.
        
        Uses a Geometric Vector Perceptron instead of the normal 
        MLP in aggregation phase.

        Inputs will be a concatenation of (vectors, features)

        Args:
        * feats_x_in: int. number of scalar dimensions in the x inputs.
        * vectors_x_in: int. number of vector dimensions in the x inputs.
        * feats_x_out: int. number of scalar dimensions in the x outputs.
        * vectors_x_out: int. number of vector dimensions in the x outputs.
        * feats_edge_in: int. number of scalar dimensions in the edge_attr inputs.
        * vectors_edge_in: int. number of vector dimensions in the edge_attr inputs.
        * feats_edge_out: int. number of scalar dimensions in the edge_attr outputs.
        * vectors_edge_out: int. number of vector dimensions in the edge_attr outputs.
        * dropout: float. dropout rate.
        * vector_dim: int. dimensions of the space containing the vectors.
        * verbose: bool. verbosity level.
    """
    # 初始化函数，接受多个参数
    def __init__(self, feats_x_in, vectors_x_in,
                       feats_x_out, vectors_x_out,
                       feats_edge_in, vectors_edge_in,
                       feats_edge_out, vectors_edge_out,
                       dropout, residual=False, vector_dim=3, 
                       verbose=False, **kwargs):
        # 调用父类的初始化函数，设置聚合方式为"mean"
        super(GVP_MPNN, self).__init__(aggr="mean",**kwargs)
        # 记录是否输出详细信息
        self.verbose = verbose
        # 记录输入特征和向量的维度
        self.feats_x_in    = feats_x_in 
        self.vectors_x_in  = vectors_x_in # 输入中的 N 个向量特征
        self.feats_x_out   = feats_x_out 
        self.vectors_x_out = vectors_x_out # 输出中的 N 个向量特征
        # 记录边属性的维度
        self.feats_edge_in    = feats_edge_in 
        self.vectors_edge_in  = vectors_edge_in # 输入中的 N 个向量特征
        self.feats_edge_out   = feats_edge_out 
        self.vectors_edge_out = vectors_edge_out # 输出中的 N 个向量特征
        # 辅助层
        self.vector_dim = vector_dim
        # 初始化归一化层
        self.norm = nn.ModuleList([GVPLayerNorm(self.feats_x_out), # + self.feats_edge_out
                                   GVPLayerNorm(self.feats_x_out)])
        # 初始化 dropout 层
        self.dropout = GVPDropout(dropout)
        # 是否使用残差连接
        self.residual = residual
        # 接收 vec_in 消息和接收节点
        self.W_EV = nn.Sequential(GVP(
                                      dim_vectors_in = self.vectors_x_in + self.vectors_edge_in, 
                                      dim_vectors_out = self.vectors_x_out + self.feats_edge_out,
                                      dim_feats_in = self.feats_x_in + self.feats_edge_in, 
                                      dim_feats_out = self.feats_x_out + self.feats_edge_out
                                  ), 
                                  GVP(
                                      dim_vectors_in = self.vectors_x_out + self.feats_edge_out, 
                                      dim_vectors_out = self.vectors_x_out + self.feats_edge_out,
                                      dim_feats_in = self.feats_x_out + self.feats_edge_out,
                                      dim_feats_out = self.feats_x_out + self.feats_edge_out
                                  ),
                                  GVP(
                                      dim_vectors_in = self.vectors_x_out + self.feats_edge_out, 
                                      dim_vectors_out = self.vectors_x_out + self.feats_edge_out,
                                      dim_feats_in = self.feats_x_out + self.feats_edge_out,
                                      dim_feats_out = self.feats_x_out + self.feats_edge_out
                                  ))
        
        # 初始化 W_dh 层
        self.W_dh = nn.Sequential(GVP(
                                      dim_vectors_in = self.vectors_x_out,
                                      dim_vectors_out = 2*self.vectors_x_out,
                                      dim_feats_in = self.feats_x_out,
                                      dim_feats_out = 4*self.feats_x_out
                                  ),
                                  GVP(
                                      dim_vectors_in = 2*self.vectors_x_out,
                                      dim_vectors_out = self.vectors_x_out,
                                      dim_feats_in = 4*self.feats_x_out,
                                      dim_feats_out = self.feats_x_out
                                  ))
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        # 获取输入张量 x 的最后一个维度的大小
        x_size = list(x.shape)[-1]
        # 分别聚合特征和向量
        feats, vectors = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # 聚合
        feats, vectors = self.dropout(feats, vectors.reshape(vectors.shape[0], -1, self.vector_dim))
        # 获取与节点相关的信息 - 不返回边
        feats_nodes  = feats[:, :self.feats_x_in]
        vector_nodes = vectors[:, :self.vectors_x_in]
        # 将向量部分重塑为最后一个 3D
        x_vectors    = x[:, :self.vectors_x_in * self.vector_dim].reshape(x.shape[0], -1, self.vector_dim)
        feats, vectors = self.norm[0]( x[:, self.vectors_x_in * self.vector_dim:]+feats_nodes, x_vectors+vector_nodes )
        # 更新位置感知前馈
        feats_, vectors_ = self.dropout( *self.W_dh( (feats, vectors) ) )
        feats, vectors   = self.norm[1]( feats+feats_, vectors+vectors_ )
        # 使其成为残差
        new_x = torch.cat( [feats, vectors.flatten(start_dim=-2)], dim=-1 )
        if self.residual:
          return new_x + x
        return new_x


    def message(self, x_j, edge_attr) -> Tensor:
        # 拼接特征和边属性
        feats   = torch.cat([ x_j[:, self.vectors_x_in * self.vector_dim:],
                              edge_attr[:, self.vectors_edge_in * self.vector_dim:] ], dim=-1)
        vectors = torch.cat([ x_j[:, :self.vectors_x_in * self.vector_dim], 
                              edge_attr[:, :self.vectors_edge_in * self.vector_dim] ], dim=-1).reshape(x_j.shape[0],-1,self.vector_dim)
        feats, vectors = self.W_EV( (feats, vectors) )
        return feats, vectors.flatten(start_dim=-2)


    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.
        Args:
            adj (Tensor or SparseTensor): `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional): If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        feats, vectors = self.message(**msg_kwargs)
        # 聚合它们
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        out_feats   = self.aggregate(feats, **aggr_kwargs)
        out_vectors = self.aggregate(vectors, **aggr_kwargs)
        # 返回元组
        update_kwargs = self.inspector.distribute('update', coll_dict)
        return self.update((out_feats, out_vectors), **update_kwargs)

        
    def __repr__(self):
        dict_print = { "feats_x_in": self.feats_x_in,
                       "vectors_x_in": self.vectors_x_in,
                       "feats_x_out": self.feats_x_out,
                       "vectors_x_out": self.vectors_x_out,
                       "feats_edge_in": self.feats_edge_in,
                       "vectors_edge_in": self.vectors_edge_in,
                       "feats_edge_out": self.feats_edge_out,
                       "vectors_edge_out": self.vectors_edge_out,
                       "vector_dim": self.vector_dim }
        return  'GVP_MPNN Layer with the following attributes: ' + str(dict_print)
class GVP_Network(nn.Module):
    r"""Sample GNN model architecture that uses the Geometric Vector Perceptron
        message passing layer to learn over point clouds. 
        Main MPNN layer introduced in https://openreview.net/forum?id=1YLJDvSx6J4.

        Inputs will be standard GNN: x, edge_index, edge_attr, batch, ...

        Args:
        * n_layers: int. number of MPNN layers
        * feats_x_in: int. number of scalar dimensions in the x inputs.
        * vectors_x_in: int. number of vector dimensions in the x inputs.
        * feats_x_out: int. number of scalar dimensions in the x outputs.
        * vectors_x_out: int. number of vector dimensions in the x outputs.
        * feats_edge_in: int. number of scalar dimensions in the edge_attr inputs.
        * vectors_edge_in: int. number of vector dimensions in the edge_attr inputs.
        * feats_edge_out: int. number of scalar dimensions in the edge_attr outputs.
        * embedding_nums: list. number of unique keys to embedd. for points
                          1 entry per embedding needed. 
        * embedding_dims: list. point - number of dimensions of
                          the resulting embedding. 1 entry per embedding needed. 
        * edge_embedding_nums: list. number of unique keys to embedd. for edges.
                               1 entry per embedding needed. 
        * edge_embedding_dims: list. point - number of dimensions of
                               the resulting embedding. 1 entry per embedding needed. 
        * vectors_edge_out: int. number of vector dimensions in the edge_attr outputs.
        * dropout: float. dropout rate.
        * vector_dim: int. dimensions of the space containing the vectors.
        * recalc: bool. Whether to recalculate edge features between MPNN layers.
        * verbose: bool. verbosity level.
    """
    # 初始化函数，接受多个参数，包括层数、输入特征和向量、输出特征和向量、边特征和向量等
    def __init__(self, n_layers, 
                       feats_x_in, vectors_x_in,
                       feats_x_out, vectors_x_out,
                       feats_edge_in, vectors_edge_in,
                       feats_edge_out, vectors_edge_out,
                       embedding_nums=[], embedding_dims=[],
                       edge_embedding_nums=[], edge_embedding_dims=[],
                       dropout=0.0, residual=False, vector_dim=3,
                       recalc=1, verbose=False):
        # 调用父类的初始化函数
        super().__init__()

        # 初始化各种属性
        self.n_layers         = n_layers 
        self.embedding_nums   = embedding_nums
        self.embedding_dims   = embedding_dims
        self.emb_layers       = torch.nn.ModuleList()
        self.edge_embedding_nums = edge_embedding_nums
        self.edge_embedding_dims = edge_embedding_dims
        self.edge_emb_layers     = torch.nn.ModuleList()
        
        # 实例化点和边的嵌入层
        for i in range( len(self.embedding_dims) ):
            self.emb_layers.append(nn.Embedding(num_embeddings = embedding_nums[i],
                                                embedding_dim  = embedding_dims[i]))
            feats_x_in += embedding_dims[i] - 1
            feats_x_out += embedding_dims[i] - 1
        for i in range( len(self.edge_embedding_dims) ):
            self.edge_emb_layers.append(nn.Embedding(num_embeddings = edge_embedding_nums[i],
                                                     embedding_dim  = edge_embedding_dims[i]))
            feats_edge_in += edge_embedding_dims[i] - 1
            feats_edge_out += edge_embedding_dims[i] - 1
        
        # 初始化其他属性
        self.fc_layers        = torch.nn.ModuleList()
        self.gcnn_layers      = torch.nn.ModuleList()
        self.feats_x_in       = feats_x_in
        self.vectors_x_in     = vectors_x_in
        self.feats_x_out      = feats_x_out
        self.vectors_x_out    = vectors_x_out
        self.feats_edge_in    = feats_edge_in
        self.vectors_edge_in  = vectors_edge_in
        self.feats_edge_out   = feats_edge_out
        self.vectors_edge_out = vectors_edge_out
        self.dropout          = dropout
        self.residual         = residual
        self.vector_dim       = vector_dim
        self.recalc           = recalc
        self.verbose          = verbose
        
        # 实例化GCNN层
        for i in range(n_layers):
            layer = GVP_MPNN(feats_x_in, vectors_x_in,
                             feats_x_out, vectors_x_out,
                             feats_edge_in, vectors_edge_in,
                             feats_edge_out, vectors_edge_out,
                             dropout, residual=residual,
                             vector_dim=vector_dim, verbose=verbose)
            self.gcnn_layers.append(layer)
    # 定义一个前向传播函数，接受输入 x、边索引 edge_index、批次 batch、边属性 edge_attr
    # bsize 为批次大小，recalc_edge 为重新计算边特征的函数，verbose 为是否输出详细信息的标志
    def forward(self, x, edge_index, batch, edge_attr,
                bsize=None, recalc_edge=None, verbose=0):
        """ Embedding of inputs when necessary, then pass layers.
            Recalculate edge features every time with the
            `recalc_edge` function.
        """
        # 复制输入数据，用于后续恢复原始数据
        original_x = x.clone()
        original_edge_index = edge_index.clone()
        original_edge_attr = edge_attr.clone()
        
        # 当需要时进行嵌入
        # 选择要嵌入的部分，逐个进行嵌入并添加到输入中
        
        # 提取要嵌入的部分
        to_embedd = x[:, -len(self.embedding_dims):].long()
        for i, emb_layer in enumerate(self.emb_layers):
            # 在第一次迭代时，对应于 `to_embedd` 部分的部分会被丢弃
            stop_concat = -len(self.embedding_dims) if i == 0 else x.shape[-1]
            x = torch.cat([x[:, :stop_concat], 
                           emb_layer(to_embedd[:, i])], dim=-1)
        
        # 传递层
        for i, layer in enumerate(self.gcnn_layers):
            # 嵌入边属性（每次都需要，因为边属性和索引在每次传递时都会重新计算）
            to_embedd = edge_attr[:, -len(self.edge_embedding_dims):].long()
            for j, edge_emb_layer in enumerate(self.edge_emb_layers):
                # 在第一次迭代时，对应于 `to_embedd` 部分的部分会被丢弃
                stop_concat = -len(self.edge_embedding_dims) if j == 0 else x.shape[-1]
                edge_attr = torch.cat([edge_attr[:, :-len(self.edge_embedding_dims) + j], 
                                       edge_emb_layer(to_embedd[:, j])], dim=-1)
            
            # 传递层
            x = layer(x, edge_index, edge_attr, size=bsize)

            # 每 self.recalc 步重新计算边信息
            # 但如果是最后一层的最后一次迭代，则不需要重新计算
            if (1 % self.recalc == 0) and not (i == self.n_layers - 1):
                edge_index, edge_attr, _ = recalc_edge(x)  # 返回属性、索引、嵌入信息
            else:
                edge_attr = original_edge_attr.clone()
                edge_index = original_edge_index.clone()
            
            if verbose:
                print("========")
                print("iter:", j, "layer:", i, "nlinks:", edge_attr.shape)
        
        return x

    # 定义对象的字符串表示形式
    def __repr__(self):
        return 'GVP_Network of: {0} layers'.format(len(self.gcnn_layers))
```