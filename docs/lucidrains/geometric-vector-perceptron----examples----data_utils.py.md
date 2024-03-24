# `.\lucidrains\geometric-vector-perceptron\examples\data_utils.py`

```
# 作者：Eric Alcaide

# 导入必要的库
import os 
import sys
# 科学计算库
import torch
import torch_sparse
import numpy as np 
from einops import repeat, rearrange
# 导入自定义工具 - 来自 https://github.com/EleutherAI/mp_nerf
from data_handler import *

# 新数据构建函数
def get_atom_ids_dict():
    """ 获取将每个原子映射到一个标记的字典 """
    # 初始化原子集合
    ids = set(["N", "CA", "C", "O"])

    # 遍历 SC_BUILD_INFO 中的键值对
    for k,v in SC_BUILD_INFO.items():
        # 遍历原子名称列表
        for name in v["atom-names"]:
            ids.add(name)
            
    # 返回原子到标记的映射字典
    return {k: i for i,k in enumerate(sorted(ids))}

#################################
##### 原始项目数据 #####
#################################

# 定义氨基酸序列和对应的数字
AAS = "ARNDCQEGHILKMFPSTWYV_"
AAS2NUM = {k: AAS.index(k) for k in AAS}
# 获取原子标记字典
ATOM_IDS = get_atom_ids_dict()
# 定义氨基酸的键值对应的键为氨基酸，值为键值对应的原子键值对
GVP_DATA = { 
    'A': {
        'bonds': [[0,1], [1,2], [2,3], [1,4]] 
         },
    'R': {
        'bonds': [[0,1], [1,2], [2,3], [2,4], [4,5], [5,6],
                  [6,7], [7,8], [8,9], [8,10]] 
         },
    # 其他氨基酸的键值对应的键值对
    # ...
    '_': {
        'bonds': []
        }
    }

#################################
##### 原始项目数据 #####
#################################

def graph_laplacian_embedds(edges, eigen_k, center_idx=1, norm=False):
    """ 返回图拉普拉斯的前 K 个特征向量中点的嵌入。
        输入：
        * edges: (2, N). 长整型张量或列表。足够表示无向边。
        * eigen_k: 整数。要返回嵌入的前 N 个特征向量。
        * center_idx: 整数。用作嵌入中心的索引。
        * norm: 布尔值。是否使用归一化拉普拉斯。不建议使用。
        输出：(n_points, eigen_k)
    """
    # 如果 edges 是列表，则转换为长整型张量
    if isinstance(edges, list):
        edges = torch.tensor(edges).long()
        # 纠正维度
        if edges.shape[0] != 2:
            edges = edges.t()
        # 如果为空，则返回零张量
        if edges.shape[0] == 0:
            return torch.zeros(1, eigen_k)
    # 获取参数
    # 计算边的最大值并加1，作为邻接矩阵的大小
    size = torch.max(edges)+1
    # 获取边所在设备信息
    device = edges.device
    # 创建邻接矩阵
    adj_mat = torch.eye(size, device=device) 
    # 遍历边的起始点和终点，将邻接矩阵对应位置设为1
    for i,j in edges.t():
        adj_mat[i,j] = adj_mat[j,i] = 1.
        
    # 计算度矩阵
    deg_mat = torch.eye(size) * adj_mat.sum(dim=-1, keepdim=True)
    # 计算拉普拉斯矩阵
    laplace = deg_mat - adj_mat
    # 如果传入了norm参数，则使用规范化的拉普拉斯矩阵
    if norm:
        # 遍历边的起始点和终点，更新拉普拉斯矩阵的值
        for i,j in edges.t():
            laplace[i,j] = laplace[j,i] = -1 / (deg_mat[i,i] * deg_mat[j,j])**0.5
    # 对拉普拉斯矩阵进行特征值分解，获取特征值和特征向量
    e, v = torch.symeig(laplace, eigenvectors=True)
    # 根据特征值的绝对值降序排列，获取排序后的索引
    idxs = torch.sort( e.abs(), descending=True)[1]
    # 获取前eigen_k个特征向量作为嵌入向量
    embedds = v[:, idxs[:eigen_k]]
    # 将嵌入向量减去中心点的嵌入向量
    embedds = embedds - embedds[center_idx].unsqueeze(-2)
    # 返回处理后的嵌入向量
    return embedds
# 返回每个氨基酸中每个原子的标记
def make_atom_id_embedds(k):
    # 创建一个长度为14的零张量
    mask = torch.zeros(14).long()
    # 定义氨基酸中的原子列表
    atom_list = ["N", "CA", "C", "O"] + SC_BUILD_INFO[k]["atom-names"]
    # 遍历原子列表，将每个原子的标记存储在mask中
    for i,atom in enumerate(atom_list):
        mask[i] = ATOM_IDS[atom]
    return mask


#################################
########## SAVE INFO ############
#################################

# 创建包含各种信息的字典
SUPREME_INFO = {k: {"cloud_mask": make_cloud_mask(k),
                    "bond_mask": make_bond_mask(k),
                    "theta_mask": make_theta_mask(k),
                    "torsion_mask": make_torsion_mask(k),
                    "idx_mask": make_idx_mask(k),
                    #
                    "eigen_embedd": graph_laplacian_embedds(GVP_DATA[k]["bonds"], eigen_k = 3),
                    "atom_id_embedd": make_atom_id_embedds(k)
                    } 
                for k in "ARNDCQEGHILKMFPSTWYV_"}

#################################
######### RANDOM UTILS ##########
#################################


# 使用正弦和余弦编码距离
def encode_dist(x, scales=[1,2,4,8], include_self = True):
    """ Encodes a distance with sines and cosines. 
        Inputs:
        * x: (batch, N) or (N,). data to encode.
              Infer devic and type (f16, f32, f64) from here.
        * scales: (s,) or list. lower or higher depending on distances.
        Output: (..., num_scales*2 + 1) if include_self or (..., num_scales*2) 
    """
    x = x.unsqueeze(-1)
    # 推断设备
    device, precise = x.device, x.type()
    # 转换为张量
    if isinstance(scales, list):
        scales = torch.tensor([scales], device=device).type(precise)
    # 获取正弦编码
    sines   = torch.sin(x / scales)
    cosines = torch.cos(x / scales)
    # 连接并返回
    enc_x = torch.cat([sines, cosines], dim=-1)
    return torch.cat([enc_x, x], dim=-1) if include_self else enc_x

# 解码距离
def decode_dist(x, scales=[1,2,4,8], include_self = False):
    """ Encodes a distance with sines and cosines. 
        Inputs:
        * x: (batch, N, 2*fourier_feats (+1) ) or (N,). data to encode.
              Infer devic and type (f16, f32, f64) from here.
        * scales: (s,) or list. lower or higher depending on distances.
        * include_self: whether to average with raw prediction or not.
        Output: (batch, N)
    """
    device, precise = x.device, x.type()
    # 转换为张量
    if isinstance(scales, list):
        scales = torch.tensor([scales], device=device).type(precise)
    # 通过 atan2 解码并校正负角度
    half = x.shape[-1]//2
    decodes = torch.atan2(x[..., :half], x[..., half:2*half])
    decodes += (decodes<0).type(precise) * 2*np.pi 
    # 调整偏移量
    offsets = torch.zeros_like(decodes)
    for i in range(decodes.shape[-1]-1, 0, -1):
        offsets[:, i-1] = 2 * ( offsets[:, i] + (decodes[:, i]>np.pi).type(precise) * np.pi )
    decodes += offsets
    avg_dec = (decodes * scales).mean(dim=-1, keepdim=True)
    if include_self:
        return 0.5*(avg_dec + x[..., -1:])
    return avg_dec

# 计算第n次邻接矩阵
def nth_deg_adjacency(adj_mat, n=1, sparse=False):
    """ Calculates the n-th degree adjacency matrix.
        Performs mm of adj_mat and adds the newly added.
        Default is dense. Mods for sparse version are done when needed.
        Inputs: 
        * adj_mat: (N, N) adjacency tensor
        * n: int. degree of the output adjacency
        * sparse: bool. whether to use torch-sparse module
        Outputs: 
        * edge_idxs: the ij positions of the adjacency matrix
        * edge_attrs: the degree of connectivity (1 for neighs, 2 for neighs^2 )
    """
    adj_mat = adj_mat.float()
    attr_mat = torch.zeros_like(adj_mat)
    # 遍历范围为n的循环
    for i in range(n):
        # 如果i为0，则将属性矩阵与邻接矩阵相加
        if i == 0:
            attr_mat += adj_mat
            continue

        # 如果i为1且sparse为True
        if i == 1 and sparse: 
            # 创建稀疏邻接张量
            adj_mat = torch.sparse.FloatTensor(adj_mat.nonzero().t(),
                                                adj_mat[adj_mat != 0]).to(adj_mat.device).coalesce()
            idxs, vals = adj_mat.indices(), adj_mat.values()
            m, k, n = 3 * [adj_mat.shape[0]]  # (m, n) * (n, k) , 但adj_mats是方阵：m=n=k

        # 如果sparse为True
        if sparse:
            # 使用torch_sparse库中的spspmm函数进行稀疏矩阵乘法
            idxs, vals = torch_sparse.spspmm(idxs, vals, idxs, vals, m=m, k=k, n=n)
            adj_mat = torch.zeros_like(attr_mat)
            adj_mat[idxs[0], idxs[1]] = vals.bool().float()
        else:
            # 如果sparse为False，则将邻接矩阵平方，转换为布尔型矩阵
            adj_mat = (adj_mat @ adj_mat).bool().float() 

        # 更新属性矩阵
        attr_mat[(adj_mat - attr_mat.bool().float()).bool()] += i + 1

    # 返回更新后的邻接矩阵和属性矩阵
    return adj_mat, attr_mat
# 返回蛋白质的共价键的索引
def prot_covalent_bond(seq, adj_degree=1, cloud_mask=None):
    """ 返回蛋白质的共价键的索引。
        输入
        * seq: str. 用1字母氨基酸代码表示的蛋白质序列。
        * cloud_mask: 选择存在原子的掩码。
        输出: edge_idxs
    """
    # 创建或推断 cloud_mask
    if cloud_mask is None: 
        cloud_mask = scn_cloud_mask(seq).bool()
    device, precise = cloud_mask.device, cloud_mask.type()
    # 获取每个氨基酸的起始位置
    scaff = torch.zeros_like(cloud_mask)
    scaff[:, 0] = 1
    idxs = scaff[cloud_mask].nonzero().view(-1)
    # 从包含 GVP_DATA 的字典中获取姿势 + 索引 - 返回所有边
    adj_mat = torch.zeros(idxs.amax()+14, idxs.amax()+14)
    for i,idx in enumerate(idxs):
        # 与下一个氨基酸的键
        extra = []
        if i < idxs.shape[0]-1:
            extra = [[2, (idxs[i+1]-idx).item()]]

        bonds = idx + torch.tensor( GVP_DATA[seq[i]]['bonds'] + extra ).long().t() 
        adj_mat[bonds[0], bonds[1]] = 1.
    # 转换为无向图
    adj_mat = adj_mat + adj_mat.t()
    # 进行 N 次邻接
    adj_mat, attr_mat = nth_deg_adjacency(adj_mat, n=adj_degree, sparse=True)

    edge_idxs = attr_mat.nonzero().t().long()
    edge_attrs = attr_mat[edge_idxs[0], edge_idxs[1]]
    return edge_idxs, edge_attrs


def dist2ca(x, mask=None, eps=1e-7):
    """ 计算每个点到 C-alfa 的距离。
        输入:
        * x: (L, 14, D)
        * mask: (L, 14) 的布尔掩码
        返回单位向量和范数。
    """
    x = x - x[:, 1].unsqueeze(1)
    norm = torch.norm(x, dim=-1, keepdim=True)
    x_norm = x / (norm+eps)
    if mask:
        return x_norm[mask], norm[mask]
    return x_norm, norm


def orient_aa(x, mask=None, eps=1e-7):
    """ 计算主链特征的单位向量和范数。
        输入:
        * x: (L, 14, D). Sidechainnet 格式的坐标。
        返回单位向量 (5) 和范数 (3)。
    """
    # 获取张量信息
    device, precise = x.device, x.type()

    vec_wrap  = torch.zeros(5, x.shape[0], 3, device=device) # (feats, L, dims+1)
    norm_wrap = torch.zeros(3, x.shape[0], device=device)
    # 第一个特征是 CB-CA
    vec_wrap[0]  = x[:, 4] - x[:, 1]
    norm_wrap[0] = torch.norm(vec_wrap[0], dim=-1)
    vec_wrap[0] /= norm_wrap[0].unsqueeze(dim=-1) + eps
    # 第二个是 CA+ - CA :
    vec_wrap[1, :-1]  = x[:-1, 1] - x[1:, 1]
    norm_wrap[1, :-1] = torch.norm(vec_wrap[1, :-1], dim=-1)
    vec_wrap[1, :-1] /= norm_wrap[1, :-1].unsqueeze(dim=-1) + eps
    # 同样但是反向向量
    vec_wrap[2] = (-1)*vec_wrap[1]
    # 第三个是 CA - CA-
    vec_wrap[3, 1:]  = x[:-1, 1] - x[1:, 1]
    norm_wrap[2, 1:] = torch.norm(vec_wrap[3, 1:], dim=-1)
    vec_wrap[3, 1:] /= norm_wrap[2, 1:].unsqueeze(dim=-1) + eps
    # 现在反向顺序的向量
    vec_wrap[4] = (-1)*vec_wrap[3]

    return vec_wrap, norm_wrap


def chain2atoms(x, mask=None):
    """ 从 (L, other) 扩展到 (L, C, other)。"""
    device, precise = x.device, x.type()
    # 获取掩码
    wrap = torch.ones(x.shape[0], 14, *x.shape[1:]).type(precise).to(device)
    # 分配
    wrap = wrap * x.unsqueeze(1)
    if mask is not None:
        return wrap[mask]
    return wrap


def from_encode_to_pred(whole_point_enc, use_fourier=False, embedd_info=None, needed_info=None, vec_dim=3):
    """ 将上述函数的编码转换为标签/预测格式。
        仅包含位置恢复所需的基本信息 (径向单位向量 + 范数)
        输入: 包含以下内容的输入元组:
        * whole_point_enc: (atoms, vector_dims+scalar_dims)
                           与上述函数相同的形状。
                           径向单位向量必须是第一个向量维度
        * embedd_info: 字典。包含标量和向量特征的数量。
    """
    vec_dims = vec_dim * embedd_info["point_n_vectors"]
    start_pos = 2*len(needed_info["atom_pos_scales"])+vec_dims
    # 如果使用傅立叶变换
    if use_fourier:
        # 解码整个点编码中的部分向量维度，不包括自身
        decoded_dist = decode_dist(whole_point_enc[:, vec_dims:start_pos+1],
                                    scales=needed_info["atom_pos_scales"],
                                    include_self=False)
    else:
        # 如果不使用傅立叶变换，直接取整个点编码中的特定维度
        decoded_dist = whole_point_enc[:, start_pos:start_pos+1]
    # 返回连接后的张量，包括单位径向向量和向量范数
    return torch.cat([whole_point_enc[:, :3], decoded_dist], dim=-1)
def encode_whole_bonds(x, x_format="coords", embedd_info={},
                       needed_info = {"cutoffs": [2,5,10],
                                      "bond_scales": [.5, 1, 2],
                                      "adj_degree": 1},
                       free_mem=False, eps=1e-7):
    """ Given some coordinates, and the needed info,
        encodes the bonds from point information.
        * x: (N, 3) or prediction format
        * x_format: one of ["coords" or "prediction"]
        * embedd_info: dict. contains the needed embedding info
        * needed_info: dict. contains additional needed info
            { cutoffs: list. cutoff distances for bonds.
                       can be a string for the k closest (ex: "30_closest"),
                       empty list for just covalent.
              bond_scales: list. fourier encodings
              adj_degree: int. degree of adj (2 means adj of adj is my adj)
                               0 for no adjacency
            }
        * free_mem: whether to delete variables
        * eps: constant for numerical stability
    """ 
    device, precise = x.device, x.type()
    # convert to 3d coords if passed as preds
    if x_format == "encode":
        pred_x = from_encode_to_pred(x, embedd_info=embedd_info, needed_info=needed_info)
        x = pred_x[:, :3] * pred_x[:, 3:4]

    # encode bonds

    # 1. BONDS: find the covalent bond_indices - allow arg -> DRY
    native = None
    if "prot_covalent_bond" in needed_info.keys():
        native = True
        native_bonds = needed_info["covalent_bond"]
    elif needed_info["adj_degree"]:
        native = True
        native_bonds  = prot_covalent_bond(needed_info["seq"], needed_info["adj_degree"])
        
    if native: 
        native_idxs, native_attrs = native_bonds[0].to(device), native_bonds[1].to(device)

    # determine kind of cutoff (hard distance threshold or closest points)
    closest = None
    if len(needed_info["cutoffs"]) > 0: 
        cutoffs = needed_info["cutoffs"].copy() 
        if sum( isinstance(ci, str) for ci in cutoffs ) > 0:
            cutoffs = [-1e-3] # negative so no bond is taken  
            closest = int( needed_info["cutoffs"][0].split("_")[0] ) 

        # points under cutoff = d(i - j) < X 
        cutoffs = torch.tensor(cutoffs, device=device).type(precise)
        dist_mat = torch.cdist(x, x, p=2)

    # normal buckets
    bond_buckets = torch.zeros(*x.shape[:-1], x.shape[-2], device=device).type(precise)
    if len(needed_info["cutoffs"]) > 0 and not closest:
        # count from latest degree of adjacency given
        bond_buckets = torch.bucketize(dist_mat, cutoffs)
        bond_buckets[native_idxs[0], native_idxs[1]] = cutoffs.shape[0]
        # find the indexes - symmetric and we dont want the diag
        bond_buckets   += cutoffs.shape[0] * torch.eye(bond_buckets.shape[0], device=device).long()
        close_bond_idxs = ( bond_buckets < cutoffs.shape[0] ).nonzero().t()
        # move away from poses reserved for native
        bond_buckets[close_bond_idxs[0], close_bond_idxs[1]] += needed_info["adj_degree"]+1

    # the K closest (covalent bonds excluded) are considered bonds 
    # 如果存在最近的键，执行以下操作
    elif closest:
        # 将距离矩阵复制一份，并将共价键屏蔽掉
        masked_dist_mat = dist_mat.clone()
        masked_dist_mat += torch.eye(masked_dist_mat.shape[0], device=device) * torch.amax(masked_dist_mat)
        masked_dist_mat[native_idxs[0], native_idxs[1]] = masked_dist_mat[0,0].clone()
        # 根据距离排序，*(-1)使得最小值在前
        _, sorted_col_idxs = torch.topk(-masked_dist_mat, k=k, dim=-1)
        # 连接索引并重复行索引以匹配列索引的数量
        sorted_col_idxs = rearrange(sorted_col_idxs[:, :k], '... n k -> ... (n k)')
        sorted_row_idxs = torch.repeat_interleave( torch.arange(dist_mat.shape[0]).long(), repeats=k ).to(device)
        close_bond_idxs = torch.stack([ sorted_row_idxs, sorted_col_idxs ], dim=0)
        # 将远离保留给原生的姿势
        bond_buckets = torch.ones_like(dist_mat) * (needed_info["adj_degree"]+1)

    # 合并所有键
    if len(needed_info["cutoffs"]) > 0:
        if close_bond_idxs.shape[0] > 0:
            whole_bond_idxs = torch.cat([native_idxs, close_bond_idxs], dim=-1)
    else:
        whole_bond_idxs = native_idxs

    # 2. ATTRS: 将键编码为属性
    bond_vecs  = x[ whole_bond_idxs[0] ] - x[ whole_bond_idxs[1] ]
    bond_norms = torch.norm(bond_vecs, dim=-1)
    bond_vecs /= (bond_norms + eps).unsqueeze(-1)
    bond_norms_enc = encode_dist(bond_norms, scales=needed_info["bond_scales"]).squeeze()

    if native:
        bond_buckets[native_idxs[0], native_idxs[1]] = native_attrs
    bond_attrs = bond_buckets[whole_bond_idxs[0] , whole_bond_idxs[1]]
    # 打包标量和向量 - 额外的令牌用于共价键
    bond_n_vectors = 1
    bond_n_scalars = (2 * len(needed_info["bond_scales"]) + 1) + 1 # 最后一个是大小为1+len(cutoffs)的嵌入
    whole_bond_enc = torch.cat([bond_vecs, # 1个向量 - 不需要反转 - 我们做2倍的键（对称性）
                                # 标量
                                bond_norms_enc, # 2 * len(scales)
                                (bond_attrs-1).unsqueeze(-1) # 1 
                               ], dim=-1) 
    # 释放 GPU 内存
    if free_mem:
        del bond_buckets, bond_norms_enc, bond_vecs, dist_mat,\
            close_bond_idxs, native_bond_idxs
        if closest: 
            del masked_dist_mat, sorted_col_idxs, sorted_row_idxs

    embedd_info = {"bond_n_vectors": bond_n_vectors, 
                   "bond_n_scalars": bond_n_scalars, 
                   "bond_embedding_nums": [ len(needed_info["cutoffs"]) + needed_info["adj_degree"] ]} # 额外一个用于共价键（默认）

    return whole_bond_idxs, whole_bond_enc, embedd_info
def encode_whole_protein(seq, true_coords, angles, padding_seq,
                         needed_info = { "cutoffs": [2, 5, 10],
                                          "bond_scales": [0.5, 1, 2]}, free_mem=False):
    """ Encodes a whole protein. In points + vectors. """
    # 获取设备和数据类型
    device, precise = true_coords.device, true_coords.type()
    #################
    # encode points #
    #################
    # 创建云掩码
    cloud_mask = torch.tensor(scn_cloud_mask(seq[:-padding_seq or None])).bool().to(device)
    flat_mask = rearrange(cloud_mask, 'l c -> (l c)')
    # 嵌入所有内容

    # 一般位置嵌入
    center_coords = true_coords - true_coords.mean(dim=0)
    pos_unit_norms = torch.norm(center_coords, dim=-1, keepdim=True)
    pos_unit_vecs  = center_coords / pos_unit_norms
    pos_unit_norms_enc = encode_dist(pos_unit_norms, scales=needed_info["atom_pos_scales"]).squeeze()
    # 重新格式化坐标到scn (L, 14, 3) - 待解决如果填充=0
    coords_wrap = rearrange(center_coords, '(l c) d -> l c d', c=14)[:-padding_seq or None] 

    # 蛋白质中的位置嵌入
    aa_pos = encode_dist( torch.arange(len(seq[:-padding_seq or None]), device=device).float(), scales=needed_info["aa_pos_scales"])
    atom_pos = chain2atoms(aa_pos)[cloud_mask]

    # 原子标识嵌入
    atom_id_embedds = torch.stack([SUPREME_INFO[k]["atom_id_embedd"] for k in seq[:-padding_seq or None]], 
                                  dim=0)[cloud_mask].to(device)
    # 氨基酸嵌入
    seq_int = torch.tensor([AAS2NUM[aa] for aa in seq[:-padding_seq or None]], device=device).long()
    aa_id_embedds   = chain2atoms(seq_int, mask=cloud_mask)

    # CA - SC 距离
    dist2ca_vec, dist2ca_norm = dist2ca(coords_wrap) 
    dist2ca_norm_enc = encode_dist(dist2ca_norm, scales=needed_info["dist2ca_norm_scales"]).squeeze()

    # 主链特征
    vecs, norms    = orient_aa(coords_wrap)
    bb_vecs_atoms  = chain2atoms(torch.transpose(vecs, 0, 1), mask=cloud_mask)
    bb_norms_atoms = chain2atoms(torch.transpose(norms, 0, 1), mask=cloud_mask)
    bb_norms_atoms_enc = encode_dist(bb_norms_atoms, scales=[0.5])

    ################
    # encode bonds #
    ################
    bond_info = encode_whole_bonds(x = coords_wrap[cloud_mask],
                                   x_format = "coords",
                                   embedd_info = {},
                                   needed_info = needed_info )
    whole_bond_idxs, whole_bond_enc, bond_embedd_info = bond_info
    #########
    # merge #
    #########

    # 连接以使最终为[矢量维度，标量维度]
    point_n_vectors = 1 + 1 + 5
    point_n_scalars = 2*len(needed_info["atom_pos_scales"]) + 1 +\
                      2*len(needed_info["aa_pos_scales"]) + 1 +\
                      2*len(needed_info["dist2ca_norm_scales"]) + 1+\
                      rearrange(bb_norms_atoms_enc, 'atoms feats encs -> atoms (feats encs)').shape[1] +\
                      2 # 最后2个尚未嵌入

    whole_point_enc = torch.cat([ pos_unit_vecs[ :-padding_seq*14 or None ][ flat_mask ], # 1
                                  dist2ca_vec[cloud_mask], # 1
                                  rearrange(bb_vecs_atoms, 'atoms n d -> atoms (n d)'), # 5
                                  # 标量
                                  pos_unit_norms_enc[ :-padding_seq*14 or None ][ flat_mask ], # 2n+1
                                  atom_pos, # 2n+1
                                  dist2ca_norm_enc[cloud_mask], # 2n+1
                                  rearrange(bb_norms_atoms_enc, 'atoms feats encs -> atoms (feats encs)'), # 2n+1
                                  atom_id_embedds.unsqueeze(-1),
                                  aa_id_embedds.unsqueeze(-1) ], dim=-1) # 最后2个尚未嵌入
    if free_mem:
        del pos_unit_vecs, dist2ca_vec, bb_vecs_atoms, pos_unit_norms_enc, cloud_mask,\
            atom_pos, dist2ca_norm_enc, bb_norms_atoms_enc, atom_id_embedds, aa_id_embedds
    # 记录嵌入维度信息，包括点向量数量和标量数量
    point_embedd_info = {"point_n_vectors": point_n_vectors,
                         "point_n_scalars": point_n_scalars,}

    # 合并点和键的嵌入信息
    embedd_info = {**point_embedd_info, **bond_embedd_info}

    # 返回整体点编码、整体键索引、整体键编码和嵌入信息
    return whole_point_enc, whole_bond_idxs, whole_bond_enc, embedd_info
def get_prot(dataloader_=None, vocab_=None, min_len=80, max_len=150, verbose=True):
    """ Gets a protein from sidechainnet and returns
        the right attrs for training. 
        Inputs: 
        * dataloader_: sidechainnet iterator over dataset
        * vocab_: sidechainnet VOCAB class
        * min_len: int. minimum sequence length
        * max_len: int. maximum sequence length
        * verbose: bool. verbosity level
    """
    # 遍历数据加载器中的训练数据批次
    for batch in dataloader_['train']:
        # 尝试在两个循环中同时中断
        try:
            # 遍历当前批次中的序列
            for i in range(batch.int_seqs.shape[0]):
                # 获取变量
                seq     = ''.join([vocab_.int2char(aa) for aa in batch.int_seqs[i].numpy()])
                int_seq = batch.int_seqs[i]
                angles  = batch.angs[i]
                mask    = batch.msks[i]
                # 获取填充
                padding_angles = (torch.abs(angles).sum(dim=-1) == 0).long().sum()
                padding_seq    = (batch.int_seqs[i] == 20).sum()
                # 仅接受具有正确维度且没有缺失坐标的序列
                # 大于0以避免后续负索引错误
                if batch.crds[i].shape[0]//14 == int_seq.shape[0]:
                    if ( max_len > len(seq) and len(seq) > min_len ) and padding_seq == padding_angles: 
                        if verbose:
                            print("stopping at sequence of length", len(seq))
                            # print(len(seq), angles.shape, "paddings: ", padding_seq, padding_angles)
                        # 触发 StopIteration 异常
                        raise StopIteration
                    else:
                        # print("found a seq of length:", len(seq),
                        #        "but oustide the threshold:", min_len, max_len)
                        pass
        except StopIteration:
            # 中断外部循环
            break
            
    # 返回序列、坐标、角度、填充序列、掩码和蛋白质ID
    return seq, batch.crds[i], angles, padding_seq, batch.msks[i], batch.pids[i]
```