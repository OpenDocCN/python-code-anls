# `.\lucidrains\tr-rosetta-pytorch\tr_rosetta_pytorch\utils.py`

```py
# 导入所需的库
import string
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# 定义函数d，用于确定张量所在的设备（CPU或CUDA）
def d(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

# 解析A3M文件并将字母转换为0到20的整数
def parse_a3m(filename):
    # 创建字母表转换表，将小写字母转换为空格
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    # 读取A3M文件中的序列并进行转换
    seqs = [line.strip().translate(table) for line in open(filename, 'r') if line[0] != '>']
    # 创建氨基酸字母表和MSA矩阵
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)

    # 将字母转换为数字
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # 将所有未知字符视为间隔
    msa[msa > 20] = 20
    return msa

# 将1-hot MSA转换为PSSM
def msa2pssm(msa1hot, w):
    beff = w.sum()
    f_i = (w[:, None, None] * msa1hot).sum(dim=0) / beff + 1e-9
    h_i = (-f_i * torch.log(f_i)).sum(dim=1)
    return torch.cat((f_i, h_i[:, None]), dim=1)

# 根据截断值重新加权MSA
def reweight(msa1hot, cutoff):
    id_min = msa1hot.shape[1] * cutoff
    id_mtx = torch.einsum('ikl,jkl->ij', msa1hot, msa1hot)
    id_mask = id_mtx > id_min
    w = 1. / id_mask.float().sum(dim=-1)
    return w

# 快速DCA（Direct Coupling Analysis）缩减协方差矩阵求逆
def fast_dca(msa1hot, weights, penalty = 4.5):
    device = msa1hot.device
    nr, nc, ns = msa1hot.shape
    x = msa1hot.view(nr, -1)
    num_points = weights.sum() - torch.sqrt(weights.mean())

    mean = (x * weights[:, None]).sum(dim=0, keepdims=True) / num_points
    x = (x - mean) * torch.sqrt(weights[:, None])

    cov = (x.t() @ x) / num_points
    cov_reg = cov + torch.eye(nc * ns).to(device) * penalty / torch.sqrt(weights.sum())

    inv_cov = torch.inverse(cov_reg)
    x1 = inv_cov.view(nc, ns, nc, ns)
    x2 = x1.transpose(1, 2).contiguous()
    features = x2.reshape(nc, nc, ns * ns)

    x3 = torch.sqrt((x1[:, :-1, :, :-1] ** 2).sum(dim=(1, 3))) * (1 - torch.eye(nc).to(device))
    apc = x3.sum(dim=0, keepdims=True) * x3.sum(dim=1, keepdims=True) / x3.sum()
    contacts = (x3 - apc) * (1 - torch.eye(nc).to(device))
    return torch.cat((features, contacts[:, :, None]), dim=2)

# 预处理函数，将MSA文件转换为适用于神经网络的输入
def preprocess(msa_file, wmin=0.8, ns=21):
    a3m = torch.from_numpy(parse_a3m(msa_file)).long()
    nrow, ncol = a3m.shape

    msa1hot = F.one_hot(a3m, ns).float().to(d())
    w = reweight(msa1hot, wmin).float().to(d())

    # 1D序列特征
    f1d_seq = msa1hot[0, :, :20].float()
    f1d_pssm = msa2pssm(msa1hot, w)

    f1d = torch.cat((f1d_seq, f1d_pssm), dim=1)
    f1d = f1d[None, :, :].reshape((1, ncol, 42))

    # 2D序列特征
    f2d_dca = fast_dca(msa1hot, w) if nrow > 1 else torch.zeros((ncol, ncol, 442)).float().to(d())
    f2d_dca = f2d_dca[None, :, :, :]

    f2d = torch.cat((
        f1d[:, :, None, :].repeat(1, 1, ncol, 1), 
        f1d[:, None, :, :].repeat(1, ncol, 1, 1),
        f2d_dca
    ), dim=-1)

    f2d = f2d.view(1, ncol, ncol, 442 + 2*42)
    return f2d.permute((0, 3, 2, 1))
```