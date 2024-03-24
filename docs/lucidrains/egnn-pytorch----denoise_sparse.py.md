# `.\lucidrains\egnn-pytorch\denoise_sparse.py`

```py
# 导入所需的库
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat

# 导入 sidechainnet 和 egnn_pytorch 库
import sidechainnet as scn
from egnn_pytorch.egnn_pytorch import EGNN_Network

# 设置默认的张量数据类型为 float64
torch.set_default_dtype(torch.float64)

# 定义批量大小和梯度累积次数
BATCH_SIZE = 1
GRADIENT_ACCUMULATE_EVERY = 16

# 定义一个循环生成器函数，用于处理数据加载器中的数据
def cycle(loader, len_thres = 200):
    while True:
        for data in loader:
            # 如果序列长度超过指定阈值，则跳过
            if data.seqs.shape[1] > len_thres:
                continue
            yield data

# 创建 EGNN 网络模型
net = EGNN_Network(
    num_tokens = 21,
    num_positions = 200 * 3,   # 最大位置数 - 绝对位置嵌入，因为序列中存在固有顺序
    depth = 5,
    dim = 8,
    num_nearest_neighbors = 16,
    fourier_features = 2,
    norm_coors = True,
    coor_weights_clamp_value = 2.
).cuda()

# 加载数据集
data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = BATCH_SIZE,
    dynamic_batching = False
)

# 创建数据加载器循环
dl = cycle(data['train'])
# 初始化优化器
optim = Adam(net.parameters(), lr=1e-3)

# 进行训练循环
for _ in range(10000):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        # 获取一个批次的数据
        batch = next(dl)
        seqs, coords, masks = batch.seqs, batch.crds, batch.msks

        # 将序列转移到 GPU 并获取最大值索引
        seqs = seqs.cuda().argmax(dim = -1)
        coords = coords.cuda().type(torch.float64)
        masks = masks.cuda().bool()

        # 获取序列长度并重新排列坐标
        l = seqs.shape[1]
        coords = rearrange(coords, 'b (l s) c -> b l s c', s = 14)

        # 保留主干坐标
        coords = coords[:, :, 0:3, :]
        coords = rearrange(coords, 'b l s c -> b (l s) c')

        # 重复序列和掩码
        seq = repeat(seqs, 'b n -> b (n c)', c = 3)
        masks = repeat(masks, 'b n -> b (n c)', c = 3)

        # 创建邻接矩阵
        i = torch.arange(seq.shape[-1], device = seq.device)
        adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

        # 添加噪声到坐标
        noised_coords = coords + torch.randn_like(coords)

        # 使用 EGNN 网络进行前向传播
        feats, denoised_coords = net(seq, noised_coords, adj_mat = adj_mat, mask = masks)

        # 计算损失
        loss = F.mse_loss(denoised_coords[masks], coords[masks])

        # 反向传播并更新梯度
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    # 打印损失值
    print('loss:', loss.item())
    # 更新优化器
    optim.step()
    # 清空梯度
    optim.zero_grad()
```