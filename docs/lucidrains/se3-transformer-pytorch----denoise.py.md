# `.\lucidrains\se3-transformer-pytorch\denoise.py`

```
# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的函数库
import torch.nn.functional as F
# 从 torch.optim 中导入 Adam 优化器
from torch.optim import Adam

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat

# 导入 sidechainnet 库，并从 se3_transformer_pytorch 中导入 SE3Transformer 类
import sidechainnet as scn
from se3_transformer_pytorch.se3_transformer_pytorch import SE3Transformer

# 设置默认的数据类型为 float64
torch.set_default_dtype(torch.float64)

# 定义批量大小为 1
BATCH_SIZE = 1
# 定义每隔多少次梯度累积
GRADIENT_ACCUMULATE_EVERY = 16

# 定义一个循环函数，用于处理数据加载器
def cycle(loader, len_thres = 500):
    while True:
        for data in loader:
            # 如果数据序列长度大于指定阈值，则继续循环
            if data.seqs.shape[1] > len_thres:
                continue
            yield data

# 创建 SE3Transformer 模型
transformer = SE3Transformer(
    num_tokens = 24,
    dim = 8,
    dim_head = 8,
    heads = 2,
    depth = 2,
    attend_self = True,
    input_degrees = 1,
    output_degrees = 2,
    reduce_dim_out = True,
    differentiable_coors = True,
    num_neighbors = 0,
    attend_sparse_neighbors = True,
    num_adj_degrees = 2,
    adj_dim = 4,
    num_degrees=2,
)

# 加载数据集
data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = BATCH_SIZE,
    dynamic_batching = False
)

# 创建数据加载器
dl = cycle(data['train'])
# 使用 Adam 优化器来优化 SE3Transformer 模型的参数
optim = Adam(transformer.parameters(), lr=1e-4)
# 将模型转移到 GPU 上
transformer = transformer.cuda()

# 进行训练循环
for _ in range(10000):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        # 获取一个批次的数据
        batch = next(dl)
        seqs, coords, masks = batch.seqs, batch.crds, batch.msks

        # 将序列转移到 GPU 上，并取最大值索引
        seqs = seqs.cuda().argmax(dim = -1)
        # 将坐标转移到 GPU 上，并设置数据类型为 float64
        coords = coords.cuda().type(torch.float64)
        # 将掩码转移到 GPU 上，并设置数据类型为布尔型
        masks = masks.cuda().bool()

        # 获取序列长度
        l = seqs.shape[1]
        # 重新排列坐标数据
        coords = rearrange(coords, 'b (l s) c -> b l s c', s = 14)

        # 保留骨架坐标
        coords = coords[:, :, 0:3, :]
        coords = rearrange(coords, 'b l s c -> b (l s) c')

        # 重复序列和掩码
        seq = repeat(seqs, 'b n -> b (n c)', c = 3)
        masks = repeat(masks, 'b n -> b (n c)', c = 3)

        # 添加高斯噪声到坐标数据
        noised_coords = coords + torch.randn_like(coords).cuda()

        # 创建邻接矩阵
        i = torch.arange(seq.shape[-1], device = seqs.device)
        adj_mat = (i[:, None] >= (i[None, :] - 1)) & (i[:, None] <= (i[None, :] + 1))

        # 使用 SE3Transformer 进行前向传播
        out = transformer(
            seq,
            noised_coords,
            mask = masks,
            adj_mat = adj_mat,
            return_type = 1
        )

        # 对去噪后的坐标数据计算均方误差损失
        denoised_coords = noised_coords + out
        loss = F.mse_loss(denoised_coords[masks], coords[masks]) 
        # 反向传播
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    # 输出损失值
    print('loss:', loss.item())
    # 更新优化器
    optim.step()
    # 梯度清零
    optim.zero_grad()
```