# `.\lucidrains\En-transformer\denoise.py`

```py
# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的函数库
import torch.nn.functional as F
# 从 torch 模块中导入 nn 模块
from torch import nn
# 从 torch.optim 模块中导入 Adam 优化器
from torch.optim import Adam

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 导入 sidechainnet 库并重命名为 scn
import sidechainnet as scn
# 从 en_transformer 模块中导入 EnTransformer 类
from en_transformer.en_transformer import EnTransformer

# 设置默认的张量数据类型为 float64
torch.set_default_dtype(torch.float64)

# 定义批量大小为 1
BATCH_SIZE = 1
# 定义每隔多少次梯度累积
GRADIENT_ACCUMULATE_EVERY = 16

# 定义一个循环函数，用于生成数据批次
def cycle(loader, len_thres = 200):
    while True:
        for data in loader:
            # 如果数据序列长度大于指定阈值，则继续循环
            if data.seqs.shape[1] > len_thres:
                continue
            # 生成数据
            yield data

# 创建 EnTransformer 模型实例
transformer = EnTransformer(
    num_tokens = 21,
    dim = 32,
    dim_head = 64,
    heads = 4,
    depth = 4,
    rel_pos_emb = True, # 序列中存在固有的顺序（氨基酸链的主干原子）
    neighbors = 16
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
# 使用 Adam 优化器来优化 EnTransformer 模型的参数
optim = Adam(transformer.parameters(), lr=1e-3)
# 将模型移动到 GPU 上
transformer = transformer.cuda()

# 进行训练循环
for _ in range(10000):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        # 获取一个数据批次
        batch = next(dl)
        seqs, coords, masks = batch.seqs, batch.crds, batch.msks

        # 将序列数据移动到 GPU 上并取最大值
        seqs = seqs.cuda().argmax(dim = -1)
        # 将坐标数据移动到 GPU 上并转换为 float64 类型
        coords = coords.cuda().type(torch.float64)
        # 将掩码数据移动到 GPU 上并转换为布尔类型
        masks = masks.cuda().bool()

        # 获取序列长度
        l = seqs.shape[1]
        # 重新排列坐标数据的维度
        coords = rearrange(coords, 'b (l s) c -> b l s c', s = 14)

        # 保留主干坐标

        coords = coords[:, :, 0:3, :]
        coords = rearrange(coords, 'b l s c -> b (l s) c')

        # 重复序列数据和掩码数据的维度
        seq = repeat(seqs, 'b n -> b (n c)', c = 3)
        masks = repeat(masks, 'b n -> b (n c)', c = 3)

        # 添加噪声到坐标数据
        noised_coords = coords + torch.randn_like(coords)

        # 使用 Transformer 模型进行特征提取和去噪
        feats, denoised_coords = transformer(seq, noised_coords, mask = masks)

        # 计算均方误差损失
        loss = F.mse_loss(denoised_coords[masks], coords[masks])

        # 反向传播并计算梯度
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    # 打印损失值
    print('loss:', loss.item())
    # 更新优化器
    optim.step()
    # 清空梯度
    optim.zero_grad()
```