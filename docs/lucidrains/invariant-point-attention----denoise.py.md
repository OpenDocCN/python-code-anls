# `.\lucidrains\invariant-point-attention\denoise.py`

```
# 导入所需的库
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.optim import Adam

# 导入 einops 库中的函数
from einops import rearrange, repeat
# 导入 sidechainnet 库
import sidechainnet as scn
# 导入自定义的模块 invariant_point_attention 中的 IPATransformer 类
from invariant_point_attention import IPATransformer

# 定义批处理大小和梯度累积次数
BATCH_SIZE = 1
GRADIENT_ACCUMULATE_EVERY = 16

# 定义一个循环生成器函数，用于处理数据加载器中的数据
def cycle(loader, len_thres = 200):
    while True:
        for data in loader:
            # 如果序列长度超过阈值，则跳过
            if data.seqs.shape[1] > len_thres:
                continue
            yield data

# 创建 IPATransformer 模型实例
net = IPATransformer(
    dim = 16,
    num_tokens = 21,
    depth = 5,
    require_pairwise_repr = False,
    predict_points = True
).cuda()

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
# 初始化 Adam 优化器
optim = Adam(net.parameters(), lr=1e-3)

# 迭代训练模型
for _ in range(10000):
    # 梯度累积
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        # 获取一个批次的数据
        batch = next(dl)
        seqs, coords, masks = batch.seqs, batch.crds, batch.msks

        # 将序列转移到 GPU 并获取最大值索引
        seqs = seqs.cuda().argmax(dim = -1)
        coords = coords.cuda()
        masks = masks.cuda().bool()

        # 获取序列长度并重新排列坐标
        l = seqs.shape[1]
        coords = rearrange(coords, 'b (l s) c -> b l s c', s = 14)

        # 仅保留 Ca 原子坐标
        coords = coords[:, :, 1, :]
        # 添加随机噪声
        noised_coords = coords + torch.randn_like(coords)

        # 输入模型进行去噪处理
        denoised_coords = net(
            seqs,
            translations = noised_coords,
            mask = masks
        )

        # 计算损失
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