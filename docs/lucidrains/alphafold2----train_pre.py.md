# `.\lucidrains\alphafold2\train_pre.py`

```py
# 导入所需的库
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange

# 导入自定义库
import sidechainnet as scn
from alphafold2_pytorch import Alphafold2
import alphafold2_pytorch.constants as constants
from alphafold2_pytorch.utils import get_bucketed_distance_matrix

# 常量定义

DEVICE = None # 默认为 cuda（如果可用），否则为 cpu
NUM_BATCHES = int(1e5)
GRADIENT_ACCUMULATE_EVERY = 16
LEARNING_RATE = 3e-4
IGNORE_INDEX = -100
THRESHOLD_LENGTH = 250

# 设置设备

DISTOGRAM_BUCKETS = constants.DISTOGRAM_BUCKETS
DEVICE = constants.DEVICE

# 辅助函数

def cycle(loader, cond = lambda x: True):
    # 无限循环遍历数据加载器
    while True:
        for data in loader:
            if not cond(data):
                continue
            yield data

# 获取数据

# 加载数据集
data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = 1,
    dynamic_batching = False
)

# 获取训练数据集的迭代器
data = iter(data['train'])
data_cond = lambda t: t[1].shape[1] < THRESHOLD_LENGTH
dl = cycle(data, data_cond)

# 模型

# 初始化 Alphafold2 模型
model = Alphafold2(
    dim = 256,
    depth = 1,
    heads = 8,
    dim_head = 64
).to(DEVICE)

# 优化器

# 初始化 Adam 优化器
optim = Adam(model.parameters(), lr = LEARNING_RATE)

# 训练循环

# 循环执行指定次数的训练批次
for _ in range(NUM_BATCHES):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        # 获取下一个数据批次
        batch = next(dl)
        seq, coords, mask = batch.seqs, batch.crds, batch.msks

        b, l, _ = seq.shape

        # 准备 mask 和 labels

        # 将序列、坐标和 mask 转换为指定设备上的张量
        seq, coords, mask = seq.argmax(dim = -1).to(DEVICE), coords.to(DEVICE), mask.to(DEVICE).bool()
        coords = rearrange(coords, 'b (l c) d -> b l c d', l = l)

        # 获取离散化的距离矩阵
        discretized_distances = get_bucketed_distance_matrix(coords[:, :, 1], mask, DISTOGRAM_BUCKETS, IGNORE_INDEX)

        # 预测

        distogram = model(seq, mask = mask)
        distogram = rearrange(distogram, 'b i j c -> b c i j')

        # 计算损失

        loss = F.cross_entropy(
            distogram,
            discretized_distances,
            ignore_index = IGNORE_INDEX
        )

        # 反向传播
        loss.backward()

    # 打印损失值
    print('loss:', loss.item())

    # 更新优化器参数
    optim.step()
    optim.zero_grad()
```