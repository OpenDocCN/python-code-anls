# `.\lucidrains\equiformer-pytorch\denoise.py`

```
# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的函数库
import torch.nn.functional as F
# 从 torch.optim 中导入 Adam 优化器
from torch.optim import Adam

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat

# 导入 sidechainnet 库，并重命名为 scn
import sidechainnet as scn
# 从 equiformer_pytorch 中导入 Equiformer 类
from equiformer_pytorch import Equiformer

# 定义常量 BATCH_SIZE
BATCH_SIZE = 1
# 定义常量 GRADIENT_ACCUMULATE_EVERY
GRADIENT_ACCUMULATE_EVERY = 16
# 定义常量 MAX_SEQ_LEN
MAX_SEQ_LEN = 512
# 定义默认数据类型为 torch.float64
DEFAULT_TYPE = torch.float64

# 设置 PyTorch 默认数据类型为 DEFAULT_TYPE
torch.set_default_dtype(DEFAULT_TYPE)

# 定义一个循环生成器函数 cycle，用于生成数据
def cycle(loader, len_thres = MAX_SEQ_LEN):
    while True:
        for data in loader:
            if data.seqs.shape[1] > len_thres:
                continue
            yield data

# 创建 Equiformer 模型对象，并将其移动到 GPU 上
transformer = Equiformer(
    num_tokens = 24,
    dim = (16, 8, 8, 8),
    dim_head = (16, 8, 8, 8),
    heads = (4, 2, 2, 2),
    depth = 10,
    reversible = True,
    attend_self = True,
    reduce_dim_out = True,
    num_neighbors = 6,
    num_degrees = 4,
    linear_out = True
).cuda()

# 加载数据集
data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = BATCH_SIZE,
    dynamic_batching = False    
)

# 创建数据循环对象 dl
dl = cycle(data['train'])
# 创建 Adam 优化器对象 optim
optim = Adam(transformer.parameters(), lr = 1e-4)

# 循环训练模型
for _ in range(10000):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        # 获取一个 batch 的数据
        batch = next(dl)
        seqs, coords, masks = batch.seqs, batch.crds, batch.msks

        # 将序列数据移动到 GPU 上，并取最大值索引
        seqs = seqs.cuda().argmax(dim = -1)
        # 将坐标数据移动到 GPU 上，并设置数据类型为默认类型
        coords = coords.cuda().type(torch.get_default_dtype())
        # 将掩码数据移动到 GPU 上，并转换为布尔类型
        masks = masks.cuda().bool()

        # 获取序列长度
        l = seqs.shape[1]
        # 重新排列坐标数据的维度
        coords = rearrange(coords, 'b (l s) c -> b l s c', s = 14)

        # 保留主干坐标
        coords = coords[:, :, 0:3, :]
        coords = rearrange(coords, 'b l s c -> b (l s) c')

        # 将序列数据重复为坐标数据的维度
        seq = repeat(seqs, 'b n -> b (n c)', c = 3)
        masks = repeat(masks, 'b n -> b (n c)', c = 3)

        # 给坐标数据添加高斯噪声
        noised_coords = coords + torch.randn_like(coords).cuda()

        # 运行 Equiformer 模型
        _, type1_out = transformer(
            seq,
            noised_coords,
            mask = masks
        )

        # 对添加噪声后的坐标数据进行去噪
        denoised_coords = noised_coords + type1_out

        # 计算坐标数据的均方误差损失
        loss = F.mse_loss(denoised_coords[masks], coords[masks]) 
        # 反向传播并计算梯度
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    # 打印损失值
    print('loss:', loss.item())
    # 更新优化器参数
    optim.step()
    # 梯度清零
    optim.zero_grad()
```