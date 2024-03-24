# `.\lucidrains\VN-transformer\denoise.py`

```
# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的函数库
import torch.nn.functional as F
# 从 torch.optim 模块中导入 Adam 优化器
from torch.optim import Adam

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat

# 导入 sidechainnet 库，并从 VN_transformer 模块中导入 VNTransformer 类
import sidechainnet as scn
from VN_transformer import VNTransformer

# 定义常量 BATCH_SIZE
BATCH_SIZE = 1
# 定义常量 GRADIENT_ACCUMULATE_EVERY
GRADIENT_ACCUMULATE_EVERY = 16
# 定义常量 MAX_SEQ_LEN
MAX_SEQ_LEN = 256
# 定义默认数据类型 DEFAULT_TYPE
DEFAULT_TYPE = torch.float64

# 设置 PyTorch 默认数据类型为 DEFAULT_TYPE
torch.set_default_dtype(DEFAULT_TYPE)

# 定义一个循环生成器函数 cycle，用于生成数据
def cycle(loader, len_thres = MAX_SEQ_LEN):
    while True:
        for data in loader:
            # 如果数据的序列长度大于 len_thres，则继续循环
            if data.seqs.shape[1] > len_thres:
                continue
            # 生成数据
            yield data

# 创建 VNTransformer 模型对象
transformer = VNTransformer(
    num_tokens = 24,
    dim = 64,
    depth = 4,
    dim_head = 64,
    heads = 8,
    dim_feat = 64,
    bias_epsilon = 1e-6,
    l2_dist_attn = True,
    flash_attn = False
).cuda()

# 加载数据集
data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = BATCH_SIZE,
    dynamic_batching = False    
)

# 创建数据生成器 dl
dl = cycle(data['train'])
# 初始化 Adam 优化器
optim = Adam(transformer.parameters(), lr = 1e-4)

# 进行训练循环
for _ in range(10000):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        # 获取一个 batch 的数据
        batch = next(dl)
        seqs, coords, masks = batch.seqs, batch.crds, batch.msks

        # 将序列数据转移到 GPU 上，并取最大值作为索引
        seqs = seqs.cuda().argmax(dim = -1)
        # 将坐标数据转移到 GPU 上，并设置数据类型为默认类型
        coords = coords.cuda().type(torch.get_default_dtype())
        # 将掩码数据转移到 GPU 上，并转换为布尔类型
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

        # 运行 Transformer 模型
        type1_out, _ = transformer(
            noised_coords,
            feats = seq,
            mask = masks
        )

        # 去噪后的坐标数据
        denoised_coords = noised_coords + type1_out

        # 计算均方误差损失
        loss = F.mse_loss(denoised_coords[masks], coords[masks]) 
        # 反向传播并计算梯度
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    # 输出当前损失值
    print('loss:', loss.item())
    # 更新优化器参数
    optim.step()
    # 清空梯度
    optim.zero_grad()
```