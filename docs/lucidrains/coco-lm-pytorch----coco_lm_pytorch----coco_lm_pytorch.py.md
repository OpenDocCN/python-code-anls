# `.\lucidrains\coco-lm-pytorch\coco_lm_pytorch\coco_lm_pytorch.py`

```
# 导入数学库
import math
# 从 functools 库中导入 reduce 函数
from functools import reduce

# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch.nn.functional 库中导入 F 模块
import torch.nn.functional as F

# 辅助函数

# 计算输入张量的对数，加上一个很小的值 eps 防止出现对数值为负数的情况
def log(t, eps=1e-9):
    return torch.log(t + eps)

# 对输入张量进行 L2 归一化
def norm(t):
    return F.normalize(t, p = 2, dim = -1)

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 使用 Gumbel 噪声对输入张量进行采样
def gumbel_sample(t, temperature = 1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)

# 根据概率生成掩码
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

# 根据给定的标记 ID 列表生成掩码
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

# 根据概率生成子集掩码
def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

# 隐藏层提取器类，用于在语言模型中神奇地添加适配器以进行预训练

class HiddenLayerExtractor(nn.Module):
    def __init__(self, net, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = output

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def forward(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

# 主要的 Electra 类

class COCO(nn.Module):
    def __init__(
        self,
        generator,
        discriminator,
        *,
        discr_dim,
        num_tokens = None,
        discr_layer = -1,
        mask_prob = 0.15,
        replace_prob = 0.85,
        random_token_prob = 0.,
        pad_token_id = 0,
        cls_token_id = 1,
        mask_token_id = 2,
        mask_ignore_token_ids = [],
        disc_weight = 50.,
        gen_weight = 1.,
        cl_weight = 1.,
        temperature = 1.,
        crop_percentage = 0.5
        ):
        # 调用父类的构造函数
        super().__init__()

        # 初始化生成器和鉴别器
        self.generator = generator
        self.discriminator = discriminator

        # 提取鉴别器的隐藏层特征
        self.discriminator = HiddenLayerExtractor(discriminator, layer = discr_layer)
        # 将鉴别器的维度映射到1维
        self.to_correction_logits = nn.Linear(discr_dim, 1)

        # MLM相关的概率
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        # token的数量
        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob

        # token的id
        self.cls_token_id = cls_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id, cls_token_id])

        # 采样温度
        self.temperature = temperature

        # 损失权重
        self.disc_weight = disc_weight
        self.gen_weight = gen_weight
        self.cl_weight = cl_weight

        # Contrastive Loss的温度参数
        self.cl_temperature = nn.Parameter(torch.tensor(1.))

        # 裁剪百分比
        self.crop_percentage = crop_percentage
```