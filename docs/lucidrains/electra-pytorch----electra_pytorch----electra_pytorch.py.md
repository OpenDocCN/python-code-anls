# `.\lucidrains\electra-pytorch\electra_pytorch\electra_pytorch.py`

```py
# 导入数学库
import math
# 导入 reduce 函数
from functools import reduce
# 导入 namedtuple 类
from collections import namedtuple

# 导入 torch 库
import torch
# 导入 torch 中的 nn 模块
from torch import nn
# 导入 torch 中的 functional 模块
import torch.nn.functional as F

# 定义一个命名元组 Results，包含多个字段
Results = namedtuple('Results', [
    'loss',
    'mlm_loss',
    'disc_loss',
    'gen_acc',
    'disc_acc',
    'disc_labels',
    'disc_predictions'
])

# 定义一些辅助函数

# 计算输入张量的自然对数
def log(t, eps=1e-9):
    return torch.log(t + eps)

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 从 Gumbel 分布中采样
def gumbel_sample(t, temperature = 1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)

# 根据概率生成掩码
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

# 使用特定的标记生成掩码
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

# 根据概率获取子集掩码
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

# 隐藏层提取器类，用于为语言模型添加适配器以进行预训练

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

# Electra 主类

class Electra(nn.Module):
    # 初始化函数，接受生成器、判别器等参数
    def __init__(
        self,
        generator,
        discriminator,
        *,
        num_tokens = None,  # 可选参数：标记数量，默认为 None
        discr_dim = -1,  # 判别器维度，默认为 -1
        discr_layer = -1,  # 判别器层，默认为 -1
        mask_prob = 0.15,  # 掩码概率，默认为 0.15
        replace_prob = 0.85,  # 替换概率，默认为 0.85
        random_token_prob = 0.,  # 随机标记概率，默认为 0
        mask_token_id = 2,  # 掩码标记 ID，默认为 2
        pad_token_id = 0,  # 填充标记 ID，默认为 0
        mask_ignore_token_ids = [],  # 忽略的掩码标记 ID 列表，默认为空
        disc_weight = 50.,  # 判别器权重，默认为 50
        gen_weight = 1.,  # 生成器权重，默认为 1
        temperature = 1.):  # 温度参数，默认为 1
        super().__init__()  # 调用父类的初始化函数

        self.generator = generator  # 初始化生成器
        self.discriminator = discriminator  # 初始化判别器

        if discr_dim > 0:  # 如果判别器维度大于 0
            self.discriminator = nn.Sequential(  # 使用判别器的特定层
                HiddenLayerExtractor(discriminator, layer = discr_layer),  # 提取特定层的隐藏层
                nn.Linear(discr_dim, 1)  # 添加线性层
            )

        # mlm 相关概率
        self.mask_prob = mask_prob  # 掩码概率
        self.replace_prob = replace_prob  # 替换概率

        self.num_tokens = num_tokens  # 标记数量
        self.random_token_prob = random_token_prob  # 随机标记概率

        # 标记 ID
        self.pad_token_id = pad_token_id  # 填充标记 ID
        self.mask_token_id = mask_token_id  # 掩码标记 ID
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])  # 忽略的掩码标记 ID 集合

        # 采样温度
        self.temperature = temperature  # 温度参数

        # 损失权重
        self.disc_weight = disc_weight  # 判别器权重
        self.gen_weight = gen_weight  # 生成器权重
    # 定义前向传播函数，接受输入和其他参数
    def forward(self, input, **kwargs):
        # 获取输入张量的形状
        b, t = input.shape

        # 根据输入张量生成一个与其形状相同的概率掩码，用于替换概率
        replace_prob = prob_mask_like(input, self.replace_prob)

        # 创建一个不需要掩码的标记列表，包括 [pad] 标记和其他指定排除的标记（如 [cls], [sep]）
        no_mask = mask_with_tokens(input, self.mask_ignore_token_ids)
        # 根据概率获取需要掩码的子集
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # 获取需要掩码的索引
        mask_indices = torch.nonzero(mask, as_tuple=True)

        # 使用掩码标记的标记替换为 [mask] 标记，保留标记不变
        masked_input = input.clone().detach()

        # 将掩码的标记替换为填充标记，用于生成标签
        gen_labels = input.masked_fill(~mask, self.pad_token_id)

        # 克隆掩码，用于可能的随机标记修改
        masking_mask = mask.clone()

        # 如果随机标记概率大于0，用于 MLM
        if self.random_token_prob > 0:
            assert self.num_tokens is not None, 'Number of tokens (num_tokens) must be passed to Electra for randomizing tokens during masked language modeling'

            # 根据概率生成随机标记
            random_token_prob = prob_mask_like(input, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, input.shape, device=input.device)
            random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_input = torch.where(random_token_prob, random_tokens, masked_input)

            # 从掩码中移除随机标记概率掩码
            masking_mask = masking_mask & ~random_token_prob

        # 将掩码的标记替换为 [mask] 标记
        masked_input = masked_input.masked_fill(masking_mask * replace_prob, self.mask_token_id)

        # 获取生成器输出和 MLM 损失
        logits = self.generator(masked_input, **kwargs)

        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            gen_labels,
            ignore_index = self.pad_token_id
        )

        # 使用之前的掩码选择需要采样的 logits
        sample_logits = logits[mask_indices]

        # 采样
        sampled = gumbel_sample(sample_logits, temperature = self.temperature)

        # 将采样值散布回输入
        disc_input = input.clone()
        disc_input[mask_indices] = sampled.detach()

        # 生成鉴别器标签，替换为 True，原始为 False
        disc_labels = (input != disc_input).float().detach()

        # 获取替换/原始的鉴别器预测
        non_padded_indices = torch.nonzero(input != self.pad_token_id, as_tuple=True)

        # 获取鉴别器输出和二元交叉熵损失
        disc_logits = self.discriminator(disc_input, **kwargs)
        disc_logits = disc_logits.reshape_as(disc_labels)

        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits[non_padded_indices],
            disc_labels[non_padded_indices]
        )

        # 收集指标
        with torch.no_grad():
            gen_predictions = torch.argmax(logits, dim=-1)
            disc_predictions = torch.round((torch.sign(disc_logits) + 1.0) * 0.5)
            gen_acc = (gen_labels[mask] == gen_predictions[mask]).float().mean()
            disc_acc = 0.5 * (disc_labels[mask] == disc_predictions[mask]).float().mean() + 0.5 * (disc_labels[~mask] == disc_predictions[~mask]).float().mean()

        # 返回加权损失的结果
        return Results(self.gen_weight * mlm_loss + self.disc_weight * disc_loss, mlm_loss, disc_loss, gen_acc, disc_acc, disc_labels, disc_predictions)
```