# `.\lucidrains\PaLM-rlhf-pytorch\palm_rlhf_pytorch\reward.py`

```
# 导入必要的库
import copy
from pathlib import Path

from tqdm import tqdm
from beartype import beartype
from beartype.typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from palm_rlhf_pytorch.utils import masked_mean, gumbel_sample
from palm_rlhf_pytorch.palm import PaLM

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 奖励模型 - 带有标量头的 PaLM

@beartype
class RewardModel(nn.Module):
    def __init__(
        self,
        palm: PaLM,
        dropout = 0.1,
        num_binned_output = 0.,
        use_lora = True,
        lora_r = 8,
        reward_lora_scope = 'reward',
    ):
        super().__init__()

        # 深拷贝传入的 PaLM 模型
        self.palm = copy.deepcopy(palm)
        self.palm.set_dropout(dropout)

        # 根据 use_lora 参数决定是否使用 LORA
        self.reward_lora_scope = reward_lora_scope if use_lora else None

        # 如果启用了 LORA，则为奖励模型添加微调参数
        if exists(self.reward_lora_scope):
            self.palm.add_finetune_params(reward_lora_scope, lora_r = lora_r)

        dim = palm.dim

        # 判断是否需要输出多个分箱
        self.binned_output = num_binned_output > 1

        # 初始化提示和响应的嵌入向量
        self.prompt_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.response_embed = nn.Parameter(torch.zeros(1, 1, dim))

        # 根据是否需要多个分箱选择不同的输出层
        if self.binned_output:
            self.to_pred = nn.Linear(dim, num_binned_output)
        else:
            self.to_pred = nn.Sequential(
                nn.Linear(dim, 1, bias = False),
                Rearrange('... 1 -> ...')
            )

    # 加载模型参数
    def load(self, path):
        path = Path(path)
        assert path.exists()
        self.load_state_dict(torch.load(str(path)))

    # 获取需要微调的参数
    def finetune_parameters(self):
        return [
            *self.to_pred.parameters(),
            *(self.palm.finetune_parameters(self.reward_lora_scope) if exists(self.reward_lora_scope) else self.palm.parameters())
        ]

    # 前向传播函数
    def forward(
        self,
        x,
        mask = None,
        prompt_mask = None,
        prompt_lengths = None,
        labels = None,
        sample = False,
        sample_temperature = 1.,
        disable_lora = False
    ):

        assert not (exists(prompt_mask) and exists(prompt_lengths))

        # 从提示长度中推���提示掩码
        if exists(prompt_lengths):
            batch, seq_len = x.shape
            arange = torch.arange(seq_len, device = x.device)
            prompt_mask = repeat(arange, 'n -> b n', b = batch) < rearrange(prompt_lengths, 'b -> b 1')

        # 奖励模型应该了解哪部分是提示，哪部分是响应

        extra_embed = None

        if exists(prompt_mask):
            extra_embed = torch.where(
                rearrange(prompt_mask, 'b n -> b n 1'),
                self.prompt_embed,
                self.response_embed
            )

        # 从 PaLM 中获取嵌入向量
        embeds = self.palm(
            x,
            extra_embed = extra_embed,
            return_only_embedding = True,
            disable_lora = disable_lora,
            finetune_scope = self.reward_lora_scope
        )

        # 对嵌入向量进行平均池化
        pooled = masked_mean(embeds, mask, dim = 1)
        pred = self.to_pred(pooled)

        # 如果需要采样并且输出为多个分箱，则对输出进行 Gumbel 采样
        if sample and self.binned_output:
            assert not exists(labels)
            pred = gumbel_sample(pred, temperature = sample_temperature, dim = -1)

        # 如果标签不存在，则直接返回预测值
        if not exists(labels):
            return pred

        # 如果输出不是多个分箱，则计算均方误差损失
        if not self.binned_output:
            return F.mse_loss(pred, labels)

        # 如果输出为多个分箱，则计算交叉熵损失
        return F.cross_entropy(pred, labels)
```